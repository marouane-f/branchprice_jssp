import math
import sys
from collections import defaultdict

from shortest_path import recover_path
from shortest_path._shortest_path_c import compute_shortest_path

from utils import JobSchedule, timing_store, timeit_accumulate

from .usage_ext import update_usage_with_check_untimed

sys.path.extend(['../'])

import time
import numpy as np

from pyscipopt import Heur, SCIP_RESULT
from functools import wraps

# Shared timing store dictionary for total time
timing_store_rmp = {
    "search": 0.0,
    "removal": 0,
    "mach_lim": 0,
    "mach_us": 0,
    "lb_conflict": 0,
    "pairwise_conflicts": 0,
}

# Call count store dictionary
call_counts_rmp = {
    "search": 0,
    "removal": 0,
    "mach_lim": 0,
    "mach_us": 0,
    "lb_conflict": 0,
    "pairwise_conflicts": 0,
}


def reset_timing_store_rmp():
    for k, v in call_counts_rmp.items():
        timing_store_rmp[k] = type(v)()  # 0.0 for float, 0 for int
    for k, v in call_counts_rmp.items():
        call_counts_rmp[k] = type(v)()  # 0.0 for float, 0 for int


def timeit_accumulate_rmp(timer_container, counter_container, key, recursive=False):
    if recursive:
        depth = {"val": 0}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if recursive:
                if depth["val"] == 0:
                    start = time.perf_counter()
                depth["val"] += 1

                result = func(*args, **kwargs)

                depth["val"] -= 1
                if depth["val"] == 0:
                    elapsed = time.perf_counter() - start
                    timer_container[key] += elapsed
                    counter_container[key] += 1

                return result

            else:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                timer_container[key] += elapsed
                counter_container[key] += 1
                return result

        return wrapper

    return decorator


@timeit_accumulate_rmp(timing_store_rmp, call_counts_rmp, "mach_lim")
def update_usage_with_check(t_list, usage_counts, max_active):
    """
    add col to usage_counts one coord after the other, stop when infeasible and backtrack otherwise return True with updated usage_counts
    """
    # if usage_counts[t_list].max() >= max_active:
    #     return True
    # usage_counts[t_list] += 1
    # return False
    return update_usage_with_check_untimed(t_list, usage_counts, max_active)


@timeit_accumulate_rmp(timing_store_rmp, call_counts_rmp, "lb_conflict")
def compute_conflict_lb(remaining_jobs, all_jobs, forbidden_mt, usage_counts, max_mach):
    """
    at a certain node of the search tree, compute a conflict-aware lb
    checking all remaining jobs: for each job, the best column is the min-cost column compatible with given state

    lb = cost(partial_state[:job_idx]) + \sum_i {i \in [job_idx:] | col_i is compatible with partial_state} cost_i      
    """

    # Cython version
    # return compute_conflict_lb_untimed(remaining_jobs, all_jobs, forbidden_mt, usage_counts, max_mach)

    # check if number of chosen jobs thus far is more than max_mach.
    # otherwise, we are necessarily less than max_mach.
    jobs_at_max_mach = len(all_jobs) - len(remaining_jobs) >= max_mach  # if False, no need to check for usage_counts.
    max_mach_1 = max_mach - 1

    total_lb = 0.0
    for job in remaining_jobs:
        curr_cols = all_jobs[job]
        for col in curr_cols:  # assumed to be sorted (asc.) by col.cost
            if forbidden_mt & col.mt_pairs_bitset: # if not disjoint
                continue
            if np.all(usage_counts[col.time_steps_flat_list] <= max_mach_1):
                total_lb += col.cost
                break
            if not jobs_at_max_mach:
                total_lb += col.cost
                break
        else:
            return None  # no feasible column found for this job
    return total_lb


class EarlyTerminate(Exception):
    pass


class PartialSolution:
    def __init__(self, chosen_cols, cost):
        self.chosen_cols = chosen_cols
        self.cost = cost
        self.jobs = [x.job_id for x in chosen_cols]


class RMPTreeSearchHeuristic(Heur):
    """
    RENS
            Most MIPs feature LP optima that can be efficiently rounded.
            The ability to round these solutions is not influenced by the fractionality of the LP solution.
            Rounding heuristics often fail to find feasible solutions, even with roundable starting points, and rarely achieve the optimal rounding

    RINS
            Defines a neighborhood by comparing the solution of the LP relaxation with an initial feasible solution and fixing common variables
    """

    def __init__(self, jobs, machines, C_max, duration, Cost_t, powers_m, sequence, KP_ineq, rc_term,
                 all_constraints,
                 all_schedules_jmt, all_vars_jmt, all_varschedules, all_vars_jobs, all_schedules_jobs,
                 price_i, priced_sols, schedule_id_to_var,
                 forbidden_arcs, forced_arcs, total_added_patterns,
                 rounding_mode, verbose, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # problem data 
        self.J = jobs
        self.M = machines
        self.M_max = KP_ineq[0].rhs
        self.C_max = C_max
        self.dur = duration
        self.C_t = Cost_t
        self.W_m = powers_m
        self.total_dur = np.sum(np.array(self.dur), axis=1)
        self.seq = sequence
        self.KP = KP_ineq
        self.rc_term = rc_term

        # duals 
        self.part_c = all_constraints[0]
        self.disj_c = all_constraints[1]
        self.res_c = all_constraints[2]

        # variables and schedules
        self.all_schedules_jmt = all_schedules_jmt
        self.all_vars_jmt = all_vars_jmt
        self.all_varschedules = all_varschedules
        self.all_vars_jobs = all_vars_jobs
        self.all_schedules_jobs = all_schedules_jobs

        # algorithm/logic utilities
        self.usage_counts = None
        self.max_t = None
        self.total_mt_capacity = None
        self.verbose = verbose
        self.rounding_mode = rounding_mode
        self.run_J2_extensions_bool = True         # TODO: add arg
        self.rc_fixing_bool = False         # Disabled by default, TODO: needs testing
        self.total_added_patterns = total_added_patterns
        self.best_cost = None
        self.best_solution = None
        self.hist = [False, False]

        # logging
        self.first_feas = sum(self.hist)
        self.processed_nodes = set()
        self.nr_expl_nd = 0
        self.nr_fathom_nd = 0
        self.i = 0
        self.node_count = 0

        # pricer logging
        self.price_i = price_i
        self.priced_sols = priced_sols
        self.schedule_id_to_var = schedule_id_to_var

        # branching constraints
        self.forbidden_arcs = forbidden_arcs
        self.forced_arcs = forced_arcs

        # tree search / lns logging
        self.partial_solutions = defaultdict(list)
        self.store_partial_bool = True
        self.best_Z_jmt = None
        self.best_sol_cols = None
        self.curr_best = float('inf')

        # primal/dual bound logging
        self.primal_progress = []
        self.dual_progress = []

        # early exit/stop utilities
        self.max_expl_nd = 1e6              # TODO: argument? (max explored nodes in tree search/dive)
        self.max_improv_stop = 10           # TODO: argument? (successive number of improvements in LNS before stop)
        self.max_no_primal_ch = 300         # TODO: argument? (primal bd no change over xx nodes)
        self.stop = False
        self.interrupt = False

    @timeit_accumulate(timing_store["RMP_Heur"])
    def heurexec(self, heurtiming, nodeinfeasible):

        # --- stop checks ---   # TODO: add a reasonable stopping condition (e.g. no improvements after 5 calls, or dual vs primal bound movement)
        if self.stop:
            self.vprint("--  Stop flag!")
            return {"result": SCIP_RESULT.DIDNOTRUN}
        if self._should_stop_no_progress():
            return {"result": SCIP_RESULT.DIDNOTRUN}

        n_J = len(self.J)

        # --- current node and bound updates ---
        curr_node, curr_node_n, node_depth, not_root_bool = self._get_node_info()
        primal_bd, dual_bd, primal_finite, incumb_cols = self._update_bounds()

        # --- decide which operations to run this iteration ---
        run_mode = self._decide_run_mode(curr_node_n, node_depth, not_root_bool,
                                         primal_bd, dual_bd, primal_finite, incumb_cols, n_J)
        if run_mode is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        run_tree, run_DR, rens, fix, fixing_bool, subtree_lb, incumb = run_mode

        # --- setup search state and guard conditions ---
        early = self._setup_search(curr_node_n, primal_finite)
        if early is not None:
            return early

        # --- build per-job filtered column sets ---
        all_jobs, starting_cols, early = self._filter_columns(
            heurtiming, not_root_bool, fixing_bool, primal_bd, subtree_lb, rens, run_tree,
        )
        if early is not None:
            return early

        # --- tree search ---
        if run_tree + fix + rens:
            self.vprint(f"rounding mode: {self.rounding_mode}")
            nr_avail_idx = np.argsort(np.array([len(x) for x in self.all_varschedules.values()]))
            self.find_best_schedule(starting_cols, nr_avail_idx, all_jobs, incumb, subtree_lb, curr_node_n)
        incumb = min(self.best_cost, incumb) if incumb is not None else self.best_cost

        # --- destroy & repair on partial solutions ---
        new_vars, new_schedules, new_sch_costs = [], [], []
        if run_DR:
            for k in self.partial_solutions:
                self.partial_solutions[k] = sorted(self.partial_solutions[k], key=lambda x: x.cost)
            new_vars, new_schedules, new_sch_costs = self._run_destroy_repair(n_J, incumb)
            self.partial_solutions.clear()

        # --- commit found solutions to the model ---
        return self._commit_results(new_vars, new_schedules, new_sch_costs)

    # -----------------------------------------------------------------------
    # heurexec sub-steps
    # -----------------------------------------------------------------------

    def _should_stop_no_progress(self) -> bool:
        """Return True (and set stop flag) if primal bound has not changed for max_no_primal_ch calls."""
        if self.node_count > self.max_no_primal_ch + 1:
            if all(x == self.best_cost for x in self.primal_progress[-self.max_no_primal_ch:]):
                self.stop = True
                self.vprint("--  Stop triggered!")
                return True
        return False

    def _get_node_info(self):
        """Return (curr_node, curr_node_n, node_depth, not_root_bool) and increment node_count."""
        curr_node = self.model.getCurrentNode()
        curr_node_n = curr_node.getNumber()
        node_depth = curr_node.getDepth()
        not_root_bool = node_depth
        self.node_count += 1
        return curr_node, curr_node_n, node_depth, not_root_bool

    def _update_bounds(self):
        """Fetch primal/dual bounds, update incumbent, and append to progress logs."""
        primal_bd, incumb_cols = self.catch_new_incumbent()
        dual_bd = self.model.getDualbound()
        primal_finite = not self.model.isInfinity(primal_bd)
        self.primal_progress.append(primal_bd)
        self.dual_progress.append(dual_bd)
        return primal_bd, dual_bd, primal_finite, incumb_cols

    def _decide_run_mode(self, curr_node_n, node_depth, not_root_bool,
                         primal_bd, dual_bd, primal_finite, incumb_cols, n_J):
        """Determine which operations to run (tree search, D&R, RENS, fixing) and compute subtree_lb.
        Returns (run_tree, run_DR, rens, fix, fixing_bool, subtree_lb, incumb), or None if DIDNOTRUN."""
        run_tree = run_DR = rens = fix = 0

        if primal_finite:
            self.best_cost = primal_bd
            curr_gap = 100.0 * (primal_bd - dual_bd) / dual_bd
            if curr_gap > 1.0 and self.node_count % 50 == 1:
                run_tree += 1
            else:
                # gap < 1: build J-1 partials from incumbent for D&R
                self.partial_solutions[n_J - 1] = []
                for i in range(n_J):
                    partial_cols = incumb_cols[:i] + incumb_cols[i+1:]
                    partial_cost = sum(col.cost for col in partial_cols)
                    self.partial_solutions[n_J - 1].append(PartialSolution(partial_cols, partial_cost))
        else:
            run_tree += 1
            self.vprint(f"No primal finite | full search again at node#{self.node_count}")

        if self.node_count % 50 == 1:
            if run_tree:
                self.vprint(f"Running full search and D&R at node#{self.node_count}")
            else:
                self.vprint(f"Running D&R only at node#{self.node_count}")
            run_DR = 1
            self.rounding_mode = None

        if node_depth >= 1:
            if self.node_count == 2 or self.node_count % 11 == 0:
                rens = 1
                self.vprint(f"Running RENS at node#{self.node_count}")
                self.rounding_mode = 'RENS'
            if not run_DR + run_tree + rens:
                if primal_finite:
                    return None  # nothing to do this call
                if not run_tree:
                    self.rounding_mode = 'RENS'
                self.vprint(f"trying search at {curr_node_n} | depth={node_depth} | (rounding:{self.rounding_mode})")

        fixing_bool = self.rounding_mode in ('UNFIX', 'RENS')
        if fixing_bool:
            curr_node = self.model.getCurrentNode()
            subtree_lb = curr_node.getLowerbound()
            if self.model.isInfinity(subtree_lb):
                subtree_lb = max(curr_node.getParent().getLowerbound(), dual_bd)
        else:
            subtree_lb = dual_bd

        if primal_finite:
            self.first_feas += 1 * not_root_bool  # stop tree search at first improving past root if primal finite
            incumb = primal_bd
        else:
            incumb = None

        return run_tree, run_DR, rens, fix, fixing_bool, subtree_lb, incumb

    def _setup_search(self, curr_node_n, primal_finite):
        """Assert solver state, initialise per-call search data, guard against duplicate processing."""
        assert not self.model.inProbing()
        self.max_t = max(t for (_, _, t) in self.all_schedules_jmt.keys())
        self.total_mt_capacity = self.M_max * self.max_t
        self.usage_counts = np.zeros(self.max_t + 1, dtype=np.intp)
        reset_timing_store_rmp()

        if not self.enough_columns():
            if primal_finite:
                self.vprint("--  Not enough total columns!")
                return {"result": SCIP_RESULT.DIDNOTRUN}

        if curr_node_n in self.processed_nodes:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.processed_nodes.add(curr_node_n)
        return None

    def _filter_columns(self, heurtiming, not_root_bool, fixing_bool, primal_bd, subtree_lb,
                        rens_anyway, run_anyway_tree):
        """Build per-job column sets filtered by LP values, rc-fixing, and branching decisions.
        Returns (all_jobs, starting_cols, early_result) where early_result is non-None if DIDNOTRUN."""
        n_J = len(self.J)
        get_lp = lambda vs: self.model.getSolVal(sol=None, expr=self.model.getTransformedVar(vs.var))
        EQ_one = lambda vs: self.model.isEQ(get_lp(vs), 1)
        is_zero = lambda x: self.model.isEQ(x, 0)
        get_rc = self.model.getVarRedcost

        if self.rounding_mode == 'UNFIX':
            get_val_rounding = lambda vs: self.model.getTransformedVar(vs.var).getUbLocal()
        else:
            get_val_rounding = get_lp  # RENS or None — safe fallback

        all_jobs = {}
        starting_cols = []

        for job_idx, var_schedules in self.all_varschedules.items():
            if not_root_bool * fixing_bool:
                filtered = []
                for vs in var_schedules:
                    var = vs.var
                    sch = vs.schedule
                    if var.getUbGlobal() == 0.0:
                        continue
                    vs.schedule.rc = get_rc(var)
                    lp_val = get_lp(vs)
                    if is_zero(get_val_rounding(vs)):
                        continue
                    if EQ_one(vs):  # x_lp = 1
                        starting_cols.append(vs.schedule)
                    if self.rc_fixing_bool * (self.total_added_patterns == 0):  # add = 0 <-> dual optimal
                        if EQ_one(vs) and vs.schedule.rc > primal_bd - subtree_lb:  # x_lp = 1 and |rc| > z_ub - z_lb
                            starting_cols.append(vs.schedule)
                        elif is_zero(lp_val) or (vs.schedule.rc > primal_bd - subtree_lb and (not EQ_one(vs))):  # x_lp = 0 or |rc| > z_ub - z_lb (z_lp != 1)
                            continue
                    filtered.append((vs.schedule, vs.var))  # not fixed by rc and x_lp > 0
            elif heurtiming:
                filtered = []
                for vs in var_schedules:
                    var = vs.var
                    sch = vs.schedule
                    if var.getUbGlobal() == 0.0:
                        continue
                    if self.rc_fixing_bool:
                        vs.schedule.rc = get_rc(vs.var)
                        if abs(vs.schedule.rc) > primal_bd - subtree_lb and (not EQ_one(vs)):
                            continue
                        elif abs(vs.schedule.rc) > primal_bd - subtree_lb and EQ_one(vs):
                            starting_cols.append(vs.schedule)
                    filtered.append((sch, var))

            if heurtiming:
                columns, vars_ = zip(*filtered) if filtered else ([], [])

            all_jobs[job_idx] = sorted(columns, key=lambda s: s.cost)

        min_nr_j = min([len(x) for x in all_jobs.values()])
        if not rens_anyway + run_anyway_tree:
            if min_nr_j <= n_J // 4:  # TODO: better criterion?
                self.vprint("-- too few columns for certain jobs after filtering!")
                return all_jobs, starting_cols, {"result": SCIP_RESULT.DIDNOTRUN}
        if min_nr_j == 0:
            self.vprint("-- no columns for certain jobs after filtering!")
            return all_jobs, starting_cols, {"result": SCIP_RESULT.DIDNOTRUN}

        return all_jobs, starting_cols, None

    def _run_destroy_repair(self, n_J, incumb):
        """Run J-1 then (optionally) J-2 destroy-and-repair passes over stored partial solutions.
        Returns (new_vars, new_schedules, new_sch_costs)."""
        new_vars, new_schedules, new_sch_costs, best_cost, found_j1 = \
            self._run_J1_dr(n_J, incumb)

        # TODO: unstable, needs debugging (some partials have two columns of the same job)
        if self.run_J2_extensions_bool and not found_j1:
            self.vprint(f"  no (extensions from) [J-1]-partial solutions were found")
            extra_vars, extra_schedules, extra_costs = self._run_J2_dr(n_J, best_cost)
            new_vars += extra_vars
            new_schedules += extra_schedules
            new_sch_costs += extra_costs

        return new_vars, new_schedules, new_sch_costs

    def _run_J1_dr(self, n_J, incumb):
        """Extend J-1 partial solutions to full solutions via complementary pricing + iterative D&R.
        Returns (new_vars, new_schedules, new_sch_costs, best_cost, found_count)."""
        new_vars, new_schedules, new_sch_costs = [], [], []
        best_cost = incumb
        found_j1 = 0

        n_avail = len(self.partial_solutions[n_J - 1])
        if not n_avail:
            return new_vars, new_schedules, new_sch_costs, best_cost, found_j1

        n_lim = min(n_avail, 300 if self.curr_best > 1e10 else 200)
        self.vprint(f"-- Checking [J-1]-partial solutions: ({n_lim} out of {n_avail} available)")

        for i, part_sol in enumerate(self.partial_solutions[n_J - 1][:n_lim]):
            if i % (n_lim // 5 + 1) == 0:
                self.vprint(f"--    partial solution #{i}")

            sch, part_sol, tot_cost = self.price_and_check(part_sol)
            if sch:
                if tot_cost < best_cost:
                    self.vprint(f"\t \t -- improvement found ({found_j1+1})  {tot_cost} < {best_cost}")
                    new_vars.append(sch)
                    new_schedules.append((sch, part_sol))
                    new_sch_costs.append(tot_cost)
                    best_cost = tot_cost
                    found_j1 += 1
                    extra_check = False
                else:
                    extra_check = True

                # iterative destroy & repair on the completed solution
                full_solution = part_sol.chosen_cols + [sch]
                self.vprint(f"\t -- iterative destroy repair")
                new_sol, new_sol_split, new_tot_cost, all_new, improv_i = \
                    self.iterative_destroy_repair(solution=full_solution, best_incumb=best_cost)
                if improv_i and new_tot_cost < best_cost:
                    self.vprint(f"\t \t -- improvement found ({found_j1+1})  {new_tot_cost} < {best_cost}")
                    best_cost = new_tot_cost
                    found_j1 += 1
                    for col in new_sol:
                        if col in all_new:
                            new_vars.append(col)
                    new_schedules.append(new_sol_split)
                    new_sch_costs.append(new_tot_cost)
                    if not extra_check:
                        extra_check = True

                # the first variable was not added initially but served to find a new solution in D&R
                if extra_check:
                    new_vars.append(sch)

            if found_j1 >= self.max_improv_stop:
                break

        return new_vars, new_schedules, new_sch_costs, best_cost, found_j1

    def _run_J2_dr(self, n_J, best_cost):
        """Extend J-2 partial solutions to full solutions via successive complementary pricing + D&R.
        Returns (new_vars, new_schedules, new_sch_costs)."""
        new_vars, new_schedules, new_sch_costs = [], [], []
        found_j2 = 0

        n_avail = len(self.partial_solutions[n_J - 2])
        if not n_avail:
            self.vprint(f"-- No improving extensions from [J-2]-partial solutions were found")
            return new_vars, new_schedules, new_sch_costs

        n_lim = min(n_avail, 300 if self.curr_best > 1e10 else 200)
        self.vprint(f"-- Running Destroy-Repair&Price on [J-2] partial solutions (on {n_lim} out of {n_avail} available)")

        for i2, orig_partial in enumerate(self.partial_solutions[n_J - 2][:n_lim]):
            partial_jobs = orig_partial.jobs
            assert len(partial_jobs) == len(set(partial_jobs))

            if found_j2 >= self.max_improv_stop:
                break

            new_col, base_solution, _ = self.price_and_check(orig_partial)

            if new_col:
                # extend J-2 → J-1
                part_sol_j2 = PartialSolution(base_solution.chosen_cols + [new_col], base_solution.cost + new_col.cost)
                sch, part_sol, tot_cost = self.price_and_check(part_sol_j2)

                if sch:  # J-1 → full solution
                    if tot_cost < best_cost:
                        self.vprint(f"--            found a better incumbent using successive complementary pricing | {tot_cost}")
                        best_cost = tot_cost
                        new_vars.append(sch)
                        new_vars.append(new_col)
                        new_schedules.append((sch, part_sol))
                        new_sch_costs.append(tot_cost)
                        found_j2 += 1
                        if found_j2 >= self.max_improv_stop:
                            break
                else:
                    # full extension failed: try D&R on the J-1 partial
                    for jb in part_sol_j2.jobs:
                        sch2, part_sol2, tot_cost2 = self.destroy_repair(part_sol_j2.chosen_cols, jb)
                        if sch2:
                            part_sol_j1_repaired = PartialSolution(part_sol2.chosen_cols + [sch2], part_sol2.cost + sch2.cost)
                            sch1, part_sol1, tot_cost1 = self.price_and_check(part_sol_j1_repaired)
                            if sch1 and tot_cost1 < best_cost:
                                self.vprint(f"--            found a better incumbent using Destroy-Repair&Price | {tot_cost1}")
                                best_cost = tot_cost1
                                new_vars.extend([sch1, sch2, new_col])
                                new_schedules.append((sch1, part_sol1))
                                new_sch_costs.append(tot_cost1)
                                found_j2 += 1
                                if found_j2 >= self.max_improv_stop:
                                    break

            else:
                if found_j2 >= self.max_improv_stop:
                    break
                if i2 % (n_lim // 5 + 1) == 0:
                    self.vprint(f"-- [nested#{i2}]  Destroy-Repair&Price on [J-2] partials")

                for jb in orig_partial.jobs:
                    sch, part_sol, tot_cost = self.destroy_repair(orig_partial.chosen_cols, jb)
                    if sch:
                        part_sol_j2_repaired = PartialSolution(part_sol.chosen_cols + [sch], part_sol.cost + sch.cost)
                        sch1, part_sol1, tot_cost1 = self.price_and_check(part_sol_j2_repaired)
                        if sch1:
                            part_sol_j1 = PartialSolution(part_sol1.chosen_cols + [sch1], tot_cost1)
                            sch_full, part_sol_full, tot_cost_full = self.price_and_check(part_sol_j1)
                            if sch_full:
                                if tot_cost_full < best_cost:
                                    self.vprint(f"--    [nested#{i2}]  found a better incumbent using complementary pricing | {tot_cost_full}")
                                    best_cost = tot_cost_full
                                    new_vars.extend([sch, sch1, sch_full])
                                    new_schedules.append((sch_full, part_sol_full))
                                    new_sch_costs.append(tot_cost_full)
                                    found_j2 += 1
                            else:
                                # full extension failed: try D&R on the J-1 partial
                                for jb in part_sol_j1.jobs:
                                    sch2, part_sol2, _ = self.destroy_repair(part_sol_j1.chosen_cols, jb)
                                    if sch2:
                                        part_sol_j1_repaired = PartialSolution(part_sol2.chosen_cols + [sch2], part_sol2.cost + sch2.cost)
                                        sch_full, part_sol_full, tot_cost_full = self.price_and_check(part_sol_j1_repaired)
                                        if sch_full and tot_cost_full < best_cost:
                                            self.vprint(f"--    [nested#{i2}]  found a better incumbent using Destroy-Repair&Price | {tot_cost_full}")
                                            best_cost = tot_cost_full
                                            new_vars.extend([sch, sch1, sch2, sch_full])
                                            new_schedules.append((sch_full, part_sol_full))
                                            new_sch_costs.append(tot_cost_full)
                                            found_j2 += 1
                                    if found_j2 >= self.max_improv_stop:
                                        break
                    if found_j2 >= self.max_improv_stop:
                        break

        if not found_j2:
            self.vprint(f"-- No improving extensions from [J-2]-partial solutions were found")
            # TODO: Attempt to greedily price from [J-3] or [J-4] partial solutions

        return new_vars, new_schedules, new_sch_costs

    def _commit_results(self, new_vars, new_schedules, new_sch_costs):
        """Inject found schedules into the model via probing, or submit a tree-search solution."""
        if new_schedules:
            return self._inject_solutions(new_vars, new_schedules, new_sch_costs)
        elif self.best_solution:
            return self._submit_tree_solution()
        self.hist.append(False)
        return {"result": SCIP_RESULT.DIDNOTFIND}

    def _inject_solutions(self, new_vars, new_schedules, new_sch_costs):
        """Add new columns to the model via probing, then post the cheapest 2/3 solutions."""
        for x in new_vars:
            self.priced_sols.append(x)

        # enter pricing once via probing to register the new columns
        self.model.startProbing()
        self.model.setParam("lp/solvefreq", -1)  # disable LP solving
        self.model.solveProbingLPWithPricing(pretendroot=False, displayinfo=False, maxpricerounds=1)
        it_while = 0
        while self.priced_sols:
            it_while += 1
            self.vprint("[RMPTreeSearch] Warning: probing was not called! trying again...")
            self.model.solveProbingLPWithPricing(pretendroot=False, displayinfo=False, maxpricerounds=1)
            if it_while > 5:
                break
        self.model.setParam("lp/solvefreq", 1)
        self.model.endProbing()

        if self.priced_sols:
            self.vprint(f"[RMPTreeSearch] Warning: pricing was not called after {it_while} trials... aborting")
            return {"result": SCIP_RESULT.DIDNOTFIND}

        sch_to_var = self.schedule_id_to_var
        n = len(new_schedules)
        top_idx = np.argsort(new_sch_costs)[: (2 * n) // 3 + 1]  # cheapest 2/3

        for sol_id, sch_idx in enumerate(top_idx):
            to_add, part_sol = new_schedules[sch_idx]
            j = to_add.job_id
            sol = self.model.createSol(self)
            new_var = sch_to_var[j].get(to_add.unique_id)
            if new_var is None:
                self.vprint(f'Variable {to_add.unique_id} has not been added to the model')
                raise KeyError
            if new_var.getUbGlobal() == 0.0:
                continue
            self.vprint(f"Setting values in solution  | Obj = {to_add.cost + part_sol.cost}")
            self.model.setSolVal(sol, new_var, 1)
            self.vprint(f"Set {new_var.name} = 1")
            var_fixed_0 = 0
            for sch in part_sol.chosen_cols:
                j = sch.job_id
                var = sch_to_var[j].get(sch.unique_id)
                assert var
                if var.getUbGlobal() == 0.0:  # TODO: discuss this: how to do faster?
                    print("[RMPTreeSearch] Warning: found improving solution but existing variable fixed at 0... skipping solution")
                    var_fixed_0 = 1
                    break
                self.model.setSolVal(sol, var, 1)
                self.vprint(f"Set {var.name} = 1")
            if var_fixed_0:
                continue
            assert self.model.checkSol(sol)
            self.model.trySol(sol)
            if sol_id == 0:
                Z_jmt, _, _ = self.construct_Z_from_feas([to_add] + part_sol.chosen_cols)
                assert abs(np.sum(Z_jmt) - np.sum(np.array(self.dur))) < 1e-6
                self.best_Z_jmt = Z_jmt
                self.curr_best = to_add.cost + part_sol.cost
                self.best_sol_cols = [to_add] + part_sol.chosen_cols

        self.hist.append(True)
        return {"result": SCIP_RESULT.FOUNDSOL}

    def _submit_tree_solution(self):
        """Post the solution found by the tree search (no D&R) to the SCIP model."""
        sol = self.model.createSol(self)
        var_fixed_0 = 0
        for sch in self.best_solution:
            j_id = sch.job_id
            var = self.schedule_id_to_var[j_id][sch.unique_id]
            if var.getUbGlobal() == 0.0:  # TODO: figure out why this happens (it should not in this case)
                var_fixed_0 = 1
                break
            self.model.setSolVal(sol, var, 1)
        if var_fixed_0:
            return {"result": SCIP_RESULT.DIDNOTFIND}
        assert self.model.checkSol(sol)
        self.model.trySol(sol)
        self.hist.append(True)
        Z_jmt, _, _ = self.construct_Z_from_feas(self.best_solution)
        assert abs(np.sum(Z_jmt) - np.sum(np.array(self.dur))) < 1e-6
        self.best_Z_jmt = Z_jmt
        self.curr_best = self.best_cost
        self.best_sol_cols = self.best_solution
        return {"result": SCIP_RESULT.FOUNDSOL}

    @timeit_accumulate(timing_store["treesearch"])
    def find_best_schedule(self, starting_cols, nr_avail_idx, all_jobs, incumb, subtree_lb, curr_node):
        self.best_solution = None
        self.best_cost = float('inf')
        self.reset_progress()

        # cheap (but loose) lower bound
        min_costs = [cols[0].cost for cols in all_jobs.values()]  
        min_costs_ordered = [min_costs[k] for k in nr_avail_idx]
        lb_diff_cumul = np.cumsum(min_costs_ordered[::-1])[::-1]

        st = time.perf_counter()
        try:
            if starting_cols != []:
                starting_j = len(starting_cols)
                starting_partial = 0.0
                starting_forbidden = 0
                starting_avail_idx = []
                for col in starting_cols:
                    starting_partial += col.cost
                    starting_forbidden |= col.mt_pairs_bitset 
                    starting_avail_idx.append(col.job_id)
                    self.usage_counts[col.time_steps_flat_list] += 1
                starting_avail_idx += [x for x in nr_avail_idx if x not in starting_avail_idx]

                self.search(starting_j, starting_cols, starting_forbidden, starting_avail_idx, starting_partial, all_jobs,           # bitset version
                    lb_diff_cumul=lb_diff_cumul, max_mach=self.M_max, incumb=incumb, subtree_lb=subtree_lb, first_feas=self.first_feas)
            else:
                self.search(0, starting_cols, 0, nr_avail_idx, 0.0, all_jobs,           # bitset version
                    lb_diff_cumul=lb_diff_cumul, max_mach=self.M_max, incumb=incumb, subtree_lb=subtree_lb, first_feas=self.first_feas)

        except EarlyTerminate:
            self.vprint("Early termination due to iteration limit")

        self.vprint()
        if self.best_solution is not None:
            self.vprint(f"{curr_node} | Found solution of value {self.best_cost:>9.3f}")
        self.vprint(f"{curr_node} | Completed search in \t {time.perf_counter() - st:>9.3f}s")

        self.vprint("\nProfiling summary:")
        if self.verbose & 1 << 2:
            for key in timing_store_rmp:
                total_time = timing_store_rmp[key]
                calls = call_counts_rmp[key]
                avg = total_time / calls if calls > 0 else 0
                self.vprint(f" - {key:18}: calls={calls:8}, total={total_time:.3f}s, avg={avg:.9f}s")
        self.vprint()

    @timeit_accumulate_rmp(timing_store_rmp, call_counts_rmp, "search", recursive=True)
    def search(self, j, chosen_cols, forbidden_mt, nr_avail_idx, partial_value, all_jobs, lb_diff_cumul, max_mach, subtree_lb, incumb, first_feas=False):
        """
        TODO::code (for later)

            work with a compatible (wrt. forbidden_mt) of columns instead of checking isdisjoint 

        TODO::algorithm (for later)

            how to find a better bound?
            how to characterize symmetries?
        """

        if first_feas and self.best_solution is not None:   # TODO: add some gap condition to exit earlier than max_iter when at root node?
            return

        best_cost = self.best_cost
        cutoff = best_cost if incumb is None else min(best_cost, incumb)
        if partial_value >= cutoff:
            return

        nJ = len(self.J)

        if partial_value < cutoff:
            if j == nJ:
                self.best_cost = partial_value
                self.best_solution = list(chosen_cols)
                return
            if self.store_partial_bool and nJ - 3 <= j <= nJ - 1:
                self.store_partial(list(chosen_cols), partial_value)

        curr_j = nr_avail_idx[j]
        remaining_jobs = nr_avail_idx[j + 1:]
        usage_cnts = self.usage_counts

        if j == 0:
            lb_diff = lb_diff_cumul[j + 1]
        else:
            lb_diff = compute_conflict_lb(remaining_jobs, all_jobs, forbidden_mt, usage_cnts, np.intp(max_mach))  # conflict aware LB

        if lb_diff is None:
            return  # prune early. no feasible extension

        remaining_mt_capacity = self.total_mt_capacity - forbidden_mt.bit_count()
        total_dur = self.total_dur
        required_mt = sum(total_dur[j] for j in remaining_jobs)
        
        if required_mt > remaining_mt_capacity:
            return

        if self.early_term():
            return

        for col in all_jobs[curr_j]:
            self.i += 1
            self.track_progress()

            obj = col.cost
            total_lb = partial_value + obj + lb_diff
            if subtree_lb > total_lb:
                total_lb = subtree_lb
            if total_lb >= cutoff: # bounding step
                break

            col_mt = col.mt_pairs_bitset
            if forbidden_mt & col_mt: # if not disjoint
                continue
            
            next_j = j + 1

            # when at/after capacity phase, check+commit usage first; skip on violation
            if next_j >= max_mach and update_usage_with_check(col.time_steps_flat_list, usage_cnts, max_mach):
                continue
            # accept column
            forbidden_mt |= col_mt  # update 
            chosen_cols.append(col)
            if next_j < max_mach:
                usage_cnts[col.time_steps_flat_list] += 1

            self.nr_expl_nd += 1
            self.search(next_j, chosen_cols, forbidden_mt, nr_avail_idx, partial_value + obj, all_jobs, lb_diff_cumul, max_mach, subtree_lb=subtree_lb, incumb=incumb, first_feas=first_feas)

            if first_feas:
                if self.best_solution is not None:
                    return

            pop_col = chosen_cols.pop()
            forbidden_mt &= ~col_mt   # update difference
            usage_cnts[pop_col.time_steps_flat_list] -= 1


    def vprint(self, *args, **kwargs):
        if self.verbose & 1 << 2:
            print(*args, **kwargs)

    def reset_progress(self):
        self.nr_expl_nd = 0
        self.nr_fathom_nd = 0
        self.i = 0

    def track_progress(self):
        if self.i >= 10 and math.log(self.i, 6) % 1 == 0:
            self.vprint(f"{self.i:<5.0f} | explored {self.nr_expl_nd}")

    def early_term(self):
        if self.nr_expl_nd >= self.max_expl_nd:
            raise EarlyTerminate

    def enough_columns(self):
        if min([len(x) for x in self.all_varschedules.values()]) >= 2 * len(self.J):  # TODO: better criterion?
            return True
        return False

    def store_partial(self, chosen_cols, partial_value):
        """
        Store current partial solution
        """
        sol_size = len(chosen_cols)
        partial_sol = PartialSolution(chosen_cols, partial_value)
        self.partial_solutions[sol_size].append(partial_sol)

    def construct_Z_from_feas(self, chosen_cols):
        Z_jmt = np.zeros((len(self.J), len(self.M), len(self.C_t)), dtype=int)
        Z_mt = np.zeros((len(self.M), len(self.C_t)), dtype=int)
        for col in chosen_cols:
            j = col.job_id
            for m, t in col.mt_pairs:
                Z_jmt[j, m, t] += 1
                Z_mt[m, t] += 1

        Z_t = np.sum(Z_mt, axis=0)
        return Z_jmt, Z_mt, Z_t

    def feas_partial_build_Z(self, chosen_cols):
        """
        Verify that the chosen columns are feasible for the problem wrt. the (m,t) unary capacity and the total resource capacity max_mach.
        1) compute the usage of each machine at each time step
        2) sum the usage per timestep to verify that it does not exceed max_mach

        # to cache
        # keep feasibility as a resource of the partial state
        """
        Z_mt = np.zeros((len(self.M), len(self.C_t)), dtype=int)
        for col in chosen_cols:             
            for m, t in col.mt_pairs:
                if Z_mt[m][t] == 1:
                    return None, None
                Z_mt[m][t] += 1
        Z_t = np.sum(Z_mt, axis=0)
        if np.any(Z_t > self.M_max):
            return None, None
        return Z_mt, Z_t

    def price_and_check(self, partial_sol, j_idx=None):
        """
        input: a partial solution of j - 1 columns, 1 per job.
        output: 1 column for the j-th job combining into a feasible solution if found, otherwise None.

        price the j-th job by ensuring feasibility wrt. the input
        """
        partial_cols = partial_sol.chosen_cols
        partial_jobs = partial_sol.jobs
        if not j_idx:
            job_idx = (set(self.J) - set(partial_jobs)).pop()

        forbidden_arcs = []

        Z_mt, Z_t = self.feas_partial_build_Z(partial_cols)
        if Z_t is None:
            return None, None, None
            
        for t in range(len(self.C_t)):
            if Z_t[t] == self.M_max:
                for m in self.M:
                    forbidden_arcs.append((job_idx, m, t))
            else:
                for m in self.M:
                    if Z_mt[m][t] == 1:
                        forbidden_arcs.append((job_idx, m, t))

        F, decisions = compute_shortest_path(n_machines=len(self.M), C_max=self.C_max,
                                             dur=self.dur[job_idx], seq=self.seq[job_idx],
                                             rc_term=self.rc_term, branch_operations=forbidden_arcs, branch_operations1=[], branch_operations0=[])
        if F[-1][-1] != float('inf'):
            job_sch = JobSchedule()
            path_job = recover_path(decisions)
            job_sch.job_id = job_idx
            job_sch.compl_times = [0 for _ in self.M]
            for (idx, op) in enumerate(self.seq[job_idx]):
                job_sch.compl_times[op - 1] = path_job[idx][1]
            job_sch.unique_id = job_sch.schedule_to_var_name()
            job_sch.start_times = job_sch.comp_start_times(self.dur[job_idx])
            job_sch.time_steps = job_sch.comp_time_steps()

            flat_steps = set().union(*job_sch.time_steps)
            job_sch.time_steps_flat = flat_steps
            job_sch.time_steps_flat_list = list(flat_steps)
            
            pairs = set()
            bits = 0 
            T_SIZE = self.C_max
            for m, t_steps in enumerate(job_sch.time_steps):
                base = m * T_SIZE
                add = pairs.add
                for t in t_steps:
                    add((m, t))
                    bits |= 1 << (base + t)
            job_sch.mt_pairs = pairs
            job_sch.mt_pairs_bitset = bits
            
            job_sch.cost = np.sum([self.W_m[m]*self.C_t[t] for (m, t) in job_sch.mt_pairs])

            return job_sch, partial_sol, partial_sol.cost + job_sch.cost

        return None, None, None

    def destroy_repair(self, solution, index):
        """
        Remove element at `index` and rebuild it using the heuristic.
        Returns the new solution and its score.
        """
        partial_solution = [x for x in solution if x.job_id != index]
        new_partial_cost = sum([col.cost for col in partial_solution])
        new_part_sol = PartialSolution(chosen_cols=partial_solution, cost=new_partial_cost)
        new_sch, new_part_sol, new_tot_cost = self.price_and_check(new_part_sol)
        return new_sch, new_part_sol, new_tot_cost

    def iterative_destroy_repair(self, solution, best_incumb):
        current_solution = solution
        current_score = sum([x.cost for x in current_solution])
        improved = True
        improved_i = 0
        new_vars = []
        current_solution_split = (None, None)

        while improved:
            improved = False
            best_solution = current_solution
            best_score = current_score

            # Try destroying each position
            for i in range(len(current_solution)):
                new_sch, new_part_sol, new_tot_cost = self.destroy_repair(solution=current_solution, index=i)
                new_vars.append(new_sch)
                candidate_score = new_tot_cost

                if candidate_score < best_score:  # assuming minimization
                    best_solution = new_part_sol.chosen_cols + [new_sch]
                    best_solution_split = (new_sch, new_part_sol)
                    best_score = candidate_score
                    improved = True
                    improved_i += 1

            # If an improvement was found, update and continue
            if improved:
                current_solution = best_solution
                current_solution_split = best_solution_split
                current_score = best_score

        return current_solution, current_solution_split, current_score, new_vars, improved_i

    def catch_new_incumbent(self):
        """
        Updates data when a better solution was found by another heuristic
        """
        ub =  int(np.round(self.model.getPrimalbound(),decimals=1))
        incumb_cols = []
        if self.model.isInfinity(ub):
            return ub, incumb_cols
        if ub < self.curr_best - 0.5:
            self.vprint(f"Update incumbent = {ub}")
            self.curr_best = ub
            best_sol = self.model.getBestSol()
            for j in self.J:
                all_sch = self.all_schedules_jobs[j]
                all_vars = self.all_vars_jobs[j]
                for var_id, var in enumerate(all_vars):
                    if self.model.getSolVal(sol=best_sol, expr=self.model.getTransformedVar(var)) > 0.5:
                        incumb_cols.append(all_sch[var_id])
                        break
            self.best_Z_jmt, _, _ = self.construct_Z_from_feas(chosen_cols=incumb_cols)
            assert(abs(np.sum(self.best_Z_jmt) - np.sum(np.array(self.dur))) < 1e-6)
            self.best_sol_cols = incumb_cols

        return ub, self.best_sol_cols
