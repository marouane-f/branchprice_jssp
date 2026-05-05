import time
import random
import numpy as np
from pyscipopt import Pricer, SCIP_RESULT
import pyscipopt as scip
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import (timing_store, timeit_accumulate, iter_accumulate, iter_store,
                   get_duals, VarSchedule, lpbased_heur_tune,
                    comp_subgradient, comp_in_sep, f_decr, f_incr)
from shortest_path import prepare_arrays, pricing_solver, compute_pattern_red_cost, compute_rc_term_njit
from collections import defaultdict


# --- verbosity bit-flags ---
class V:
    GENERAL = 1 << 0       # pricing progress per iteration
    EARLY_BRANCH = 1 << 1  # early-branching decisions
    HEUR_COLS = 1 << 2     # LNS/heuristic column additions


# --- algorithm constants ---
MAX_MISPRICES = 10
ALPHA_MIN = 1e-6            # unifies the 1e-6 and 0.00001 spellings used in the original
ALPHA_INIT = 0.7
TAILOFF_PATIENCE = 10
TAILOFF_GAP_CHG_THRESHOLD = 0.01
LARS_THRESHOLD = 0.1
RC_FIXING_GAP_THRESHOLD = 0.5
HEADING_IN_THRESHOLD = 1e-2
RC_FIXING_SLACK = 1


# ---------------------------------------------------------------------------
# Parameter groups (passed to __init__)
# ---------------------------------------------------------------------------

@dataclass
class ProblemData:
    jobs: object
    machines: object
    powers: object
    KP: object
    duration: object
    job_sequence: object
    Cost_t: object
    part_c: object
    disj_c: object
    res_c: object
    matrix_rhs: object
    all_cons_parts: object
    all_conss: object
    all_cons_names: object
    n_conss: int
    dualfeastol: float

@dataclass
class BranchingState:
    arcs_to_forbid: object
    arcs_to_force: object
    arcs_to_forbid_prob: object
    arcs_to_force_prob: object


@dataclass
class CGConfig:
    cg_bool: bool
    wentges_bool: bool
    sg_algo_bool: bool
    comp_pricing: bool
    early_branch_int: bool
    early_branch_tailoff: bool
    early_branch_lars: bool
    verbose: int
    reduc_bool: object
    reli_strongbr: object


@dataclass
class VarRegistry:
    all_vars_jobs: object
    all_schedules_jobs: object
    all_vars_jmt: object
    all_schedules_jmt: object
    all_varschedules: object
    schedule_id_to_var: object
    added_probing_vars: object


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class ShortestPathPricer(Pricer):

    @classmethod
    def from_legacy_args(cls,
                         jobs, machines, powers, KP_ineq, duration, job_sequence, Cost_t,
                         all_constraints,
                         all_rhss, dualfeastol_val,
                         priced_sols, schedule_id_to_var,
                         arcs_to_forbid, arcs_to_forbid_prob, arcs_to_force_prob,
                         all_vars_jobs, all_schedules_jobs, all_vars_jmt, all_schedules_jmt,
                         all_varschedules, added_probing_vars, reli_strongbr,
                         early_branch_int, early_branch_tailoff, early_branch_lars, early_branched,
                         cg_bool, wentges_bool, comp_pricing, sg_algo_bool,
                         dual_evol, best_lag_bound, lag_bound, branch_evol, lp_evol, lower_bound,
                         req_iters, arcs_to_force,
                         ext_rmp,
                         reduc_bool,
                         all_cons_parts, all_conss, all_cons_names, n_conss,
                         solve_begin_time,
                         total_added_patterns,
                         verbose=False,
                         *args, **kwargs):
        """Backwards-compatible constructor that accepts the original flat argument list."""
        problem = ProblemData(
            jobs=jobs, machines=machines, powers=powers, KP=KP_ineq,
            duration=duration, job_sequence=job_sequence, Cost_t=Cost_t,
            part_c=all_constraints[0], disj_c=all_constraints[1], res_c=all_constraints[2],
            matrix_rhs=all_rhss,
            all_cons_parts=all_cons_parts, all_conss=all_conss,
            all_cons_names=all_cons_names, n_conss=n_conss,
            dualfeastol=dualfeastol_val,
        )
        branching = BranchingState(
            arcs_to_forbid=arcs_to_forbid, arcs_to_force=arcs_to_force,
            arcs_to_forbid_prob=arcs_to_forbid_prob, arcs_to_force_prob=arcs_to_force_prob,
        )
        config = CGConfig(
            cg_bool=cg_bool, wentges_bool=wentges_bool, sg_algo_bool=sg_algo_bool,
            comp_pricing=comp_pricing, early_branch_int=early_branch_int,
            early_branch_tailoff=early_branch_tailoff, early_branch_lars=early_branch_lars,
            verbose=verbose, reduc_bool=reduc_bool, reli_strongbr=reli_strongbr,
        )
        registry = VarRegistry(
            all_vars_jobs=all_vars_jobs, all_schedules_jobs=all_schedules_jobs,
            all_vars_jmt=all_vars_jmt, all_schedules_jmt=all_schedules_jmt,
            all_varschedules=all_varschedules, schedule_id_to_var=schedule_id_to_var,
            added_probing_vars=added_probing_vars,
        )
        return cls(
            problem=problem, branching=branching, config=config, registry=registry,
            priced_sols=priced_sols, early_branched=early_branched,
            lp_evol=lp_evol, req_iters=req_iters,
            solve_begin_time=solve_begin_time, total_added_patterns=total_added_patterns,
            *args, **kwargs,
        )

    def __init__(self, problem: ProblemData, branching: BranchingState, config: CGConfig,
                 registry: VarRegistry, priced_sols, early_branched, lp_evol, req_iters,
                 solve_begin_time, total_added_patterns, debug_checks=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- problem data ---
        self.J = problem.jobs
        self.M = problem.machines
        self.W_m = problem.powers
        self.KP = problem.KP
        self.dur = problem.duration
        self.seq = problem.job_sequence
        self.C_t = problem.Cost_t
        self.part_c = problem.part_c
        self.disj_c = problem.disj_c
        self.res_c = problem.res_c
        self.matrix_rhs = problem.matrix_rhs
        self.all_cons_parts = problem.all_cons_parts
        self.all_conss = problem.all_conss
        self.all_cons_names = problem.all_cons_names
        self.n_conss = problem.n_conss
        self.dualfeastol = problem.dualfeastol

        # --- SCIP variable references ---
        self.all_vars_jobs = registry.all_vars_jobs
        self.all_schedules_jobs = registry.all_schedules_jobs
        self.all_vars_jmt = registry.all_vars_jmt
        self.all_schedules_jmt = registry.all_schedules_jmt
        self.all_varschedules = registry.all_varschedules
        self.added_probing_vars = registry.added_probing_vars

        # --- branching arc sets ---
        self.arcs_to_forbid = branching.arcs_to_forbid
        self.arcs_to_force = branching.arcs_to_force
        self.arcs_to_forbid_prob = branching.arcs_to_forbid_prob
        self.arcs_to_force_prob = branching.arcs_to_force_prob
        self.prev_arcs_probing = None

        # --- column generation config ---
        self.verbose = config.verbose
        self.cg_bool = config.cg_bool
        self.sg_algo_bool = config.sg_algo_bool   # subgradient (disabled)
        self.sg_bool = None
        self.proba_pricing = config.comp_pricing  # probability complementary pricing (root only)

        # --- early branching ---
        self.early_branch_int = config.early_branch_int
        self.early_branch_tailoff = config.early_branch_tailoff
        self.early_branch_lars = config.early_branch_lars
        self.early_branched = early_branched

        # --- dual stabilization (Wentges smoothing) ---
        self.wentges_bool = config.wentges_bool
        self.dir_wentges_bool = True
        self.local_wentges_bool = config.wentges_bool
        self.best_lag_bound = 0
        self.curr_lag_bound = 0
        self.abs_best_lag_bound = 0
        self.best_part_duals = None
        self.best_disj_duals = None
        self.best_res_duals = None
        self.best_subgrad_normalized = None
        self.w_alpha = 0
        self.w_beta = 0
        self.nr_misprices = 0
        self.i = 0

        # --- shared state (written here, read by branching rule) ---
        self.schedule_id_to_var = registry.schedule_id_to_var
        self.fixed_vars = defaultdict(list)
        self.priced_sols = priced_sols

        # --- tree tracking and logging ---
        self.total_added_patterns = total_added_patterns
        self.prev_node = 0
        self.probing_bool = 0
        self.reli_strongbr = config.reli_strongbr
        self.tailoff_count = 0
        self.gap = []
        self.lp_evol = lp_evol
        self.req_iters = req_iters
        self.root_cg_vars = 0
        self.root_cg_time = solve_begin_time
        self.last_cg_opt_node = -1

        # --- reduced cost fixing ---
        self.rc_fixing_bool = 0
        self.best_primal = float('inf')
        self.subtree_dual_bd = 0
        self.best_dual = 0

        # --- debug / validation ---
        # When True, runs in-solver assertions to verify LP correctness (hot path overhead).
        # Set to False in production once the implementation is trusted.
        self._debug_checks = debug_checks

    # -----------------------------------------------------------------------
    # Main pricing entry points (SCIP callbacks — signatures are fixed)
    # -----------------------------------------------------------------------

    @iter_accumulate(iter_store["pricer"])
    @timeit_accumulate(timing_store["pricer"])
    def price(self, farkas):
        mod = self.model

        # --- flush LNS-priced columns ---
        early = self._handle_priced_sols()
        if early is not None:
            return early

        # --- resolve current node; guard against re-entry after heuristic column add ---
        curr_node_obj = mod.getCurrentNode()
        curr_node = curr_node_obj.getNumber()
        if self.last_cg_opt_node == curr_node:  # done to avoid re-entering pricing despite convergence
            return {"result": scip.SCIP_RESULT.SUCCESS}
        curr_node_depth = curr_node_obj.getDepth()
        root_bool = (curr_node == 1)
        self.vprint(f"Pricing | iter {self.i - self.req_iters[-1]} | node {curr_node} | depth {curr_node_depth}")
        self.vprint()

        # --- reset stability center if node or probing state changed ---
        self._update_node_state(mod, curr_node)

        # --- early branching checks (tailoff, integrality gap, Lars criterion) ---
        mod_Ceil = mod.feasCeil
        lb = max(1, mod_Ceil(curr_node_obj.getLowerbound()))
        curr_lp = mod.getLPObjVal()
        gap = mod_Ceil(curr_lp) - lb
        early = self._check_early_branching(gap, lb, curr_lp, curr_node, curr_node_depth, root_bool, mod, farkas)
        if early is not None:
            return early
        self.early_branched[curr_node] = False

        # --- compute dual vectors and determine stabilization mode ---
        arcs_to_forbid, arcs_to_force = self._resolve_arc_sets(curr_node)
        curr_wentges_bool, orig_dual_sols, out_dual_dict, out_dual_vector, rc_term, all_cons_Rhs, T = \
            self._prepare_duals(farkas, curr_lp)

        # --- select pricing strategy and solve ---
        all_patterns, all_min_red_costs = self._select_and_run_pricing(
            curr_wentges_bool, orig_dual_sols, out_dual_dict, out_dual_vector,
            arcs_to_forbid, arcs_to_force, rc_term, all_cons_Rhs,
            farkas, root_bool, curr_lp, curr_node, curr_node_depth,
        )

        # --- add columns and update Lagrangian bound ---
        self._commit_columns(all_patterns, all_min_red_costs, farkas)
        self._run_proba_pricing(orig_dual_sols, rc_term, arcs_to_forbid, arcs_to_force, T, farkas, root_bool, curr_node)
        self.track_progress(curr_lp, curr_node, curr_node_obj)
        self._handle_cg_optimal(root_bool, mod, curr_node)

        return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': self.abs_best_lag_bound}

    def pricerredcost(self):
        self.i += 1
        return self.price(farkas=False)

    @timeit_accumulate(timing_store["farkas_pricer"])
    def pricerfarkas(self):
        self.i += 1
        return self.price(farkas=True)

    # -----------------------------------------------------------------------
    # price() sub-steps
    # -----------------------------------------------------------------------

    def _handle_priced_sols(self) -> Optional[dict]:
        """Flush columns found by the LNS heuristic into the RMP.
        Returns a result dict (causing price() to return early) in the probing branch,
        or None to fall through to normal pricing."""
        if not self.priced_sols:
            return None
        mod = self.model
        if mod.inProbing() and mod.getParam("lp/solvefreq") == -1:
            if self.verbose & V.HEUR_COLS:
                print("[RMPTreeSearch] Adding cols from compl. pricing")
            self._flush_priced_cols()
            return {"result": scip.SCIP_RESULT.SUCCESS}
        elif not mod.inProbing() and not mod.getParam("lp/solvefreq") == -1:
            if self.verbose & V.HEUR_COLS:
                print("[RMPTreeSearch//Pricer] Adding cols at pricing since it did not work during Heur call (no solution passed)")
            self._flush_priced_cols()
        return None

    def _flush_priced_cols(self) -> None:
        sch_to_var = self.schedule_id_to_var
        for to_add in self.priced_sols:
            j = to_add.job_id
            var_names = [x.name for x in sch_to_var[j].values()]
            if to_add.unique_id not in var_names:
                if self.verbose & V.HEUR_COLS:
                    print(f"Added {to_add.schedule_to_var_name()}")
                new_var = self.add_pattern_column(to_add)
                sch_to_var[j][to_add.unique_id] = new_var
        self.priced_sols.clear()

    def _check_early_branching(self, gap, lb, curr_lp, curr_node, curr_node_depth,
                                root_bool, mod, farkas) -> Optional[dict]:
        """Run all three early-branching criteria in order. Returns a result dict if
        any criterion fires, else None."""
        if curr_node_depth >= 0 and (not farkas) * self.early_branch_tailoff:
            if self.early_branching_tailoff(gap, lb, curr_lp):
                lpbased_heur_tune(root_bool, mod, max_iters=1e5) # TODO: move 1e5 to algorithm's args
                return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lb, "stopearly": True}

        if self.early_branching_int(lb, curr_lp, gap, curr_node_depth, node_depth=0):
            self.early_branched[curr_node] = True
            lpbased_heur_tune(root_bool, mod, max_iters=1e5) # TODO: move 1e5 to algorithm's args
            if root_bool:
                self.root_cg_time = time.perf_counter() - self.root_cg_time
                self.root_cg_vars = mod.getNVars(True)
            self.last_cg_opt_node = curr_node
            return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lb, "stopearly": True}

        if self.early_branching_lars(lb, curr_lp, self.model.getPrimalbound()):
            self.early_branched[curr_node] = True
            lpbased_heur_tune(root_bool, mod, max_iters=1e5) # TODO: move 1e5 to algorithm's args
            return {"result": scip.SCIP_RESULT.SUCCESS, "lowerbound": lb, "stopearly": True}

        return None

    def _update_node_state(self, mod, curr_node) -> None:
        """Reset stability center on new node or probing arc change; stop stabilization if budget exceeded."""
        if curr_node != self.prev_node:
            self.prev_node = curr_node
            self.reset_stability_center()
        self.probing_bool = mod.inProbing()
        if self.probing_bool:
            if self.prev_arcs_probing != self.arcs_to_forbid_prob | self.arcs_to_force_prob:
                self.sg_bool = self.sg_algo_bool
                self.reset_stability_center()
        if self.local_wentges_bool and (self.nr_misprices >= MAX_MISPRICES or self.w_alpha <= ALPHA_MIN):
            self.vprint(f"Turning off adaptive stabilization after {self.nr_misprices} mis-prices with alpha = {np.round(self.w_alpha, decimals=3)} at iteration {self.i}")
            self.local_wentges_bool = False
            self.w_alpha = 0.0

    def _resolve_arc_sets(self, curr_node):
        """Return (arcs_to_forbid, arcs_to_force) for the current pricing context."""
        if self.probing_bool:
            return self.arcs_to_forbid_prob, self.arcs_to_force_prob
        elif not self.cg_bool:
            return self.arcs_to_forbid[curr_node], self.arcs_to_force[curr_node]
        else:
            return set(), set()

    def _prepare_duals(self, farkas, curr_lp):
        """Fetch LP duals, build rc_term, and determine whether Wentges smoothing is active."""
        heading_in_bool = True
        if self.i - self.req_iters[-1] >= 2:
            heading_in_bool = abs(curr_lp - self.lp_evol[-1]) / abs(self.lp_evol[-1]) > HEADING_IN_THRESHOLD

        # Stabilization: after heading in, not farkas, and at the presence of a lagrangian incumbent
        curr_wentges_bool = (
            self.local_wentges_bool
            and self.best_part_duals is not None
            and not farkas
            and not heading_in_bool
            and not self.probing_bool
        )

        powers = self.W_m
        n_machines = len(powers)
        C_max = len(self.C_t)
        T = range(C_max)
        n_KP = len(self.KP)

        part_duals = get_duals(constraint_set=self.part_c, rmp=self.model, farkas_bool=farkas)      # TODO: work with arrays
        disj_duals = get_duals(constraint_set=self.disj_c, rmp=self.model, farkas_bool=farkas)
        res_duals = [get_duals(constraint_set=self.res_c[t], rmp=self.model, farkas_bool=farkas) for t in T]
        cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr = prepare_arrays(self.C_t, farkas, self.KP, res_duals, disj_duals, n_machines, n_KP, C_max)
        rc_term = compute_rc_term_njit(powers, cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr)

        all_cons_Rhs = [self.model.getRhs(cons) for cons in self.all_conss]
        orig_dual_sols, out_dual_dict, out_dual_vector = self.build_dual_vector(
            part_duals, disj_duals, res_duals, self.all_cons_names, self.all_cons_parts,
        )

        return curr_wentges_bool, orig_dual_sols, out_dual_dict, out_dual_vector, rc_term, all_cons_Rhs, T

    def _run_directional_smoothing(self, all_cons_names, all_cons_parts, out_dual_vector, out_dual_dict,
                                    arcs_to_forbid, arcs_to_force, farkas, root_bool,
                                    curr_lp, curr_node, curr_node_depth, dual_feas_tol_param,
                                    rc_term, part_duals, matrix_rhs, all_cons_Rhs, n_conss,
                                    ) -> Optional[Tuple]:
        """Directional Wentges smoothing step.
        Returns (all_patterns, all_min_red_costs) if improving columns were found under the
        smoothed duals, else None (sets self.dir_wentges_bool = False as side effect)."""
        in_dual_vector, out_in_vector, out_in_norm = self.compute_out_in(
            all_cons_names=all_cons_names, all_cons_parts=all_cons_parts,
            out_dual_vector=out_dual_vector,
        )

        if out_in_norm == 0:
            self.dir_wentges_bool = False
            self.vprint(f"Norm = 0 | iter {self.i}, node {curr_node} depth {curr_node_depth}: switching off directional stabilization")
            return None

        smooth_dual_vector = self.compute_sep_duals(
            out_dual_vector=out_dual_vector, in_dual_vector=in_dual_vector,
            out_in_vector=out_in_vector, out_in_norm=out_in_norm,
        )

        keys = list(out_dual_dict.keys())
        smooth_dual_sols_dict = dict(zip(keys, smooth_dual_vector))
        smooth_dual_sols = self.extract_smoothed_duals(smooth_dual_sols_dict, all_cons_parts, range(len(self.C_t)))

        all_min_red_costs_smooth, all_patterns_smooth = pricing_solver(
            self.J, self.W_m, self.KP, self.dur, self.seq, self.C_t,
            smooth_dual_sols, forbidden_arcs=arcs_to_forbid, forced_arcs=arcs_to_force,
            farkas_bool=farkas, root_bool=root_bool,
        )

        sum_min_red_cost_smooth = sum(all_min_red_costs_smooth.values())
        lag_bound = self.model.feasCeil(smooth_dual_vector @ matrix_rhs + sum_min_red_cost_smooth)
        subgrad_vec, subgrad_norm = comp_subgradient(all_cons_parts, n_conss, all_cons_Rhs, self.M, self.dur, all_patterns_smooth)
        best_subgrad_normalized = subgrad_vec / subgrad_norm
        self.update_incumbent_duals(lag_bound, curr_lp, smooth_dual_sols, best_subgrad_normalized)

        # recompute reduced cost of each found column under the original (unsmoothed) duals
        for job_idx, patt in all_patterns_smooth.items():
            if patt != []:
                orig_red_cost = compute_pattern_red_cost(pattern_pairs=patt.mt_pairs, rc_term=rc_term, part_dual_term=part_duals[job_idx])
                all_min_red_costs_smooth[job_idx] = orig_red_cost

        # misprice check 1: all columns non-improving wrt. smoothed duals (using isGE tolerance)
        if all(self.model.isGE(x, 0) for x in all_min_red_costs_smooth.values()):
            self.vprint(f"Misprice | iter {self.i}, node {curr_node} depth {curr_node_depth}: switching off directional stabilization")
            self.dir_wentges_bool = False
            return None

        # misprice check 2: all columns non-improving wrt. explicit tolerance
        if sum(1 for x in all_min_red_costs_smooth.values() if x >= -dual_feas_tol_param) == len(self.J):
            self.vprint(f"Misprice | iter {self.i}, node {curr_node} depth {curr_node_depth}: switching off directional stabilization")
            self.dir_wentges_bool = False
            return None

        self.vprint(f"Smoothing | iter {self.i}, node {curr_node} depth {curr_node_depth}: using directional stabilization")
        return all_patterns_smooth.copy(), all_min_red_costs_smooth.copy()

    def _run_static_smoothing(self, all_cons_names, all_cons_parts, out_dual_vector,
                               arcs_to_forbid, arcs_to_force, farkas, root_bool,
                               curr_lp, curr_node, curr_node_depth, dual_feas_tol_param,
                               rc_term, matrix_rhs, all_cons_Rhs, n_conss, orig_dual_sols,
                               ) -> Optional[Tuple]:
        """Static (alpha-blended) Wentges smoothing loop.
        Returns (all_patterns, all_min_red_costs) if improving columns were found; else None
        (sets self.local_wentges_bool = False and self.w_alpha = 0.0 as side effect)."""
        part_duals, disj_duals, res_duals = orig_dual_sols
        mod_Ceil = self.model.feasCeil

        while self.w_alpha > ALPHA_MIN and self.nr_misprices < MAX_MISPRICES:

            smooth_dual_sols_nodir = self.compute_tilde_duals(part_duals, disj_duals, res_duals)
            smooth_dual_sols_nodir_vector = duals_to_dict(all_cons_names, all_cons_parts, smooth_dual_sols_nodir)
            smooth_dual_sols_nodir_vector = np.fromiter((smooth_dual_sols_nodir_vector[cons] for cons in smooth_dual_sols_nodir_vector), dtype=float)

            all_min_red_costs_smooth, all_patterns_smooth = pricing_solver(
                self.J, self.W_m, self.KP, self.dur, self.seq, self.C_t,
                smooth_dual_sols_nodir, forbidden_arcs=arcs_to_forbid, forced_arcs=arcs_to_force,
                farkas_bool=farkas, root_bool=root_bool,
            )

            sum_min_red_cost_smooth = sum(all_min_red_costs_smooth.values())
            lag_bound = mod_Ceil(smooth_dual_sols_nodir_vector @ matrix_rhs + sum_min_red_cost_smooth)  # TODO: optimize computation
            subgrad_vec, subgrad_norm = comp_subgradient(all_cons_parts, n_conss, all_cons_Rhs, self.M, self.dur, all_patterns_smooth)
            best_subgrad_normalized = subgrad_vec / subgrad_norm
            self.update_incumbent_duals(lag_bound, curr_lp, smooth_dual_sols_nodir, best_subgrad_normalized)

            # recompute reduced cost of each found column under the original (unsmoothed) duals
            for job_idx, patt in all_patterns_smooth.items():
                if patt != []:
                    orig_red_cost = compute_pattern_red_cost(pattern_pairs=patt.mt_pairs, rc_term=rc_term, part_dual_term=part_duals[job_idx])
                    all_min_red_costs_smooth[job_idx] = orig_red_cost

            # all columns non-improving under smooth duals: check against original duals
            if all(x >= -dual_feas_tol_param for x in all_min_red_costs_smooth.values()):
                all_min_red_costs_orig, all_patterns_orig = pricing_solver(
                    self.J, self.W_m, self.KP, self.dur, self.seq, self.C_t, orig_dual_sols, rc_term=rc_term,
                    forbidden_arcs=arcs_to_forbid, forced_arcs=arcs_to_force,
                    farkas_bool=farkas, root_bool=root_bool,
                )

                if any(x <= -dual_feas_tol_param for x in all_min_red_costs_orig.values()):  # misprice: orig has improving columns
                    self.mispricing_update(all_cons_parts, n_conss, orig_dual_sols, smooth_dual_sols_nodir, subgrad_vec, subgrad_norm, curr_node, curr_node_depth)
                else:
                    # no improving columns wrt. original duals either: CG optimal under stabilization
                    self.vprint(f"Smoothing | iter {self.i}, node {curr_node} depth {curr_node_depth}: no improving columns wrt. original duals")
                    lag_bound_opt = mod_Ceil(curr_lp + sum(all_min_red_costs_orig.values()))
                    self.curr_lag_bound = lag_bound_opt
                    self.best_lag_bound = lag_bound_opt
                    if self._debug_checks:
                        assert self.model.isGE(math.ceil(curr_lp), lag_bound_opt)
                    return all_patterns_smooth, all_min_red_costs_smooth

            else:
                # found columns improving under smooth duals; check for misprice against original duals
                misprice_counter = sum(1 for x in all_min_red_costs_smooth.values() if self.model.isGE(x, 0))  # TODO: use map
                if misprice_counter == len(self.J):
                    # all non-improving under original duals: misprice
                    self.mispricing_update(all_cons_parts, n_conss, orig_dual_sols, smooth_dual_sols_nodir, subgrad_vec, subgrad_norm, curr_node, curr_node_depth)
                else:
                    self.vprint(f"Smoothing | iter {self.i}, node {curr_node} depth {curr_node_depth}: using adaptive stabilization")
                    return all_patterns_smooth.copy(), all_min_red_costs_smooth.copy()

        # while-loop exhausted: too many misprices or alpha hit floor
        self.vprint(f"Turning off adaptive stabilization after {self.nr_misprices} mis-prices with alpha = {np.round(self.w_alpha, decimals=3)} | iter {self.i}, node {curr_node} depth {curr_node_depth}")
        self.local_wentges_bool = False
        self.w_alpha = 0.0
        return None

    def _run_unstabilized_pricing(self, orig_dual_sols, out_dual_vector, rc_term,
                                   arcs_to_forbid, arcs_to_force, farkas, root_bool,
                                   curr_lp, matrix_rhs, all_cons_parts, n_conss, all_cons_Rhs,
                                   ) -> Tuple:
        """Price using original (unsmoothed) duals and update the Lagrangian bound."""
        if self.wentges_bool:
            self.vprint(f"Using original duals | Farkas={farkas}")

        all_min_red_costs, all_patterns = pricing_solver(
            self.J, self.W_m, self.KP, self.dur, self.seq, self.C_t, orig_dual_sols, rc_term=rc_term,
            forbidden_arcs=arcs_to_forbid, forced_arcs=arcs_to_force,
            farkas_bool=farkas, root_bool=root_bool,
        )

        sum_min_red_cost = sum(all_min_red_costs.values())

        if not farkas:
            mod_Ceil = self.model.feasCeil
            lag_bound = mod_Ceil(curr_lp + sum_min_red_cost)

            if self._debug_checks:
                lag_bound_check = mod_Ceil(out_dual_vector @ matrix_rhs + sum_min_red_cost)
                assert abs(curr_lp - (out_dual_vector @ matrix_rhs)) < 1e-3  # TODO: debug mode
                assert lag_bound == lag_bound_check                           # TODO: debug mode

            subgrad_vec, subgrad_norm = comp_subgradient(all_cons_parts, n_conss, all_cons_Rhs, self.M, self.dur, all_patterns)
            best_subgrad_normalized = subgrad_vec / subgrad_norm
            self.update_incumbent_duals(lag_bound, curr_lp, orig_dual_sols, best_subgrad_normalized)

        return all_patterns, all_min_red_costs

    def _select_and_run_pricing(self, curr_wentges_bool, orig_dual_sols, out_dual_dict, out_dual_vector,
                                arcs_to_forbid, arcs_to_force, rc_term, all_cons_Rhs,
                                farkas, root_bool, curr_lp, curr_node, curr_node_depth) -> Tuple:
        """Route to directional smoothing, static smoothing, or original-dual fallback.
        Returns (all_patterns, all_min_red_costs)."""
        part_duals = orig_dual_sols[0]
        dual_feas_tol = self.dualfeastol
        all_min_red_costs = {i: 1 for i in self.J}
        all_patterns = None

        if curr_wentges_bool:
            if self.dir_wentges_bool:
                result = self._run_directional_smoothing(
                    self.all_cons_names, self.all_cons_parts, out_dual_vector, out_dual_dict,
                    arcs_to_forbid, arcs_to_force, farkas, root_bool,
                    curr_lp, curr_node, curr_node_depth, dual_feas_tol,
                    rc_term, part_duals, self.matrix_rhs, all_cons_Rhs, self.n_conss,
                )
                if result is not None:
                    all_patterns, all_min_red_costs = result

            if not self.dir_wentges_bool:  # either was already False, or just disabled above
                result = self._run_static_smoothing(
                    self.all_cons_names, self.all_cons_parts, out_dual_vector,
                    arcs_to_forbid, arcs_to_force, farkas, root_bool,
                    curr_lp, curr_node, curr_node_depth, dual_feas_tol,
                    rc_term, self.matrix_rhs, all_cons_Rhs, self.n_conss, orig_dual_sols,
                )
                if result is not None:
                    all_patterns, all_min_red_costs = result

        if self.w_alpha <= ALPHA_MIN or not curr_wentges_bool:
            all_patterns, all_min_red_costs = self._run_unstabilized_pricing(
                orig_dual_sols, out_dual_vector, rc_term,
                arcs_to_forbid, arcs_to_force, farkas, root_bool,
                curr_lp, self.matrix_rhs, self.all_cons_parts, self.n_conss, all_cons_Rhs,
            )

        return all_patterns, all_min_red_costs

    def _commit_columns(self, all_patterns, all_min_red_costs, farkas) -> None:
        """Add improving columns to the RMP and update the absolute best Lagrangian bound.

        updating best with curr introduces adverse effects (dual vectors not updated, slows convergence);
        this serves only to verify if gap < 1."""
        dual_feas_tol = self.dualfeastol
        self.total_added_patterns = 0
        add_var_objects = self.add_min_red_cost_patterns(all_patterns, all_min_red_costs, dual_feas_tol, farkas)

        curr_max = max(self.curr_lag_bound, self.best_lag_bound)
        if curr_max > self.abs_best_lag_bound:
            self.abs_best_lag_bound = curr_max

        if self._debug_checks:
            self.verify_reduced_costs_and_update_bound(add_var_objects, all_min_red_costs, dual_feas_tol, farkas)

    def _run_proba_pricing(self, orig_dual_sols, rc_term, arcs_to_forbid, arcs_to_force,
                           T, farkas, root_bool, curr_node) -> None:
        """Probability complementary pricing — only active at the root node."""
        if not self.proba_pricing * (curr_node == 1):
            return
        proba_patts, _ = self.probability_complementary_pricing(
            self.model.isLT, self.dualfeastol, T, rc_term, orig_dual_sols,
            arcs_to_forbid, arcs_to_force, farkas, root_bool,
        )
        for proba_patt in proba_patts:
            self.add_pattern_column(proba_patt)

    def _handle_cg_optimal(self, root_bool, mod, curr_node) -> None:
        """Actions taken when no new columns were added (CG optimality at current node)."""
        if self.total_added_patterns != 0:
            return

        lpbased_heur_tune(root_bool, mod, max_iters=1e5) # TODO: move 1e5 to args

        if root_bool:
            self.root_cg_time = time.perf_counter() - self.root_cg_time
            self.root_cg_vars = mod.getNVars(True)

        self.last_cg_opt_node = curr_node

        if self.rc_fixing_bool and not self.cg_bool and not self.probing_bool:
            primal_bd = self.model.getPrimalbound()
            dual_bd = self.model.getDualbound()
            int_gap = 100.0 * (primal_bd - dual_bd) / dual_bd
            if int_gap < RC_FIXING_GAP_THRESHOLD:
                self.rc_fixing_loop(primal_bd)

    # -----------------------------------------------------------------------
    # Early-branching criteria
    # -----------------------------------------------------------------------

    def early_branching_tailoff(self, gap, lb, curr_lp):
        lp_fractional = self.model.isGT(self.model.feasFrac(curr_lp), 0)

        curr_gap = gap / lb
        self.gap.append(curr_gap)
        if len(self.gap) >= 2:
            prev_gap = self.gap[-2]
            if prev_gap != 0:
                gap_chg = abs(curr_gap - prev_gap) / prev_gap
                if gap_chg < TAILOFF_GAP_CHG_THRESHOLD:
                    self.tailoff_count += 1

        if self.tailoff_count > TAILOFF_PATIENCE and lp_fractional:
            if self.verbose & V.EARLY_BRANCH:
                print("--abort pricing due to tailing-off")
            return True
        return None

    def early_branching_int(self, lb, curr_lp, gap, curr_node_depth, node_depth):
        if self.early_branch_int and curr_node_depth >= node_depth:

            if gap < 0:
                self.vprint("Curr node", self.model.getCurrentNode().getNumber())
                self.vprint("ceil LPObjVal", curr_lp)
                self.vprint("LB", lb)
                raise ValueError(f"Negative gap encountered {gap} | curr_lp ={curr_lp} | lb={lb}")

            if gap < 1:
                if self.verbose & V.EARLY_BRANCH:
                    print(f"Gap too small = {gap}, stopping pricing with lb = {lb}")
                self.req_iters.append(self.i)
                return True

        return None

    def early_branching_lars(self, lb, curr_lp, incumb):
        if self.early_branch_lars:
            if incumb > 1e+10:
                return None

            lp_fractional = self.model.isGT(self.model.feasFrac(curr_lp), 0)
            if lp_fractional:
                g_max = (incumb - lb) / lb
                g_min = (incumb - curr_lp) / curr_lp
                cl1 = (g_max - g_min) / g_min < LARS_THRESHOLD
                cl2 = incumb > curr_lp

                if cl1 * cl2:
                    if self.verbose & V.EARLY_BRANCH:
                        print("--abort pricing due to Lars Jäger")
                    return True
        return None

    # -----------------------------------------------------------------------
    # Stabilization helpers
    # -----------------------------------------------------------------------

    @timeit_accumulate(timing_store["stab_utils"])
    def build_dual_vector(self, part_duals, disj_duals, res_duals, all_cons_names, cons_name_parts):
        orig_dual_sols = [part_duals, disj_duals, res_duals]
        out_dual_dict = duals_to_dict(all_cons_names, cons_name_parts, orig_dual_sols)
        out_dual_vector = np.fromiter((out_dual_dict[cons] for cons in all_cons_names), dtype=float)
        return orig_dual_sols, out_dual_dict, out_dual_vector

    @timeit_accumulate(timing_store["stab_utils"])
    def extract_smoothed_duals(self, smooth_dual_sols_dict, cons_name_parts, T):
        wentges_part_duals = {}
        wentges_disj_duals = {}
        wentges_res_duals = [0 for _ in T]

        smooth_duals = list(smooth_dual_sols_dict.values())

        for i, val in enumerate(smooth_duals):
            parts = cons_name_parts[i]
            ktype = parts[0]

            if ktype == 'part':
                j = int(parts[1])
                wentges_part_duals[j] = val
                continue

            elif ktype == 'disj':
                m, t = map(int, parts[1:])
                wentges_disj_duals[(m, t)] = val
                continue

            elif ktype == 'res':
                t = int(parts[-1])
                wentges_res_duals[t] = {0: val}
                continue

        return [wentges_part_duals, wentges_disj_duals, wentges_res_duals]

    @timeit_accumulate(timing_store["stab_utils"])
    def mispricing_update(self, cons_parts, n_conss, orig_dual_sols, smooth_dual_sols_nodir, subgrad_vec, subgrad_norm, curr_node, curr_node_depth):
        in_sep_vec, in_sep_norm = comp_in_sep(cons_parts=cons_parts, out_duals=orig_dual_sols, in_duals=smooth_dual_sols_nodir, n_conss=n_conss)  # TODO: compute norm when aggregating
        alpha_change = self.w_alpha

        if self.model.isLE(in_sep_norm, 0):
            self.w_alpha = f_decr(self.w_alpha)
            self.vprint(f"in & sep same point | iter {self.i}, node {curr_node} depth {curr_node_depth}: decreasing alpha by {abs(np.round(alpha_change - self.w_alpha, decimals=5))}, new alpha= {np.round(self.w_alpha, decimals=5)}")

        else:
            cos_angle = np.dot(subgrad_vec, in_sep_vec) / (subgrad_norm * in_sep_norm)

            self.vprint(f"cos angle is {'positive' if np.sign(cos_angle) > 0 else 'negative'}")
            if self.model.isGE(cos_angle, 0):
                self.w_alpha = f_decr(self.w_alpha)
                self.vprint(f"Misprice | iter {self.i}, node {curr_node} depth {curr_node_depth}: decreasing alpha by {abs(np.round(alpha_change - self.w_alpha, decimals=5))}, new alpha= {np.round(self.w_alpha, decimals=5)}")
            else:
                self.w_alpha = f_incr(self.w_alpha)
                self.vprint(f"Misprice | iter {self.i}, node {curr_node} depth {curr_node_depth}: increasing alpha by {abs(np.round(alpha_change - self.w_alpha, decimals=5))}, new alpha= {np.round(self.w_alpha, decimals=5)}")

            self.nr_misprices += 1

    def reset_stability_center(self):
        self.best_part_duals = None
        self.best_disj_duals = None
        self.best_res_duals = None
        self.best_lag_bound = 0
        self.curr_lag_bound = 0
        self.abs_best_lag_bound = 0
        self.nr_misprices = 0
        self.w_alpha = ALPHA_INIT
        self.local_wentges_bool = self.wentges_bool
        self.dir_wentges_bool = self.wentges_bool
        self.gap = []
        self.tailoff_count = 0

    @timeit_accumulate(timing_store["stab_utils"])
    def update_incumbent_duals(self, update_lag_bound, curr_lp, update_dual_sols, best_subgrad_normalized):
        if self._debug_checks:
            assert self.model.isGE(math.ceil(curr_lp), update_lag_bound)  # TODO: debug mode
        if update_lag_bound > self.best_lag_bound:
            update_part_duals, update_disj_duals, update_res_duals = update_dual_sols
            self.best_lag_bound = update_lag_bound
            self.best_part_duals = update_part_duals
            self.best_disj_duals = update_disj_duals
            self.best_res_duals = update_res_duals
            self.best_subgrad_normalized = best_subgrad_normalized

    @timeit_accumulate(timing_store["stab_utils"])
    def compute_tilde_duals(self, part_duals, disj_duals, res_duals):
        T = range(len(self.C_t))
        w  = self.w_alpha
        ow = 1.0 - w

        best_part = self.best_part_duals        # dict: j -> float
        best_disj = self.best_disj_duals        # dict: (m,t) -> float
        best_res  = self.best_res_duals         # list[dict], each with key 0

        wentges_part_duals = {k: w*v + ow*part_duals[k]   for k, v in best_part.items()}
        wentges_disj_duals = {k: w*v + ow*disj_duals[k]   for k, v in best_disj.items()}
        wentges_res_duals = [{0: w*br[0] + ow*rr[0]} for br, rr in zip(best_res, res_duals)]

        return [wentges_part_duals, wentges_disj_duals, wentges_res_duals]

    @timeit_accumulate(timing_store["stab_utils"])
    def compute_out_in(self, all_cons_names, all_cons_parts, out_dual_vector):
        in_duals = [self.best_part_duals, self.best_disj_duals, self.best_res_duals]
        in_dual_dict = duals_to_dict(all_cons_names, all_cons_parts, in_duals)
        in_dual_vector = np.fromiter((in_dual_dict[cons] for cons in in_dual_dict), dtype=float)

        out_in_vector = out_dual_vector - in_dual_vector
        out_in_norm = np.linalg.norm(out_in_vector)

        return in_dual_vector, out_in_vector, out_in_norm

    @timeit_accumulate(timing_store["stab_utils"])
    def compute_sep_duals(self, out_dual_vector, in_dual_vector, out_in_vector, out_in_norm):

        # g_n = g_in/||g_in||
        subgrad_in_normalized = self.best_subgrad_normalized

        # π_g = π_in + ||π_out - π_in|| * g_n
        g_dual_vector = in_dual_vector + out_in_norm * subgrad_in_normalized

        # γ = ( π_out - π_in , π_g - π_in ) (angle)
        # β = cos γ =  <π_out - π_in, g_n > / || π_out - π_in ||  (after simplifications)
        beta = np.dot(out_in_vector, subgrad_in_normalized) / out_in_norm
        rho_vector = beta * g_dual_vector + (1 - beta) * out_dual_vector

        # ρ_in = ρ - π_in
        rho_in_vector = rho_vector - in_dual_vector
        # ρ_in / || ρ_in ||
        rho_in_vec_norm1 = rho_in_vector / np.linalg.norm(rho_in_vector)
        # || π_tild - π_in || = (1 - α) ||π_out - π_in||
        tilde_in_norm = (1 - self.w_alpha) * out_in_norm

        # π_sep = π_in + || π_tild - π_in || * ρ_in / || ρ_in ||
        sep_dual_vector = in_dual_vector + tilde_in_norm * rho_in_vec_norm1

        # (π_sep)⁺ : projection onto "positive" orthant (assumes Ax ≥ b)
        mask_pos = out_dual_vector > 0
        sep_dual_vec_proj = np.where(mask_pos,
                                 np.maximum(sep_dual_vector, 0.0),
                                 np.minimum(sep_dual_vector, 0.0))
        return sep_dual_vec_proj

    # -----------------------------------------------------------------------
    # Column addition
    # -----------------------------------------------------------------------

    def add_min_red_cost_patterns(self, all_patterns, all_min_red_costs, dual_feas_tol_param, farkas=False):
        model = self.model
        add_var_objects = []
        for job_idx in self.J:
            curr_pattern = all_patterns.get(job_idx)
            if curr_pattern is None:
                continue
            if model.isLT(all_min_red_costs[job_idx], -dual_feas_tol_param):  # REV: handle tolerances
                var = self.add_pattern_column(curr_pattern)
                if var is not None:
                    add_var_objects.append(var)
        return add_var_objects

    @timeit_accumulate(timing_store["add_cols"])
    def add_pattern_column(self, to_add):  # TODO: optimize
        pattern_str = to_add.schedule_to_var_name()
        job_idx = to_add.job_id
        pattern_cost = to_add.cost

        if to_add.unique_id == -1:
            to_add.unique_id = pattern_str
        if self.rc_fixing_bool and self.schedule_id_to_var is not None:
            # necessary for rc_fixing
            dup_patt = to_add.unique_id not in self.schedule_id_to_var[job_idx]
            if not dup_patt:
                return None

        var_type = "C" if self.cg_bool else "I"
        new_var = self.model.addVar(vtype=var_type, lb=0, name=pattern_str, obj=pattern_cost, pricedVar=True)
        self.total_added_patterns += 1

        if self.schedule_id_to_var is not None:
            self.schedule_id_to_var[job_idx][to_add.unique_id] = new_var
        self.all_varschedules[job_idx].append(VarSchedule(new_var, to_add))  # added to communicate branching decisions to custom b&b
        self.all_vars_jobs[job_idx].append(new_var)
        self.all_schedules_jobs[job_idx].append(to_add)

        get_cons = self.model.getTransformedCons
        add_coeff = self.model.addConsCoeff
        disj_c = self.disj_c
        res_c = self.res_c
        KP = self.KP
        all_vars_jmt = self.all_vars_jmt
        all_schedules_jmt = self.all_schedules_jmt

        # Partitioning constraint
        add_coeff(get_cons(self.part_c[job_idx]), new_var, 1)

        for (m, t) in to_add.mt_pairs:
            # Disjunction constraints
            disj = get_cons(disj_c[(m, t)])
            add_coeff(disj, new_var, 1)

            # Resource constraints
            for kp_id, kp_cstr in res_c[t].items():
                coeff = KP[kp_id].coefficients[m]
                cstr = get_cons(kp_cstr)
                add_coeff(cstr, new_var, coeff)

            # Variables data
            all_vars_jmt[job_idx, m, t].append(new_var)
            all_schedules_jmt[job_idx, m, t].append(to_add)

        return new_var

    @timeit_accumulate(timing_store["add_cols"])
    def verify_reduced_costs_and_update_bound(self, add_var_objects, all_min_red_costs, dual_feas_tol_param, farkas=False):
        # TODO: debug mode
        if farkas:
            return
        model = self.model
        for var in add_var_objects:
            job_idx = int(var.name.split("-")[0])
            var_red_cost = model.getVarRedcost(var)
            diff = abs(var_red_cost - all_min_red_costs[job_idx])
            assert diff <= dual_feas_tol_param  # NB: holds wrt. duals of current LP (and not for SG)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    @timeit_accumulate(timing_store["other_pricing"])
    def track_progress(self, curr_lp, curr_node, curr_node_obj):
        self.lp_evol.append(curr_lp)

        self.vprint()
        self.vprint("--curr node  \t", curr_node)
        self.vprint("--iter       \t", self.i)
        self.vprint("--lp obj     \t", curr_lp)
        self.vprint("--loc dual bd\t", curr_node_obj.getLowerbound())
        self.vprint("--curr Lag bd\t", self.curr_lag_bound)
        self.vprint("--best Lag bd\t", self.abs_best_lag_bound)
        self.vprint("--nr add vars\t", self.total_added_patterns)
        self.vprint()

        lb = max(self.model.getDualbound(), self.best_lag_bound)  # REV: min if max. problem
        if self.total_added_patterns == 0 or abs(self.model.feasCeil(curr_lp) - lb) < 1:
            self.req_iters.append(self.i)

    # -----------------------------------------------------------------------
    # Reduced cost fixing
    # -----------------------------------------------------------------------

    @iter_accumulate(iter_store["rc_fixing"])
    @timeit_accumulate(timing_store["rc_fixing"])
    def rc_fixing_loop(self, primal_bd):
        mod = self.model
        scip_infty = mod.isInfinity
        curr_node_obj = mod.getCurrentNode()

        if not scip_infty(primal_bd):
            subtree_dual_bd = curr_node_obj.getLowerbound()
            primal_update = 0
            if primal_bd < self.best_primal:
                self.best_primal = primal_bd
                primal_update += 1

            # vanilla rc fixing: iterate all variables, fix those with rc > gap + 1
            if primal_update > 0:
                gap = self.best_primal - subtree_dual_bd
                get_rc = mod.getVarRedcost
                for j_vars in self.all_vars_jobs.values():
                    for var in j_vars:
                        var_rc = get_rc(var)
                        if var_rc > gap + RC_FIXING_SLACK:
                            mod.chgVarUb(var, 0)

    # -----------------------------------------------------------------------
    # Probability complementary pricing
    # -----------------------------------------------------------------------

    @timeit_accumulate(timing_store["comp-proba"])
    def probability_complementary_pricing(self, isLT, dual_feas_tol_param, T, rc_term, orig_dual_sols, arcs_to_forbid, arcs_to_force, farkas, root_bool):

        part_duals, _, _ = orig_dual_sols

        model_get = self.model.getSolVal
        sched = self.all_varschedules
        rand = random.random
        randsd = random.seed
        arcs_proba_forbid = set(arcs_to_forbid)
        curr_i = self.i

        J, M = self.J, self.M
        Z_jmt = np.zeros((len(J), len(M), len(T)), dtype=float)

        for j in J:
            for col in sched[j]:
                val_col = model_get(sol=None, expr=col.var)
                mt_pairs = col.schedule.mt_pairs
                ms, ts = zip(*mt_pairs)
                Z_jmt[j, ms, ts] += val_col

        # use only cells where Z > 0
        js, ms, ts = np.nonzero(Z_jmt)

        for j_idx, m_idx, t_idx in zip(js, ms, ts):
            p = Z_jmt[j_idx, m_idx, t_idx]
            randsd(int(m_idx + 10 * j_idx + 100 * t_idx + 1000 * curr_i))
            if rand() < p:
                arcs_proba_forbid.add((j_idx, m_idx, t_idx))

        _, all_patterns_proba = pricing_solver(self.J, self.W_m, self.KP, self.dur, self.seq, self.C_t, dual_sols=orig_dual_sols, rc_term=rc_term,
                                               forbidden_arcs=arcs_proba_forbid, forced_arcs=arcs_to_force,
                                               farkas_bool=farkas, partial_bool=False, root_bool=root_bool)
        proba_patts = []
        sum_min_red_cost = 0
        for job_idx, patt in all_patterns_proba.items():
            if patt:
                orig_red_cost = compute_pattern_red_cost(pattern_pairs=patt.mt_pairs, rc_term=rc_term, part_dual_term=part_duals[job_idx])
                if isLT(orig_red_cost, -dual_feas_tol_param):
                    proba_patts.append(patt)
                    sum_min_red_cost += orig_red_cost

        return proba_patts, sum_min_red_cost

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def feasCeil_cg(self, val):
        if self.cg_bool:
            return val
        return self.model.feasCeil(val)

    def vprint(self, *args, **kwargs):
        if self.verbose & V.GENERAL:
            print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def duals_to_dict(all_cons_names, all_cons_parts, dual_sols):
    dual_part, dual_disj, dual_res = dual_sols
    dual_vec = {_: 0.0 for _ in all_cons_names}

    for i, cons_parts in enumerate(all_cons_parts):
        cons_type = cons_parts[0]
        cons_name = all_cons_names[i]

        # partitioning constraints
        if cons_type == 'part':
            j = int(cons_parts[1])
            dual_vec[cons_name] = dual_part[j]
            continue

        # disjunction constraints
        if cons_type == 'disj':
            m, t = map(int, cons_parts[1:])
            dual_vec[cons_name] = dual_disj[(m, t)]
            continue

        # resource constraints
        if cons_type == 'res':
            t = int(cons_parts[-1])
            dual_vec[cons_name] = dual_res[t][0]  # TODO: KP!
            continue

    return dual_vec
