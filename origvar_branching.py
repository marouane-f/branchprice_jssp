from pyscipopt import Branchrule, SCIP_RESULT
from utils import timing_store, timeit_accumulate, compute_proj_Z, node_dist, mean_or_one, dive_params, revert_dive_params

import numpy as np
from collections import defaultdict
from itertools import chain
from copy import deepcopy

EPS = 1e-6


class OrigVar(Branchrule):

    def __init__(self, jobs, machines, powers, duration, job_sequence, Cost_t, KP,
                 all_vars_jmt, all_vars_jobs, added_probing_vars, all_schedules_jobs, all_varschedules,
                 forbidden_arcs_prob, forced_arcs_prob,
                 pseudocost_track, pseudocost_comp, pseudocost_op_comp, pseudocost_time_comp,
                 reli_budget, n_pricing_iters, strong_br_bool, dom_propag_bool,
                 verbose_bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- problem data ---
        self.J = jobs
        self.M = machines
        self.W_m = powers
        self.dur = duration
        self.seq = job_sequence
        self.C_t = Cost_t
        self.KP = KP

        # --- SCIP variable references ---
        self.all_vars_jmt = all_vars_jmt            # {(j,m,t): [cols]} — columns covering Z_jmt = 1
        self.all_vars_jobs = all_vars_jobs           # {j: [cols]}       — all columns for job j
        self.all_varschedules = all_varschedules     # {j: [VarSchedule]}— columns with schedule metadata
        self.all_schedules_jobs = all_schedules_jobs
        self.added_probing_vars = added_probing_vars # columns added during probing (reset after each round)

        # --- branching arc sets: per-node (keyed by node number, inherited by children at branching time) ---
        self.forbidden_arcs = defaultdict(lambda: set())  # Z_jmt = 0 decisions accumulated on the path
        self.forced_arcs = defaultdict(lambda: set())     # Z_jmt = 1 decisions accumulated on the path

        # --- branching arc sets: probing (shared mutable refs, reset to parent state before each SB call) ---
        self.forbidden_arcs_prob = forbidden_arcs_prob
        self.forced_arcs_prob = forced_arcs_prob

        # --- branching strategy flags ---
        self.strong_br = strong_br_bool
        self.dom_propag = dom_propag_bool
        self.dive_params_bool = 1           # TODO: add arg
        self.weight_pscost_bool = True      # TODO: add arg
        self.fallout_rule = 1   # 1 = most spread | 2 = approx. pseudocost  (fallback when no relpscost candidate)

        # --- pseudocost tables (keyed by (j,m,t), direction in {0,1}) ---
        self.pseudocost_track = pseudocost_track          # {node_nr: (jmt, metadata)} — updated on next node visit
        self.pseudocost_comp = pseudocost_comp            # {jmt: {0: [gains], 1: [gains]}}
        self.pseudocost_op_comp = pseudocost_op_comp      # same, aggregated by (j,m)
        self.pseudocost_time_comp = pseudocost_time_comp  # same, aggregated by t
        self.pscost_comp_node = defaultdict(lambda: {0: [], 1: []})  # {jmt: {dir: [(node_id, gain)]}}

        # --- reliability strong branching ---
        self.n_pricing_iters = n_pricing_iters  # pricing rounds per SB LP solve (0 = dive/RMP only)
        self.reli_budget = reli_budget           # max re-evaluations to improve pseudocost reliability
        self.reli_calls = 0
        self.reli_strongbr = False

        # --- tree tracking ---
        self.node_id = {1: ''}                           # {node_nr: binary string path from root}
        self.track_branches = defaultdict(dict, {1: {}}) # {node_nr: {jmt: direction}} — full branching history

        self.verbose = verbose_bool

    @timeit_accumulate(timing_store["branch"])
    def branchexeclp(self, allowaddcons):
        
        mod = self.model
        curr_lp = mod.getLPObjVal()
        node_depth = mod.getCurrentNode().getDepth()
        parent_node = mod.getCurrentNode().getNumber()
        self.reli_strongbr = False
        if self.strong_br:
            self.update_pseudocosts(parent_node, curr_lp)
        get_lp = mod.getSolVal

        J = self.J
        nJ = len(J)
        M = self.M
        nM = len(M)
        nT = len(self.C_t)
        T = range(nT)

        patterns, sol_coeffs = self.extract_lp_candidates(self.all_varschedules)
        Z_jmt = compute_proj_Z(nT, nM, nJ, patterns, sol_coeffs)

        parent_forbidden_arcs = self.forbidden_arcs[parent_node]
        parent_forced_arcs = self.forced_arcs[parent_node]

        if not self.strong_br:
            best_jmt = fallout_branching_rule(J, M, T, Z_jmt, self.all_vars_jmt, self.all_varschedules, get_lp, rule=self.fallout_rule)
            if best_jmt != 0:
                self.branch_on_Z_jmt(best_jmt, Z_jmt, J, parent_node, parent_forbidden_arcs, parent_forced_arcs, curr_lp, node_depth)
                return {"result": SCIP_RESULT.BRANCHED}
            else:
                return {"result": SCIP_RESULT.DIDNOTFIND}

        self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs) # important

        priced_strong_branching = bool(self.n_pricing_iters)

        # --- near-root (depth 0–2): strong branching over all candidates, RMP only ---
        if 0 <= node_depth <= 2:        # TODO: add arg for max strong-branching depth
            # candidate set at root is all (j,m) pairs; at deeper nodes it shrinks and mixes in pseudocost

            if node_depth == 0:
                top_k = nJ * nM
                if self.fallout_rule == 1:
                    jmt_cands = choose_spread_frac_jmt_topk(J, M, T, Z_jmt, top_k=top_k)
                else:
                    jmt_cands = choose_approx_pscost_jmt_topk(J, M, T, Z_jmt, self.all_vars_jmt, self.all_varschedules, get_lp, top_k=top_k)
            else:
                top_k = nJ * nM // min(max(node_depth, 1), 2)
                jmt_cands = choose_mix_frac_jmt_topk(J, M, T, Z_jmt, 
                                                     self.pseudocost_op_comp, self.pseudocost_time_comp, 
                                                     top_k=top_k,
                                                     mix_weight=0.05)
            
            strong_br_scores = {}
            best_strong_br_score = 0
            best_strong_br_cand = 0
            best_lp_right = 0
            best_lp_left = 0
            iters_no_improv = 0

            self.vprint(f"--quick strong branching \t considering {len(jmt_cands)} candidates")
            
            if self.dive_params_bool:
                dive_params(mod)

            for best_jmt in jmt_cands:
                # right branch -> forbid
                curr_lp_right, strong_br_scores = self.quick_strong_branching(J, br_dir=0, best_jmt=best_jmt, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=0)
                if curr_lp_right > best_lp_right:
                    best_lp_right = curr_lp_right
                self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)  # important

                # left branch -> force
                curr_lp_left, strong_br_scores = self.quick_strong_branching(J, br_dir=1, best_jmt=best_jmt, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=0)
                if curr_lp_left > best_lp_left:
                    best_lp_left = curr_lp_left
                self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)  # important

                if strong_br_scores[best_jmt] > best_strong_br_score:
                    iters_no_improv = 0 # reset
                    best_strong_br_cand = best_jmt
                    best_strong_br_score = strong_br_scores[best_jmt]
                else:
                    iters_no_improv += 1 # increment

            if self.dive_params_bool:
                revert_dive_params(mod)

            # --- optional second pass: re-evaluate top-3 candidates with full pricing ---
            if priced_strong_branching:
                jmt_cands_topk = sorted(strong_br_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                strong_br_scores = {}
                best_strong_br_score = 0
                best_strong_br_cand = 0
                best_lp_right = 0
                best_lp_left = 0

                self.vprint(f"-- priced strong branching \t considering {len(jmt_cands_topk)} candidates")

                for str_i, best_jmt in enumerate(jmt_cands_topk):
                    self.vprint(f"-- priced strong branching \t candidate {str_i}")

                    best_jmt = best_jmt[0]

                    # right branch -> forbid
                    curr_lp_right, strong_br_scores = self.quick_strong_branching(J, br_dir=0, best_jmt=best_jmt, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=self.n_pricing_iters)
                    if curr_lp_right > best_lp_right:
                        best_lp_right = curr_lp_right
                    self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)  # /!\ important

                    # left branch -> force
                    curr_lp_left, strong_br_scores = self.quick_strong_branching(J, br_dir=1, best_jmt=best_jmt, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=self.n_pricing_iters)
                    if curr_lp_left > best_lp_left:
                        best_lp_left = curr_lp_left
                    self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)  # /!\ important

                    if strong_br_scores[best_jmt] > best_strong_br_score:
                        best_strong_br_cand = best_jmt
                        best_strong_br_score = strong_br_scores[best_jmt]
                self.added_probing_vars = []  # reset variables added in current strong branching
                self.vprint(f"-- end priced strong branching")
            if best_strong_br_cand:
                best_jmt = best_strong_br_cand
            else:
                best_jmt = fallout_branching_rule(J, M, T, Z_jmt, self.all_vars_jmt, self.all_varschedules, get_lp, rule=self.fallout_rule)
            self.branch_on_Z_jmt(best_jmt, Z_jmt, J, parent_node, parent_forbidden_arcs, parent_forced_arcs, curr_lp, node_depth)
            return {"result": SCIP_RESULT.BRANCHED}

        # --- deeper nodes: reliability pseudocost branching ---
        else:
            # restrict to variables that have been evaluated at least once (have pseudocost history)
            filtered_pseudocost, filtered_Z, filt_weighted_pscost = self.filter_frac_pseudocost(Z_jmt)

            jmt_card = {k: sum(1 for x in v if x.getUbLocal() != 0) for k, v in self.all_vars_jmt.items()}  # only variables that have not been fixed

            if len(filtered_pseudocost) > 0:
                # --- reliability phase: add observations for top candidates with too few samples ---
                if self.reli_calls < self.reli_budget:
                    self.reli_strongbr = True

                    if self.weight_pscost_bool:
                        best_jmtk = comp_weight_pscost_infer_score(filt_weighted_pscost, filtered_Z, jmt_card, topk=3, node_id=self.node_id[parent_node])
                    else:
                        best_jmtk = comp_pscost_infer_score_topk(filtered_pseudocost, filtered_Z, jmt_card, topk=3)

                    if self.dive_params_bool:
                        dive_params(mod)

                    for jmt_k in best_jmtk:
                        if len(filtered_pseudocost[jmt_k][0]) + len(filtered_pseudocost[jmt_k][1]) < 6: # TODO: add reliability threshold arg
                            self.vprint(f'-- quick strong branching to improve reliability')
                            strong_br_scores = {}
                            self.quick_strong_branching(J, br_dir=0, best_jmt=jmt_k, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=self.n_pricing_iters // 2, reliable_eval=True)
                            self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)
                            self.quick_strong_branching(J, br_dir=1, best_jmt=jmt_k, curr_lp=curr_lp, strong_br_scores=strong_br_scores, pricing_iters=self.n_pricing_iters // 2, reliable_eval=True)
                            self.reset_arc_sets_probing(parent_forbidden_arcs, parent_forced_arcs)
                            self.reli_calls += 1
                    if self.dive_params_bool:
                        revert_dive_params(mod)

                # --- selection: score by pseudocost × inferability, pick best ---
                if self.weight_pscost_bool:
                    best_jmt = comp_weight_pscost_infer_score(filt_weighted_pscost, filtered_Z, jmt_card, topk=1, node_id=self.node_id[parent_node])
                else:
                    best_jmt = comp_pscost_infer_score_topk(filtered_pseudocost, filtered_Z, jmt_card, topk=1)
            else:
                # no pseudocost history yet — fall back to heuristic rule
                best_jmt = fallout_branching_rule(J, M, T, Z_jmt, self.all_vars_jmt, self.all_varschedules, get_lp, rule=self.fallout_rule)

            if best_jmt != 0:
                self.branch_on_Z_jmt(best_jmt, Z_jmt, J, parent_node, parent_forbidden_arcs, parent_forced_arcs, curr_lp, node_depth)

                return {"result": SCIP_RESULT.BRANCHED}
            else:
                return {"result": SCIP_RESULT.DIDNOTFIND}  # should not occur (normally) at frac node


    def extract_lp_candidates(self, all_varschedules):
        lpcandssol = []
        lpcands_patt = []

        for varsch in list(chain.from_iterable(all_varschedules.values())):
            var = varsch.var 
            sch = varsch.schedule
            val_x = self.model.getSolVal(sol=None, expr=var)
            if self.model.isGT(val_x, 0): # get all x > 0 to project properly in Z_jmt
                lpcands_patt.append(sch)
                lpcandssol.append(val_x)

        return lpcands_patt, lpcandssol
    
    def create_single_child(self, parent_forbidden_arcs, parent_forced_arcs, nodeselprio):
        child = self.model.createChild(nodeselprio=nodeselprio, estimate=self.model.getLocalEstimate())
        child_id = child.getNumber()
        self.forbidden_arcs[child_id] = set(parent_forbidden_arcs)
        self.forced_arcs[child_id] = set(parent_forced_arcs)
        return child_id

    def reset_arc_sets_probing(self, parent_forbidden_arcs, parent_forced_arcs):
        self.forbidden_arcs_prob.clear()
        self.forbidden_arcs_prob.update(parent_forbidden_arcs)
        self.forced_arcs_prob.clear()
        self.forced_arcs_prob.update(parent_forced_arcs)

    def branch_on_Z_jmt(self, best_jmt, Z_jmt, J, parent_node, parent_forbidden_arcs, parent_forced_arcs, curr_lp, node_depth):
        right_child_id = self.create_single_child(parent_forbidden_arcs, parent_forced_arcs, nodeselprio=100)
        left_child_id = self.create_single_child(parent_forbidden_arcs, parent_forced_arcs, nodeselprio=100)
        parent_node = self.model.getCurrentNode().getNumber()
        parent_node_id = self.node_id[parent_node]
        self.node_id[right_child_id] = parent_node_id + '1'
        self.node_id[left_child_id] = parent_node_id + '0'
        self.track_branches[right_child_id] = {**self.track_branches[parent_node], best_jmt: 1}
        self.track_branches[left_child_id] = {**self.track_branches[parent_node], best_jmt: 0}

        j0, m0, t0 = best_jmt
        nT = len(self.C_t)
        self.vprint(f"--Z_jmt branching (depth {node_depth})")

        # ---------------up/right branch: force Z_j0m0t0 = 1---------------
        forbidden_right = self.forbidden_arcs[right_child_id]
        forced_right = self.forced_arcs[right_child_id]
        forced_right.add(best_jmt)
        if self.dom_propag:
            self._propagate_force(j0, m0, t0, J, nT, parent_forced_arcs, parent_forbidden_arcs,
                                  forced_right, forbidden_right)

        # ---------------down/left branch: forbid Z_j0m0t0 = 0---------------
        forbidden_left = self.forbidden_arcs[left_child_id]
        forbidden_left.add(best_jmt)
        if self.dom_propag:
            self._propagate_forbid(j0, m0, t0, nT, parent_forced_arcs, forbidden_left)

        var_val = Z_jmt[t0][m0][j0]
        self.pseudocost_track[right_child_id] = (best_jmt, {
            "parent": parent_node, "dir": 1, "val": var_val, "currlp": curr_lp, "nextlp": None
        })
        self.pseudocost_track[left_child_id] = (best_jmt, {
            "parent": parent_node, "dir": 0, "val": var_val, "currlp": curr_lp, "nextlp": None
        })

    def _propagate_force(self, j0, m0, t0, J, nT, parent_forced_arcs, parent_forbidden_arcs,
                         forced_set, forbidden_set):
        """Implied arc fixings when Z_j0m0t0 is forced to 1."""
        # 1. Precedence: predecessor ops must finish before t0, successors must start after t0
        curr_seq = np.array(self.seq[j0]) - 1
        seq_idx = list(curr_seq).index(m0)
        if seq_idx == 0:  # first operation: all successors must start after t0
            for m2 in curr_seq[seq_idx + 1:]:
                for tau in range(t0):
                    forbidden_set.add((j0, m2, tau))
        elif seq_idx == len(curr_seq) - 1:  # last operation: all predecessors must end before t0
            for m1 in curr_seq[:seq_idx]:
                for tau in range(t0 + 1, nT):
                    forbidden_set.add((j0, m1, tau))
        else:  # middle operation: both directions
            for m1 in curr_seq[:seq_idx]:
                for tau in range(t0 + 1, nT):
                    forbidden_set.add((j0, m1, tau))
            for m2 in curr_seq[seq_idx + 1:]:
                for tau in range(t0):
                    forbidden_set.add((j0, m2, tau))

        # 2. Disjunction: at time t0, no other job on m0 and j0 on no other machine
        for k in J:
            if k != j0:
                forbidden_set.add((k, m0, t0))  # machine disjunction
        for n in self.M:
            if n != m0:
                forbidden_set.add((j0, n, t0))  # job disjunction

        # 3.1 Non-preemption: two forced slots for (j0,m0) imply all slots between them are forced
        forced_t = [t1 for (j1, m1, t1) in parent_forced_arcs if j1 == j0 and m1 == m0]
        if forced_t:
            min_t, max_t = min(forced_t), max(forced_t)
            if t0 < min_t:
                forced_set.update((j0, m0, tau) for tau in range(t0 + 1, min_t))
            elif t0 > max_t:
                forced_set.update((j0, m0, tau) for tau in range(max_t + 1, t0))

        # 3.2 Non-preemption: a forced slot adjacent to a forbidden slot bounds the active window
        forbidden_t_same = [t1 for (j1, m1, t1) in parent_forbidden_arcs if j1 == j0 and m1 == m0]
        t_above = min((t1 for t1 in forbidden_t_same if t1 > t0), default=nT)
        t_below = max((t1 for t1 in forbidden_t_same if t1 < t0), default=-1)
        if t_above < nT:
            forbidden_set.update((j0, m0, tau) for tau in range(t_above, nT))
        if t_below > -1:
            forbidden_set.update((j0, m0, tau) for tau in range(t_below))

        # 4. Cardinality: if rhs slots at time t are already forced, forbid all remaining (j,m) at t
        rhs = self.KP[0].rhs  # assuming single knapsack constraint
        jmt_groups = defaultdict(list)
        for _j, _m, _t in forced_set:
            jmt_groups[_t].append((_j, _m, _t))
        for t, lst in jmt_groups.items():
            if len(lst) >= rhs:
                for j in J:
                    for m in self.M:
                        if (j, m, t) not in lst:
                            forbidden_set.add((j, m, t))

    def _propagate_forbid(self, j0, m0, t0, nT, parent_forced_arcs, forbidden_set):
        """Implied arc fixings when Z_j0m0t0 is forbidden (set to 0)."""
        # 3.2 Non-preemption: a forbidden slot adjacent to a forced slot bounds the active window
        forced_t_same = [t1 for (j1, m1, t1) in parent_forced_arcs if j1 == j0 and m1 == m0]
        t_above = min((t1 for t1 in forced_t_same if t1 > t0), default=nT)
        t_below = max((t1 for t1 in forced_t_same if t1 < t0), default=-1)
        if t_above < nT:
            forbidden_set.update((j0, m0, tau) for tau in range(t_above))
        if t_below > -1:
            forbidden_set.update((j0, m0, tau) for tau in range(t_below, nT))

    @timeit_accumulate(timing_store["strongbranch"])
    def quick_strong_branching(self, J, br_dir, best_jmt, curr_lp, strong_br_scores, pricing_iters, reliable_eval=False, single_IO=False):
        j0, m0, t0 = best_jmt
        curr_node_id = self.node_id[self.model.getCurrentNode().getNumber()]
        next_node_id = curr_node_id + str(br_dir)

        if single_IO:
            fixed_vars = []
        else:
            self.enter_strbr(pricing_iters)

        def fix_ub(var):
            if single_IO:
                fixed_vars.append((var, var.getUbLocal()))
            if pricing_iters:
                self.model.chgVarUbProbing(var, 0.0)
            else:
                self.model.chgVarUbDive(var, 0.0)

        self._apply_sb_fixings(br_dir, j0, m0, t0, J, fix_ub)

        if pricing_iters:
            self.model.propagateProbing(-1)
            self.model.solveProbingLPWithPricing(pretendroot=False, displayinfo=False, maxpricerounds=pricing_iters)
        else:
            self.model.solveDiveLP()

        if self.model.getLPObjVal() >= 1e+10:
            br_chg = 0
        else:
            approx_lp = (self.model.getLPObjVal() + self.model.getCurrentNode().getLowerbound()) / 2  # since no pricing, estimate gain by averaging with dual bound
            br_chg = abs(approx_lp - curr_lp)

        self._record_pscost(best_jmt, br_dir, br_chg, next_node_id, pricing_iters, reliable_eval)

        # strong-branching score: product of down- and up-gains (set on dir=0, multiplied on dir=1)
        if br_dir == 0:
            strong_br_scores[best_jmt] = max(br_chg, EPS)
        elif br_dir == 1:
            strong_br_scores[best_jmt] *= max(br_chg, EPS)

        if single_IO:
            for (var, pre_ub) in fixed_vars:
                self.model.chgVarUbDive(var, pre_ub)
        else:
            self.exit_strbr(pricing_iters)

        return self.model.getLPObjVal(), strong_br_scores

    def _apply_sb_fixings(self, br_dir, j0, m0, t0, J, fix_ub):
        """Fix variable upper bounds in the current dive/probing context for one SB direction."""
        # right branch -> forbid Z_j0m0t0
        if br_dir == 0:
            self.forbidden_arcs_prob.add((j0, m0, t0))
            for var in self.all_vars_jmt[j0, m0, t0]:
                fix_ub(var)

        # left branch -> force Z_j0m0t0 = 1
        elif br_dir == 1:
            self.forced_arcs_prob.add((j0, m0, t0))
            if self.dom_propag:
                # only most impactful propagations to not overload the iteration
                for k in J:
                    if k != j0:
                        self.forbidden_arcs_prob.add((k, m0, t0))
                        for var in self.all_vars_jmt[k, m0, t0]:
                            fix_ub(var)  # machine disjunction: forbid Z_kmt for k != j0

                for n in self.M:
                    if n != m0:
                        for var in self.all_vars_jmt[j0, n, t0]:
                            fix_ub(var)  # job disjunction: forbid Z_jnt for n != m0

                for var in self.all_vars_jobs[j0]:
                    b = int(var.name[2:].split("_")[m0])
                    a = b - self.dur[j0][m0] + 1
                    if a > t0 or b < t0:
                        fix_ub(var)  # forbid schedules for j0 whose execution window excludes t0

    def _record_pscost(self, best_jmt, br_dir, br_chg, next_node_id, pricing_iters, reliable_eval):
        """Append or overwrite pseudocost observations for all four tracking tables.

        RMP-only evaluations append a new entry; a subsequent priced evaluation overwrites
        the last entry ([-1]) with the more accurate estimate.
        """
        if pricing_iters == 0 or reliable_eval:
            self.pscost_comp_node[best_jmt][br_dir].append((str(br_dir), br_chg))  # RMP-only: node tagged as root direction
            self.pseudocost_comp[best_jmt][br_dir].append(br_chg)
            self.pseudocost_op_comp[best_jmt[0], best_jmt[1]][br_dir].append(br_chg)
            self.pseudocost_time_comp[best_jmt[-1]][br_dir].append(br_chg)
        else:
            self.pscost_comp_node[best_jmt][br_dir][-1] = (next_node_id, br_chg)
            self.pseudocost_comp[best_jmt][br_dir][-1] = br_chg
            self.pseudocost_op_comp[best_jmt[0], best_jmt[1]][br_dir][-1] = br_chg
            self.pseudocost_time_comp[best_jmt[-1]][br_dir][-1] = br_chg

    @timeit_accumulate(timing_store["in_out_strbr"])
    def enter_strbr(self, pricing_iters):
        if pricing_iters:
            self.model.startProbing()
        else:
            self.model.startDive()

    @timeit_accumulate(timing_store["in_out_strbr"])
    def exit_strbr(self, pricing_iters):
        if pricing_iters:
            self.model.endProbing()
        else:
            self.model.endDive()

    def filter_frac_pseudocost(self, Z_jmt, EPS=1e-6):
        local_pseudo_cost = deepcopy(self.pseudocost_comp)
        filtered_pseudocost = {}
        filt_weighted_pscost = {}
        filtered_Z = {}

        for (j, m, t), val in local_pseudo_cost.items():
            z_val = Z_jmt[t][m][j]
            if EPS < z_val < 1 - EPS:
                filtered_pseudocost[(j, m, t)] = val
                filt_weighted_pscost[(j, m, t)] = self.pscost_comp_node[(j, m, t)]
                filtered_Z[(j, m, t)] = z_val

        return filtered_pseudocost, filtered_Z, filt_weighted_pscost

    @timeit_accumulate(timing_store["pscost_update"])
    def update_pseudocosts(self, parent_node, curr_lp):
        if parent_node < 2 or parent_node not in self.pseudocost_track:
            return

        var_br, pscost_data = self.pseudocost_track[parent_node]
        pscost_data['nextlp'] = curr_lp
        br_dir = pscost_data['dir']
        br_val = pscost_data['val']

        denom = abs(br_dir - br_val)
        if denom < 1e-8:  # avoid division by zero
            return

        br_chg = abs(pscost_data['nextlp'] - pscost_data['currlp']) / denom

        self.pscost_comp_node[var_br][br_dir].append((self.node_id[parent_node], br_chg))
        self.pseudocost_comp[var_br][br_dir].append(br_chg)
        self.pseudocost_op_comp[(var_br[0], var_br[1])][br_dir].append(br_chg)
        self.pseudocost_time_comp[var_br[-1]][br_dir].append(br_chg)

    def vprint(self, *args, **kwargs):
        if self.verbose & 1 << 1:
            print(*args, **kwargs)

def fallout_branching_rule(J, M, T, Z_jmt, all_vars_jmt, all_varschedules, get_lp, rule):
    if rule == 1:
        return choose_frac_jmt_spread(J, M, T, Z_jmt)
    return choose_approx_pscost_jmt_topk(J, M, T, Z_jmt, all_vars_jmt, all_varschedules, get_lp, top_k=1)
     

@timeit_accumulate(timing_store["frac_t"])
def choose_frac_jmt_spread(J, M, T, Z_jmt):
    """ Branch on (j, m, t) where (j, m) is least/most spread across horizon and t has highest Z_jmt """
    jm_count = defaultdict(list)
    for t in T:
        for m in M:
            for j in J:
                val = Z_jmt[t][m][j]
                if val * (1 - val) > 1e-6:
                    jm_count[(j, m)].append((t, val))

    if not jm_count:
        return 0

    # (j, m) with least/most t entries
    best_jm = max(jm_count.items(), key=lambda x: len(x[1]))[0]
    j, m = best_jm

    # pick t with highest Z_jmt
    best_t = max(jm_count[(j, m)], key=lambda x: x[1])[0]

    return (j, m, best_t)


def choose_spread_frac_jmt_topk(J, M, T, Z_jmt, top_k=10):
    """
    Return top-k (j, m, t) triples where (j, m) has the least number of fractional Z_jmt values over T.
    Within each (j, m), pick t with highest Z_jmt (or Z*(1-Z))
    """
    jm_to_frac = defaultdict(list)

    for t in T:
        for m in M:
            for j in J:
                val = Z_jmt[t][m][j]
                if EPS < val * (1 - val) < 1 - EPS:
                    jm_to_frac[(j, m)].append((t, val))

    if not jm_to_frac:
        return 0

    # Find top_k (j,m) with largest number of fractional time steps
    least_spread_jm = sorted(jm_to_frac.items(), key=lambda x: len(x[1]), reverse=True)[:top_k]

    # For each, pick the t with highest fractional score
    top_entries = [(j, m, max(t_scores, key=lambda x: x[1])[0]) for (j, m), t_scores in least_spread_jm]

    return top_entries

def choose_approx_pscost_jmt_topk(J, M, T, Z_jmt, all_vars_jmt, all_varschedules, get_lp, top_k=10):
    """Rank fractional (j,m,t) triples by approximate up-pseudocost derived from current LP column costs."""
    all_jmt = {}

    for t in T:
        for m in M:
            for j in J:
                val = Z_jmt[t][m][j]
                if EPS < val * (1 - val) < 1 - EPS:
                    jmt_vars = all_vars_jmt[j, m, t]
                    jmt_objs = np.array([var.getObj() for var in jmt_vars])
                    jmt_lp = np.array([get_lp(sol=None, expr=var) for var in jmt_vars])
                    C_up = jmt_lp @ jmt_objs + (1 - val) * np.mean(jmt_objs)
                    C_curr = jmt_objs @ jmt_lp
                    all_jmt[j, m, t] = max(C_up - C_curr, EPS)

    if not all_jmt:
        print("no branching cand found!")
        return 0

    if top_k == 1:
        return min(all_jmt, key=all_jmt.get)

    return [k for k, _ in sorted(all_jmt.items(), key=lambda x: x[1])[:top_k]]

def choose_mix_frac_jmt_topk(J, M, T, Z_jmt, pscost_op, pscost_t, top_k=10, mix_weight=0.05):
    """Top-k (j,m,t) by spread score augmented with a small pseudocost term (mix_weight controls the blend)."""
    jm_to_frac = defaultdict(list)

    for t in T:
        for m in M:
            for j in J:
                val = Z_jmt[t][m][j]
                if EPS < val * (1 - val) < 1 - EPS:
                    jm_to_frac[(j, m)].append((t, val))

    if not jm_to_frac:
        return 0

    jm_spread_pscost = {}
    for (i, x) in jm_to_frac.items():
        jm_spread_pscost[i] = len(x)
        val_i = pscost_op.get(i)
        if val_i:
            mean0 = max(mean_or_one(val_i[0]), 1e-6)
            mean1 = max(mean_or_one(val_i[1]), 1e-6)
            jm_spread_pscost[i] = len(x) + mix_weight * mean0 * mean1

    # Find top_k (j,m) with largest number of fractional time steps
    least_spread_pscost_jm = sorted(jm_spread_pscost.items(), key=lambda x: x[1], reverse=True)[:top_k]

    pscost_t_scores = {}
    for t, score_t in pscost_t.items():
        pscost_t_scores[t] = mean_or_one(score_t[0]) * mean_or_one(score_t[1])

    top_entries = []
    for (j, m), _ in least_spread_pscost_jm:
        t_max = 0
        t_max_sc = 0
        for t0, sc in jm_to_frac[(j, m)]:
            t0_sc = sc
            val = pscost_t_scores.get(t0)
            if val:
                t0_sc += mix_weight * val
            if t0_sc > t_max_sc:
                t_max = t0
                t_max_sc = t0_sc
        top_entries.append((j, m, t_max))

    return top_entries

def g_func(x):
    # normalization sigmoid (see Achterberg. thesis)
    return x / (x + 1)


def compute_infer(jmt_card, jmt_idx, local_eps=1e-6):
    # Inferability score: product of variables implied 0 (fixing jmt=0 → same-job other slots freed)
    # and implied 1 (fixing jmt=1 → same-machine same-slot and same-job other slots forbidden).
    # Higher score = more propagation expected from branching on this variable.
    (j, m, t) = jmt_idx
    score_neg = jmt_card[(j, m, t)]
    all_idx = jmt_card.keys()
    score_job = sum([jmt_card[j0, m0, t0] for (j0, m0, t0) in all_idx if j0 == j and m0 == m and t0 != t])
    score_disj = sum([jmt_card[j0, m0, t0] for (j0, m0, t0) in all_idx if j0 != j and m0 == m and t0 == t])
    score_pos = score_job + score_disj
    return max(score_neg, local_eps) * max(score_pos, local_eps)


def pscost_score(val, f_n, f_p, local_eps):
    mean0 = mean_or_one(val[0])
    mean1 = mean_or_one(val[1])
    return max(f_n * mean0, local_eps) * max(f_p * mean1, local_eps)


def weighted_pscost_score(w_val, f_n, f_p, node_id, local_eps):
    # Weighted pseudocost: observations closer in the tree (small node_dist) get higher weight.
    # w_val[d] is a list of (observed_node_id, gain) pairs for direction d.
    if w_val[0]:
        w_mean0 = 0
        weights0 = []
        for (nod, val) in w_val[0]:
            nod_weight_score = 1 + 2 * max(len(nod), len(node_id)) - node_dist(nod, node_id)
            assert (nod_weight_score >= 0)
            w_mean0 += nod_weight_score * val
            weights0.append(nod_weight_score)
    else:
        weights0 = [1]
        w_mean0 = 1

    if w_val[1]:
        w_mean1 = 0
        weights1 = []
        for (nod, val) in w_val[1]:
            nod_weight_score = 1 + 2 * max(len(nod), len(node_id)) - node_dist(nod, node_id)
            assert (nod_weight_score >= 0)
            w_mean1 += nod_weight_score * val
            weights1.append(nod_weight_score)
    else:
        weights1 = [1]
        w_mean1 = 1

    return max(f_n * w_mean0 / np.sum(weights0), local_eps) * max(f_p * w_mean1 / np.sum(weights1), local_eps)


def comp_weight_pscost_infer_score(w_d, f_val, jmt_card, node_id, topk=2, w_reli=1, w_infer=1e-3, local_eps=1e-6):
    # Score = w_reli * g(pseudocost / mean) + w_infer * g(inferability / mean), normalized via g_func.
    # Pseudocost term uses tree-distance-weighted observations; inferability measures implied fixings.
    pscost_mean = np.mean([weighted_pscost_score(w_dk, f_val[k], 1 - f_val[k], node_id, local_eps) for (k, w_dk) in w_d.items()])
    infer_mean = np.mean([compute_infer(jmt_card, k) for k in w_d])

    var_score = {}
    for k in w_d:
        rel_k = 1
        reli_score = g_func(rel_k * weighted_pscost_score(w_d[k], f_val[k], 1 - f_val[k], node_id, local_eps) / pscost_mean)
        infer_score = g_func(compute_infer(jmt_card, k) / infer_mean)
        var_score[k] = w_reli * reli_score + w_infer * infer_score

    if topk == 1:
        return max(var_score, key=var_score.get)
    return sorted(var_score, key=var_score.get, reverse=True)[:topk]


def comp_pscost_infer_score_topk(d, f_val, jmt_card, topk=10, w_reli=1, w_infer=1e-3, local_eps=1e-6):
    pscost_vals = [pscost_score(dk, f_val[k], 1 - f_val[k], local_eps) for k, dk in d.items()]
    pscost_mean = mean_or_one(pscost_vals)

    infer_vals = [compute_infer(jmt_card, k) for k in d]
    infer_mean = mean_or_one(infer_vals)

    var_score = {}
    for k in d:
        rel_k = np.sqrt(len(d[k][0]) + len(d[k][1]))
        reli_score = g_func(rel_k * pscost_score(d[k], f_val[k], 1 - f_val[k], local_eps) / pscost_mean)
        infer_score = g_func(compute_infer(jmt_card, k) / infer_mean)
        var_score[k] = w_reli * reli_score + w_infer * infer_score

    if topk == 1:
        return max(var_score, key=var_score.get)
    return sorted(var_score, key=var_score.get, reverse=True)[:topk]