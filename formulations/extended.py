import time, sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from collections import defaultdict

from pyscipopt import Model, SCIP_PARAMSETTING, quicksum, SCIP_HEURTIMING

from branching_eventhdlr import OrigVarBranchingEventhdlr
from origvar_branching import OrigVar
from pricer_directionalsmoothing import ShortestPathPricer

from shortest_path import compute_opt_schedule

from primal_heur.rmp_tree_last import RMPTreeSearchHeuristic

from utils import iter_store, preprocess_constraints, keep_relevant_stats, parse_scip_log, compute_rc_term_njit
import numpy as np

GAP_LIMIT = 1e-5
TIME_LIMIT_SECONDS = 3600
SOL_FEAS_TOL = 1e-5

Reduc_Bool = False
SG_Bool = False       # subgradient in strong branching
Pwc_Bool = False      # deprecated
Wentges_Bool = True   # dual stabilization smoothing
Partial_Bool = False  # deprecated
Cg_Bool = False       # solve root node only
Heur_Bool = True      # RMP tree search heuristic
Rounding = None       # None | "UNFIX" | "RENS"
Comp_Bool = False     # complementary pricing (deprecated)
Str_Branch = True     # strong branching
Early_Branch_Int = True   # integer gap early branching
Early_Branch_Tail = False # tailoff early branching
Early_Branch_Lars = False # Lars' rule early branching

dual_evol = []
lag_bound = []
best_lag_bound = []
branch_evol = []
lower_bound = []
req_iters = [0]
lp_evol = []


def _configure_model(model):
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam("conflict/enable", 0)
    # suppress heuristics with low gain/time ratio
    model.setParam('heuristics/alns/freq', -1)
    model.setParam('heuristics/rens/freq', -1)
    model.setParam('heuristics/intdiving/freq', -1)
    model.setParam('heuristics/pscostdiving/freq', -1)
    model.setParam('heuristics/farkasdiving/freq', -1)
    # RMP LP algorithm
    model.setParam("lp/initalgorithm", 'p')
    model.setParam("lp/resolvealgorithm", 'd')
    # stability params for dense constraint matrices # TODO: to test
    model.setParam('lp/pricing', 's')
    model.setParam('lp/minmarkowitz', 0.8)      
    model.setParam('lp/scaling', 0)
    model.setParam('lp/rowrepswitch', 0)
    # limits and display
    model.setParam('limits/gap', GAP_LIMIT)     # TODO: argument
    model.setParam("display/freq", 10)
    model.setParam('limits/time', TIME_LIMIT_SECONDS)
    model.setParam('randomization/randomseedshift', 0)


def _add_constraints(model, jobs, machines, C_max, KP_ineq, X, dur, seq, W_m, Cost_t, M):
    partitioning_constraints = {}
    DOB_geq = {}
    disjunction_constraints = defaultdict(dict)
    resource_constraints = defaultdict(dict)
    all_rhss = []

    for j in jobs:
        DOB_geq[j] = model.addVar(name=f"DOI_geq_{j}", vtype="C", lb=0, obj=-compute_opt_schedule(M, C_max, dur[j], seq[j], W_m, Cost_t))  # NB: this enforces positivity of part_duals, no need to relax part cons
        partitioning_constraints[j] = model.addCons(quicksum(X) - DOB_geq[j] == 1, separate=False, name=f"part_{j}", modifiable=True)  # careful about matrix_rhs if >=
        all_rhss.append(1)

    for t in range(C_max):
        for m in machines:
            disjunction_constraints[m, t] = model.addCons(quicksum(X) <= 1, separate=False, name=f"disj_{m}_{t}", modifiable=True)
            all_rhss.append(1)
        for (id, kp_ineq) in enumerate(KP_ineq):
            rhs = kp_ineq.rhs
            resource_constraints[t][id] = model.addCons(quicksum(X) <= rhs, separate=False, name=f"res_{1}_{t}", modifiable=True)
            all_rhss.append(rhs)

    model.setMinimize()
    return partitioning_constraints, DOB_geq, disjunction_constraints, resource_constraints, all_rhss


def extended_kpjssp(sg_algo_bool, wentges_bool, cg_bool, heur_bool, comp_pricing_bool, early_branch_int, early_branch_tailoff, early_branch_lars, Cost_t,
                    rounding_mode=None, log_file_name=None, pb_data=None, verbose=False, reduc_bool=False, strong_br_bool=False, dom_propag_bool=False,
                    reli_budget=5, n_pricing_iters=2):
    """
    Extended KPJSSP model, returns the solved model, pricer and branching rule
    """
    model = Model("Extended kpjssp")
    if log_file_name:
        model.setLogfile(log_file_name)

    _configure_model(model)
    dual_feas_tol = model.getParam("numerics/dualfeastol")

    J, M, W_m, dur, seq, Cost_t, KP_ineq = pb_data
    C_max = len(Cost_t)

    jobs = range(J)
    machines = range(M)
    powers = np.array(W_m)
    duration = dur
    job_sequence = seq
    Cost_t = np.array(Cost_t)

    X = {}
    partitioning_constraints, _, disjunction_constraints, resource_constraints, all_rhss = \
        _add_constraints(model, jobs, machines, C_max, KP_ineq, X, dur, seq, W_m, Cost_t, M)

    all_vars_jobs = defaultdict(list)
    all_vars_jmt = defaultdict(list)
    all_schedules_jobs = defaultdict(list)
    all_schedules_jmt = defaultdict(list)
    all_varschedules = defaultdict(list)
    pseudocost_track = defaultdict(dict)
    pseudocost_comp = defaultdict(lambda: {0: [], 1: []})
    pseudocost_op_comp = defaultdict(lambda: {0: [], 1: []})
    pseudocost_time_comp = defaultdict(lambda: {0: [], 1: []})

    forbidden_arcs_prob = set()
    forced_arcs_prob = set()

    early_branched = {}

    all_constraints = [partitioning_constraints, disjunction_constraints, resource_constraints]
    added_probing_vars = []

    all_cons = model.getConss(transformed=True)
    all_cons_names = [cons.name for cons in all_cons]
    n_conss = len(all_cons_names)
    conss_parts = preprocess_constraints(all_cons_names, n_conss)
    total_added_patterns = 0

    if not cg_bool:

        branching_rule = OrigVar(jobs=jobs, machines=machines, powers=powers, duration=duration, job_sequence=job_sequence, Cost_t=Cost_t, KP=KP_ineq,
                                 all_vars_jmt=all_vars_jmt, all_vars_jobs=all_vars_jobs, all_varschedules=all_varschedules, added_probing_vars=added_probing_vars, all_schedules_jobs=all_schedules_jobs,
                                 forbidden_arcs_prob=forbidden_arcs_prob, forced_arcs_prob=forced_arcs_prob,
                                 pseudocost_track=pseudocost_track, pseudocost_comp=pseudocost_comp, pseudocost_op_comp=pseudocost_op_comp, pseudocost_time_comp=pseudocost_time_comp, 
                                 reli_budget=reli_budget, n_pricing_iters=n_pricing_iters, strong_br_bool=strong_br_bool, dom_propag_bool=dom_propag_bool,
                                 verbose_bool=verbose)

        model.includeBranchrule(branching_rule, "Original Variable Branching", "Branching rule based on compact formulation variables",
                                priority=1000000,
                                maxdepth=-1,
                                maxbounddist=1.0)

        priced_sols = []  # Holder for priced solutions in heuristic call
        schedule_id_to_var = {i: {} for i in range(J)}

        pricer = ShortestPathPricer.from_legacy_args(
                                    jobs, machines, powers, KP_ineq, duration, job_sequence, Cost_t, all_constraints,
                                    all_rhss=all_rhss, dualfeastol_val=dual_feas_tol,
                                    total_added_patterns=total_added_patterns,
                                    priced_sols=priced_sols, schedule_id_to_var=schedule_id_to_var,
                                    arcs_to_forbid=branching_rule.forbidden_arcs,
                                    arcs_to_force=branching_rule.forced_arcs,
                                    arcs_to_forbid_prob=branching_rule.forbidden_arcs_prob,
                                    arcs_to_force_prob=branching_rule.forced_arcs_prob,
                                    all_vars_jobs=all_vars_jobs, all_schedules_jobs=all_schedules_jobs, all_vars_jmt=all_vars_jmt, all_schedules_jmt=all_schedules_jmt, all_varschedules=all_varschedules,
                                    added_probing_vars=added_probing_vars, reli_strongbr=branching_rule.reli_strongbr,
                                    early_branched=early_branched,
                                    dual_evol=dual_evol, best_lag_bound=best_lag_bound, lag_bound=lag_bound, branch_evol=branch_evol, req_iters=req_iters, lower_bound=lower_bound, lp_evol=lp_evol,
                                    early_branch_int=early_branch_int, early_branch_tailoff=early_branch_tailoff, early_branch_lars=early_branch_lars, cg_bool=cg_bool, wentges_bool=wentges_bool, comp_pricing=comp_pricing_bool, sg_algo_bool=sg_algo_bool,
                                    ext_rmp=None, verbose=verbose,
                                    all_conss=all_cons, all_cons_names=all_cons_names, all_cons_parts=conss_parts, n_conss=n_conss,
                                    reduc_bool=reduc_bool, solve_begin_time=time.perf_counter())

        branch_eventhdlr = OrigVarBranchingEventhdlr(all_varschedules=all_varschedules, all_patterns_jmt=all_vars_jmt,
                                              forbidden_arcs=branching_rule.forbidden_arcs, forced_arcs=branching_rule.forced_arcs,
                                              dur=dur)       
        model.includeEventhdlr(branch_eventhdlr, "Original Variable Branching Event Handler", "")

        if heur_bool:

            rc_term = compute_rc_term_njit(W_m, Cost_t)

            RMP_tree_search_heur = RMPTreeSearchHeuristic(jobs=jobs, machines=machines, duration=duration, KP_ineq=KP_ineq, C_max=C_max, sequence=seq, Cost_t=Cost_t, powers_m=W_m, rc_term=rc_term,  # TODO: account for KP
                                                          all_constraints=all_constraints,
                                                          all_varschedules=all_varschedules, all_schedules_jmt=all_schedules_jmt, all_vars_jmt=all_vars_jmt, all_vars_jobs=all_vars_jobs, all_schedules_jobs=all_schedules_jobs,
                                                          price_i=pricer.i, priced_sols=priced_sols, schedule_id_to_var=schedule_id_to_var,
                                                          forbidden_arcs=branch_eventhdlr.forbidden_arcs, forced_arcs=branch_eventhdlr.forced_arcs,
                                                          total_added_patterns=pricer.total_added_patterns,
                                                          rounding_mode=rounding_mode, verbose=verbose)
            model.includeHeur(RMP_tree_search_heur, "RMPTreeSearch", "RMPTreeSearch", "Y",
                              timingmask=SCIP_HEURTIMING.DURINGLPLOOP, usessubscip=False, freq=1, maxdepth=-1, priority=-1000001)   # priority just below feaspump and above diving/submip heuristics

    else:
        pricer = ShortestPathPricer.from_legacy_args(  # TODO: scuffed, shorten later
                                    jobs, machines, powers, KP_ineq, duration, job_sequence, Cost_t, all_constraints,
                                    all_rhss=all_rhss, dualfeastol_val=dual_feas_tol,
                                    total_added_patterns=total_added_patterns,
                                    priced_sols=None, schedule_id_to_var=None,
                                    arcs_to_forbid=None,
                                    arcs_to_force=None,
                                    arcs_to_forbid_prob=None,
                                    arcs_to_force_prob=None,
                                    all_vars_jobs=all_vars_jobs, all_schedules_jobs=all_schedules_jobs, all_vars_jmt=all_vars_jmt, all_schedules_jmt=all_schedules_jmt, all_varschedules=all_varschedules,
                                    added_probing_vars=None, reli_strongbr=None,
                                    early_branched=early_branched,
                                    dual_evol=dual_evol, best_lag_bound=best_lag_bound, lag_bound=lag_bound, branch_evol=branch_evol, req_iters=req_iters, lower_bound=lower_bound, lp_evol=lp_evol,
                                    early_branch_int=early_branch_int, early_branch_tailoff=early_branch_tailoff, early_branch_lars=early_branch_lars, cg_bool=cg_bool, wentges_bool=wentges_bool, comp_pricing=comp_pricing_bool, sg_algo_bool=sg_algo_bool,
                                    ext_rmp=None, verbose=verbose,
                                    all_conss=all_cons, all_cons_names=all_cons_names, all_cons_parts=conss_parts, n_conss=n_conss,
                                    reduc_bool=reduc_bool, solve_begin_time=time.perf_counter())

    model.includePricer(pricer, "ShortestPathPricer", "Pricer for Shortest Path in DAG Problem")

    model.setObjIntegral()
    start = time.perf_counter()
    model.optimize()
    end = time.perf_counter()

    if cg_bool:
        return model, pricer, None, start, end
    return model, pricer, branching_rule, start, end


def _format_stats_lines(tot_time, timing_store, avg_pricing_solver, avg_add_cols,
                        pricer_time, strbr_time, br_utils, heur_time,
                        treesearch_time, remaining_heur_time, pricer,
                        mod, all_patts, cg_iters, Cg_Bool, sol_status):
    lines = [
        '\n________________________________________________',
        f"{' time Tot':<9} {tot_time:>9.3f}",
        f"{'      Prc':<9} {timing_store['pricer'][0]:>9.3f}",
        f"{'      *SP':<9} {timing_store['pricing_solver'][0]:>9.3f}         (avg) {avg_pricing_solver:>8.5f}",
        f"{'     *alg':<9} {timing_store['shortest_path'][0]:>9.3f}         (avg) {timing_store['shortest_path'][0] / iter_store['shortest_path'][0]:>8.5f}",
        f"{'   smooth':<9} {timing_store['stab_utils'][0]:>9.3f}",
        f"{'    proba':<9} {timing_store['comp-proba'][0]:>9.3f}",
        f"{'     Frks':<9} {timing_store['farkas_pricer'][0]:>9.3f}",
        f"{'     root':<9} {pricer.root_cg_time:>9.3f}",
        f"{'       Br':<9} {timing_store['branch'][0]:>9.3f}",
        f"{'     *str':<9} {strbr_time:>9.3f}",
        f"{'  *pscost':<9} {timing_store['pscost_update'][0]:>9.3f}",
        f"{'   *utils':<9} {br_utils:>9.3f}",
        f"{'        H':<9} {heur_time:>9.3f}",
        f"{'   rc_fix':<9} {timing_store['rc_fixing'][0]:>9.3f}         (nr) {iter_store['rc_fixing'][0]:>8.0f}",
        f"{'     Tree':<9} {treesearch_time:>9.3f}",
        f"{'      D&R':<9} {remaining_heur_time:>9.3f}",
        '',
        f"{'  Pr code':<9} {pricer_time:>9.3f}",
        f"{'     +Col':<9} {timing_store['add_cols'][0]:>9.3f}         (avg) {avg_add_cols:>8.5f}",
        f"{'       ΣΣ':<9} {timing_store['cumsum_rc'][0]:>9.3f}",
        '',
        f"{'   #Nodes':<9} {mod.getNNodes():>9.0f}",
        f"{'    #Iter':<9} {pricer.req_iters[-1]:>9.0f}",
    ]
    if not Cg_Bool:
        lines += [
            f"{'     root':<9} {cg_iters[0]:>9.0f}",
            f"{'     mean':<9} {int(np.mean(cg_iters)):>9.0f}",
            f"{'      med':<9} {int(np.median(cg_iters)):>9.0f}",
        ]
    lines += [
        '',
        f"{'   #Vars':<9} {len(all_patts):>9.0f}",
        f"{'   *root':<9} {pricer.root_cg_vars:>9.0f}",
        f"{'     Obj':<9} {np.round(mod.getObjVal(), decimals=1):>9.0f}",
    ]
    if sol_status not in ('optimal', 'gaplimit'):
        lines.append(f"{'     Gap':<9} {mod.getGap():>9.3f}")
    lines.append('\n________________________________________________________________________________________________')
    return lines


def write_stats(inst_char, pricer, start, end, timing_store, mod, J, log_file, output_dir, print_mode=False):
    sol_status = mod.getStatus()

    tot_time = np.round(end - start, decimals=3)
    for t_key in timing_store:
        timing_store[t_key][0] = float(np.round(timing_store[t_key][0] + 0.001, decimals=3))

    all_vars = mod.getVars(transformed=True)
    all_patts = [x for x in all_vars if 'res' not in x.name and 'DOI' not in x.name and 'disj' not in x.name]
    all_patts_val = [mod.getVal(x) for x in all_patts]
    job_occ = {j: 0 for j in range(J)}
    for p in all_patts:
        p_idx = int(p.name[0])
        job_occ[p_idx] += 1

    np.set_printoptions(suppress=True, precision=3)
    cg_iters = np.diff(pricer.req_iters)

    avg_pricing_solver = timing_store['pricing_solver'][0] / (J * iter_store['pricing_solver'][0])
    avg_add_cols = timing_store["add_cols"][0] / len(all_vars)
    pricer_time = timing_store["pricer"][0] - timing_store["pricing_solver"][0] - timing_store["add_cols"][0]
    heur_time = timing_store["RMP_Heur"][0]
    treesearch_time = timing_store["treesearch"][0]
    remaining_heur_time = heur_time - treesearch_time
    Cg_Bool = pricer.cg_bool

    br_time = timing_store["branch"][0]
    strbr_time = timing_store["strongbranch"][0]
    br_utils = br_time - strbr_time + timing_store['branching_eventhandler'][0]

    keep_relevant_stats(mod, Cg_Bool)
    tmp_path = "temp_stats.txt"
    mod.writeStatistics(tmp_path)
    with open(tmp_path, "r") as tmpfile:
        tmp_content = tmpfile.read()
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(tmp_content)
        f.write("\n")
    os.remove(tmp_path)

    summ_stats = parse_scip_log(log_file)

    sol_vec = [x for x in all_patts if abs(mod.getVal(x) - 1) <= SOL_FEAS_TOL]
    sol_cost = sum(x.getObj() for x in sol_vec)
    sol = [x.name for x in sol_vec]

    stats_lines = _format_stats_lines(
        tot_time, timing_store, avg_pricing_solver, avg_add_cols,
        pricer_time, strbr_time, br_utils, heur_time,
        treesearch_time, remaining_heur_time, pricer,
        mod, all_patts, cg_iters, Cg_Bool, sol_status,
    )

    summ_log_file = os.path.join(output_dir, "summ_" + os.path.basename(log_file))
    if not print_mode:
        with open(summ_log_file, "a") as f:
            print(f"\n{summ_stats}\n", file=f)
            f.write(f"(dim, pow, rhs, lamb, c_t)  {inst_char}\n")
            if Cg_Bool:
                f.write(f"(CG)      Smoothing: {Wentges_Bool}, Complementary Pricing: {Comp_Bool}\n")
            else:
                f.write(f"(B&P)     Early Branching: {Early_Branch_Int}, Smoothing: {Wentges_Bool}, Complementary Pricing: {Comp_Bool}, RM Heuristic: {Heur_Bool}, Subgradient: {SG_Bool}\n")
            f.write(f"Solution status: {sol_status}\n")
            if sol_status in ('optimal', 'gaplimit'):
                f.write(f"Solution value {sol_cost}\n")
                f.write(f"Solution \t {sol}\n")
                if not Cg_Bool:
                    for j in range(J):
                        ll = [x.name for x in pricer.all_vars_jobs[j]]
                        sol_job = [x for x in pricer.all_vars_jobs[j] if abs(mod.getVal(x) - 1) <= SOL_FEAS_TOL][0].name
                        f.write(f"Job {j} \t idx {ll.index(sol_job)}  \t (tot {len(ll)})\n")
                    f.write(f" \t  \t  \t (tot {len(all_patts)})\n")
                    f.write('\n')
            for line in stats_lines:
                f.write(line + '\n')
    else:
        print(f"\n{summ_stats}\n")
        print(timing_store)
        os.remove(log_file)
        if Cg_Bool:
            print(f"(CG)      Smoothing: {Wentges_Bool}, Complementary Pricing: {Comp_Bool}")
        else:
            print(f"(B&P)     Early Branching: {Early_Branch_Int}, Smoothing: {Wentges_Bool}, Complementary Pricing: {Comp_Bool}, RM Heuristic: {Heur_Bool}, Subgradient: {SG_Bool}")
        print(f"Solution status: {sol_status}")
        if sol_status in ('optimal', 'gaplimit'):
            print(f"Solution value {sol_cost}")
            print("Solution \t", sol)
            if not Cg_Bool:
                for j in range(J):
                    ll = [x.name for x in pricer.all_vars_jobs[j]]
                    sol_job = [x for x in pricer.all_vars_jobs[j] if abs(mod.getVal(x) - 1) <= SOL_FEAS_TOL][0].name
                    print(f"Job {j} \t idx {ll.index(sol_job)}  \t (tot {len(ll)})")
                print(f" \t  \t  \t (tot {len(all_patts)})")
                print()
        for line in stats_lines:
            print(line)