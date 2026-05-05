import ast

import numpy as np
import time

from numba import njit

LOG_FILE = "branch_and_price_log.txt"

timing_store = {"branch": [0.0],
                "farkas_pricer": [0.0],
                "comp-proba": [0.0],
                "consenfolp": [0.0],
                "strongbranch": [0.0],
                "pscost_update": [0.0],
                "cheapbranch": [0.0],
                "d_strong_branch_eval": [0.0],
                "other_pricing": [0.0],
                "subgrad_algo": [0.0],
                "sep_disjc": [0.0],
                "pricer": [0.0],
                "cumsum_rc": [0.0],
                "get_duals": [0.0],
                "frac_t": [0.0],
                "branching_eventhandler": [0.0],
                "pricing_solver": [0.0],
                "shortest_path": [0.0],
                "array_prep": [0.0],
                "res_usage": [0.0],
                "pattern_cost": [0.0],
                "check_red_cost": [0.0],
                "LP_solve": [0.0],
                "subMIP": [0],
                
                "stab_utils": [0.0],
                "subgradient": [0.0],

                "FW_utils": [0.0],
                "map_Binv": [0.0],
                "SE_edgedir": [0.0],
                "SE_get_Binv": [0.0],
                "SE_map_mt": [0.0],

                "FW_lmo": [0.0],
                "SE_cache": [0.0],
                "SE_LNS": [0.0],
                "col_repr": [0.0],

                "rc_fixing": [0.0],

                "treesearch": [0.0],
                "price_check": [0.0],
                "RMP_Heur": [0.0],
                "add_cols": [0.0],
                "in_out_strbr": [0.0],
                "conn_comp": [0.0],
                "disj_cliques": [0.0],
                "ipm_duals": [0.0]}

iter_store = {
    "subgrad_algo": [0],
    "sep_disjc": [0],
    "cumsum_rc": [0],
    "pricer": [0],
    "rc_fixing": [0],
    "pricing_solver": [0],
    "shortest_path": [0],
    "subMIP": [0],
    "SE_get_Binv": [0],
    "conn_comp": [0],
    "destroy_repair": [0.0],
    "price_check": [0.0],
}


def reset_store(storing_dict):
    for k in storing_dict:
        storing_dict[k][0] = 0


def timeit_accumulate(timer_container):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            timer_container[0] += time.perf_counter() - start
            return result

        return wrapper

    return decorator


def iter_accumulate(iter_container):
    def decorator(func):
        def wrapper(*args, **kwargs):
            iter_container[0] += 1
            return func(*args, **kwargs)

        return wrapper

    return decorator


def TOU_peak(base, mid_peak, on_peak, L, H):
    base_pattern = [2 * L, L / 2, L, L / 2]
    cost_pattern = [base, mid_peak, on_peak, mid_peak]

    intervals = []
    c_p = []
    current_start = 0.0

    while current_start < H:
        for length, cost in zip(base_pattern, cost_pattern):
            if current_start >= H:
                break
            intervals.append(current_start)
            c_p.append(cost)
            current_start += length

    if not intervals or intervals[-1] < H:
        intervals.append(H)

    s_p = [int(round(t)) for t in intervals]

    C_t = np.zeros(H)
    for i, s in enumerate(s_p[:-1]):
        s1 = s_p[i + 1] + 1
        C_t[s:s1] = c_p[i]
    return C_t, c_p, s_p

def TOU_pyr(base_rate, evol_func, L_p, Cmax):
    s_p = list(range(0, Cmax, L_p))
    s_p.append(Cmax)
    N_p = len(s_p) - 1
    c_p = [base_rate for _ in range(N_p)]
    N_p2 = N_p // 2

    for i in range(1, N_p2 + 1):
        c_p[i] = evol_func(c_p[i - 1])

    for i in range(N_p2 + 1, N_p):
        c_p[i] = c_p[N_p - 1 - i] 

    C_t = np.zeros(Cmax)
    for i, s in enumerate(s_p[:-1]):
        C_t[s:s_p[i+1]] = c_p[i]  # exclusive upper bound

    return C_t, c_p, s_p

def gen_cost_func(cost_func_type, C_max, pwc=False):
    T = np.arange(1, C_max + 1)
    np.random.seed(123)

    cost_t = []

    if 'tou_pyr' in cost_func_type:
        l_p = int(np.floor(C_max/6))
        cost_t, C_p, s_p = TOU_pyr(base_rate=10, evol_func=lambda x:2*x, L_p=l_p, Cmax=C_max)

    elif 'tou1231' in cost_func_type:
        l_p = int(np.floor(C_max/6))
        cost_t, C_p, s_p = TOU_peak(base=10, mid_peak=20, on_peak=40, L=l_p, H=C_max)

    elif 'tou12' in cost_func_type:
        l_p = int(np.floor(C_max/6))
        s_p = generate_sp(l_p, C_max)
        a, b = 13, 16
        C_p = [a if i % 2 == 0 else b for i in range(len(s_p) - 1)]
        cost_t = []
        for i, x in enumerate(C_p):
            cost_t += [x] * (s_p[i + 1] - s_p[i])

    elif 'tou11' in cost_func_type:
        l_p = int(np.floor(C_max/6))
        s_p = list(range(0, C_max, l_p)) + [C_max]
        a, b = 13, 20
        C_p = [a if i % 2 == 0 else b for i in range(len(s_p) - 1)]
        cost_t = []
        for i, x in enumerate(C_p):
            cost_t += [x] * (s_p[i + 1] - s_p[i])

    elif 'rand' in cost_func_type:
        cost_t = 50 + np.random.randint(-10, 10, C_max)

    elif 'v' in cost_func_type:
        cost_t = [1 + C_max // 2 - abs(i - C_max // 2) for i in T]

    elif 'bar_v' in cost_func_type:
        cost_t = [abs(C_max // 2 - i) for i in T]

    if pwc:
        return cost_t, C_p, s_p
    return cost_t


class InstanceKPData:
    def __init__(self):
        self.W_m = []
        self.alph_dict = {}
        self.fac_dict = {}


class FacetIneq:
    def __init__(self, coeffs, rhs):
        self.coefficients = coeffs
        self.rhs = rhs


class JobSchedule:
    """
    id: job id
    compl_times: completion times (sufficient to characterize a job schedule in the JSSP)
    """

    def __init__(self, var_name=None):
        self.unique_id = -1
        self.job_id = 0
        self.compl_times = []
        self.start_times = []
        self.time_steps = []
        self.time_steps_flat = None
        self.cost = 0
        self.mt_pairs = set()
        self.curr_rc = None
        if var_name:
            self.var_name_to_schedule(var_name)

    def var_name_to_schedule(self, var_name):
        job_id, job_sched = var_name.split("-")
        self.job_id = int(job_id)
        schedule_string = job_sched.split("_")
        self.compl_times = [int(n) for n in schedule_string]

    def schedule_to_var_name(self):
        str_compl_times = [str(int(_)) for _ in self.compl_times]
        return str(self.job_id) + "-" + "_".join(str_compl_times)

    def comp_start_times(self, dur):
        return [x - dur[i] + 1 for i, x in enumerate(self.compl_times)]

    def comp_time_steps(self):
        c = self.compl_times
        s = self.start_times
        n = len(c)
        return [list(range(s[i], c[i] + 1)) for i in range(n)]


class VarSchedule:

    def __init__(self, var, schedule):
        self.var = var
        self.var_name = self.var.name
        self.schedule = schedule
        self.val_hist = []
        self.rc_hist = []
        self.age = 0

    def compute_age(self):
        count = 0
        for val in reversed(self.val_hist):
            if val == 0:
                count += 1
            else:
                break
        self.age = count
        return count

# Bitset stuff (for reference purposes only)

def idx_bit(m, t, T_size):
    return m * T_size + t

def to_bitset(pairs, T_size):
    b = 0
    for m, t in pairs:
        b |= 1 << idx_bit(m, t, T_size)
    return b

# A ← A ∪ B
def update_bit(A_bits, B_bits):            
    return A_bits | B_bits

# A ← A \ B
def difference_update_bit(A_bits, B_bits):
    return A_bits & ~B_bits

def disjoint_bits(A_bits: int, B_bits: int) -> bool:
    return (A_bits & B_bits) == 0



def preprocess_constraints(all_cons_names, n_conss):

    parsed = [None] * n_conss
    for i, name in enumerate(all_cons_names):
        parts = name.split('_')
        ctype = parts[0]
        if ctype == 'disj':
            parsed[i] = ('disj', int(parts[1]), int(parts[2]))
        elif ctype == 'res':
            parsed[i] = ('res', int(parts[-1]))
        else:  # 'part'
            parsed[i] = ('part', int(parts[1]))
    return parsed

def retrieve_job_schedule(schedule_var_name, model, job_index, n_machines, dur_vec):
    s_values_arr = np.zeros(n_machines, int)

    for (j, m), var_expr in schedule_var_name.items():
        if j == job_index:
            s_values_arr[m] = model.getSolVal(sol=None, expr=var_expr) + dur_vec[j][m]

    return s_values_arr


def compute_job_sch_cost(proc_var_name, model, job_index, cost_vec, m_powers):
    total_cost = 0

    for (j, m, p), var_expr in proc_var_name.items():
        if j == job_index:
            d_value = model.getSolVal(sol=None, expr=var_expr)
            total_cost += cost_vec[p] * m_powers[m] * d_value

    return total_cost


@timeit_accumulate(timing_store["pattern_cost"])
def compute_pattern_cost(job_dur, cost_vec, m_powers, compl_times):
    total_cost = 0
    Z_jt = []
    M = range(len(m_powers))
    T = range(len(cost_vec))
    for t in T:
        flag_exec = False
        for m in M:
            if compl_times[m] - job_dur[m] + 1 <= t <= compl_times[m]:
                Z_jt.append(1)
                total_cost += cost_vec[t] * m_powers[m]
                flag_exec = True
            if flag_exec:
                break
        if flag_exec:
            continue
        Z_jt.append(0)

    total_cost = np.round(total_cost, decimals=7)  # TODO : uniformize tolerances
    return Z_jt, total_cost


def compute_proj_Z(nT, nM, nJ, all_patterns, coeffs):
    Z = np.zeros((nT, nM, nJ), dtype=np.float64)
    coeffs = np.asarray(coeffs)

    for pid, patt in enumerate(all_patterns):
        j = patt.job_id
        w = coeffs[pid]
        Z_j = Z[:, :, j]  # 2D view to cut one index
        for m, t in patt.mt_pairs:
            Z_j[t, m] += w

    return Z


def weighted_min_score(a, b):
    num = 0.0
    den = 0

    if a:
        num += len(a) * min(a)
        den += len(a)
    else:
        den += 1

    if b:
        num += len(b) * min(b)
        den += len(b)
    else:
        den += 1

    return num / den

def f_decr(alpha):
    return max(0.0, round(alpha - 0.1, 10))


def f_incr(alpha):
    return min(round(alpha + (1.0 - alpha) * 0.1, 10), 0.9999)


@timeit_accumulate(timing_store["subgradient"])
def comp_subgradient(all_cons_parts, n_conss, all_cons_Rhs, machines, _durations, all_patterns):  # TODO: iterate over mt_pairs instead of running checks.

    subgrad_vec = np.empty(n_conss, dtype=float)

    jobs = [j for j in all_patterns if all_patterns[j] != []]

    for cons_id, cons_parts in enumerate(all_cons_parts):
        cons_type = cons_parts[0]

        if cons_type == 'part':
            subgrad_vec[cons_id] = 0
            continue

        if cons_type == 'disj':
            m, t = cons_parts[1:]
            subgrad_vec[cons_id] = all_cons_Rhs[cons_id]
            for j in jobs:
                curr_pattern = all_patterns[j]
                if curr_pattern.start_times[m] <= t <= curr_pattern.compl_times[m]:
                    subgrad_vec[cons_id] -= 1.0
            continue

        if cons_type == 'res':
            t = cons_parts[-1]
            subgrad_vec[cons_id] = all_cons_Rhs[cons_id]
            for j in jobs:
                curr_pattern = all_patterns[j]
                for m in machines:
                    if curr_pattern.start_times[m] <= t <= curr_pattern.compl_times[m]:
                        subgrad_vec[cons_id] -= 1.0
            continue


    subgrad_norm = np.linalg.norm(subgrad_vec)
    return subgrad_vec, subgrad_norm


def comp_in_sep(cons_parts, out_duals, in_duals, n_conss, row_gen=False):
    out_part, out_disj, out_res = out_duals
    in_part, in_disj, in_res = in_duals

    in_sep_dir_vec = np.empty(n_conss, dtype=float)

    for i, cons_n in enumerate(cons_parts):
        cons_type = cons_n[0]

        # partitioning constraints
        if cons_type == 'part':
            j = cons_n[1]
            in_sep_dir_vec[i] = out_part[j] - in_part[j]  # REV: to skip or not
            continue 

        # disjunction constraints
        if cons_type == 'disj':
            m, t = cons_n[1:]
            in_sep_dir_vec[i] = out_disj[(m, t)] - in_disj[(m, t)]
            continue 

        # resource constraints
        if cons_type == 'res':
            t = cons_n[-1]
            in_sep_dir_vec[i] = out_res[t][0] - in_res[t][0]  # TODO: KP!
            continue 

    in_sep_dir_norm = np.linalg.norm(in_sep_dir_vec)
    return in_sep_dir_vec, in_sep_dir_norm


@timeit_accumulate(timing_store["get_duals"])
def get_duals(constraint_set, rmp, farkas_bool):
    constr_duals = {}
    for (cons_id, cons) in constraint_set.items():
        cons = rmp.getTransformedCons(cons)
        if farkas_bool:
            constr_duals[cons_id] = rmp.getDualfarkasLinear(cons)
        else:
            constr_duals[cons_id] = rmp.getDualsolLinear(cons)

    return constr_duals

def intersects(a, b, c, d):
    return max(a, c) <= min(b, d)


def intrvl_intrsct_same_mach1(list_A, list_B):
    """
    intersecting processing intervals of tasks on different machines
    """
    indexed_A = sorted(enumerate(list_A), key=lambda x: x[1][0])  # sort by start time
    indexed_B = sorted(enumerate(list_B), key=lambda x: x[1][0])

    i, j = 0, 0  # pointers

    while i < len(indexed_A) and j < len(indexed_B):
        idx_a, (a_start, a_end) = indexed_A[i]
        idx_b, (b_start, b_end) = indexed_B[j]

        a = min(a_end, b_end)
        b = max(a_start, b_start)
        t0 = np.round((a + b) / 2)
        if idx_a == idx_b and b <= a:
            return idx_a, idx_b, t0  # intersection found

        if a_end < b_end:
            i += 1  # mv A forward
        else:
            j += 1  # mv B forward

    return False  # no valid intersections


def node_dist(a: str, b: str) -> int:
    lcp = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            lcp += 1
        else:
            break
    return (len(a) - lcp) + (len(b) - lcp)


def min_consecutive_sum(A, n):
    min_sum = float('inf')
    min_index = 0

    for i in range(len(A) - n + 1):
        current_sum = sum(A[i:i + n])
        if current_sum < min_sum:
            min_sum = current_sum
            min_index = i

    return A[min_index:min_index + n], min_index, min_sum


def generate_sp(len_p, Cmax):
    a = 0
    count = 1
    s_p = [0]
    while a + len_p * (count % 2 + 1) < Cmax:
        a += len_p * (count % 2 + 1)
        count += 1
        s_p.append(a)
    s_p.append(Cmax)
    return s_p


def mean_or_one(arr):
    return np.mean(arr) if arr else 1


def keep_relevant_stats(mod, cg_bool):
    mod.setParam("table/status/active", 0)
    mod.setParam("table/origprob/active", 0)
    mod.setParam("table/presolvedprob/active", 0)
    mod.setParam("table/presolver/active", 0)
    mod.setParam("table/constraint/active", 0)
    mod.setParam("table/branchrules/active", 0)
    mod.setParam("table/pricer/active", 0)
    mod.setParam("table/constiming/active", 0)
    mod.setParam("table/timing/active", 0)
    mod.setParam("table/propagator/active", 0)
    mod.setParam("table/conflict/active", 0)
    mod.setParam("table/separator/active", 0)
    mod.setParam("table/symmetry/active", 0)
    mod.setParam("table/cutsel/active", 0)
    mod.setParam("table/scheduler/active", 0)
    mod.setParam("table/root/active", 0)
    mod.setParam("table/relaxator/active", 0)
    if cg_bool:
        mod.setParam("table/heuristics/active", 0)
        mod.setParam("table/tree/active", 0)
        mod.setParam("table/solution/active", 0)


def parse_scip_log(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    output = []

    # 1. Primal Heuristics
    in_heuristics = False
    output.append("[Primal Heuristics]")
    for line in lines:
        if line.strip().startswith("Primal Heuristics"):
            in_heuristics = True
            continue
        if in_heuristics:
            if line.strip().startswith("LP") or line.strip() == "":
                if not line.strip().startswith("LP solutions"):
                    break
            parts = line.strip().split()
            if ':' in parts[0]:
                parts.insert(1, ':')
            if len(parts) < 6:
                continue
            if (parts[1] == 'solutions' or parts[1] == 'branching') and parts[0] != 'other':
                name = parts[0]
                found = int(parts[6])
                best = int(parts[7])
                if found > 0:
                    if best == 0:
                        output.append(f"{name:<18} Time: {time:7.3f} Found: {found}")
                    else:
                        output.append(f"{name:<18} Time: {time:7.3f} Found: {found} ({'*' * best})")
            elif parts[0] != 'other':
                time = float(parts[2])
                calls = int(parts[4])
                found = int(parts[5])
                best = int(parts[6])
                if time > 0.0 or calls > 0:
                    name = parts[0]
                    if best == 0:
                        output.append(f"{name:<18} Time: {time:7.3f} \tCalls: {calls:<8} Found: {found:<4}")
                    else:
                        output.append(f"{name:<18} Time: {time:7.3f} \tCalls: {calls:<8} Found: {found:<4} ({'*' * best})")

    output.append("")

    # 2. LP solving
    output.append("[LP Solving]")
    record = False
    for line in lines:
        if line.strip().startswith("LP"):
            record = True
            continue
        if record:
            if ("primal LP" in line or "dual LP" in line or "diving/probing LP" in line) and "lex" not in line:
                parts = line.strip().split()
                if 'diving' in line:
                    parts.append("")
                    parts.append("")
                name = " ".join(parts[:2])
                time = float(parts[-7])
                calls = parts[-6]
                iters = parts[-5]
                output.append(f"{name:<20} Time: {time:.2f} \tCalls: {calls:<8} Iters: {iters:<8}")
            elif line.strip() == "":
                break

    output.append("")

    # 3. B&B Tree
    output.append("[B&B Tree]")
    record = False
    for line in lines:
        if line.strip().startswith("B&B Tree"):
            record = True
            continue
        if record:
            if any(x in line for x in ["nodes", "nodes left", "max depth", "backtracks"]):
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    output.append(f"{key:<20}: {val}")
            elif line.strip().startswith("Solution"):
                break

    output.append("")

    # 4. Solution
    output.append("[Solution]")
    record = False
    incumb = False
    incumb_output = False
    for line in lines:
        if line.strip().startswith("Solution") and 'found' not in line.strip():
            record = True
            continue
        if record:
            if line.strip() == "":
                break
            if "Solutions found" in line:
                parts = line.strip().split()
                if float(parts[3]) > 0:
                    output.append(" ".join(parts[:-2]))
                    incumb = True
            if incumb:
                if "First Solution" in line or "Primal" in line:
                    parts = line.strip().split()
                    val = str(int(float(parts[3])))
                    heur = parts[-1][1:-2]
                    tab_sp = "\t \t "
                    output.append(" ".join(parts[:3]) + " " + val + f"{tab_sp} (" + " ".join(parts[8:14]) + " " + heur + ")")
            if 'Dual Bound' in line:
                parts = line.strip().split()
                val = str(int(float(parts[3])))
                output.append(" ".join(parts[:3]) + " " + val)
            if "Gap" in line:
                output.append(" ".join(line.strip().split()))

    output.append("\n________________________________________________________________________________________________\n\n")
    return "\n".join(output)


@njit
def compute_rc_term_njit(powers, cost_vec):
    n_machines = powers.shape[0]
    C_max = cost_vec.shape[0]

    rc_term = np.zeros((n_machines, C_max))

    for m in range(n_machines):
        p = powers[m]
        for tau in range(C_max):
            rc_term[m, tau] = p * cost_vec[tau]

    return rc_term

# Param settings

def lpbased_heur_tune(root_bool, mod, max_iters):
    if not root_bool:
        return
    curr_lp_iters = mod.getNLPIterations()
    heurs = ['adaptivediving', 'indicatordiving', 'farkasdiving', 'feaspump', 
            'conflictdiving','coefdiving', 'pscostdiving', 'fracdiving', 
            'veclendiving', 'distributiondiving','intdiving', 'actconsdiving', 
            'objpscostdiving', 'rootsoldiving', 'linesearchdiving', 'guideddiving']
    getparam = mod.getParam
    setparam = mod.setParam
    for heur in heurs:
        heur_prio = "heuristics/" + heur + "/maxlpiterquot"
        curr_ratio = getparam(heur_prio)
        new_ratio = min(1/curr_lp_iters * curr_ratio*max_iters, curr_ratio)         # TODO: unnecessary when curr_lp_iters < max_iters; kept for clarity
        setparam(heur_prio, new_ratio)


def dive_params(mod):
    mod.setParam('lp/solutionpolishing', 0)
    mod.setParam("lp/initalgorithm", "d")
    mod.setParam('lp/pricing', 'd')
    mod.setParam('lp/fastmip', 0)
    mod.setParam('lp/checkstability', False)
    mod.setParam('lp/checkdualfeas', False)
    mod.setParam('lp/checkprimfeas', False)
    # mod.setParam('lp/iterlim', 5000)            # TODO: argument

def revert_dive_params(mod):
    mod.setParam('lp/solutionpolishing', 3)
    mod.setParam("lp/initalgorithm", "p")
    mod.setParam('lp/pricing', 's')
    mod.setParam('lp/fastmip', 1)
    mod.setParam('lp/checkstability', True)
    mod.setParam('lp/checkdualfeas', True)
    mod.setParam('lp/checkprimfeas', True)
    # mod.setParam('lp/iterlim', -1)