import numpy as np
import math
from copy import copy
from itertools import chain
from collections import defaultdict
from numba import njit, prange # type: ignore

from utils import JobSchedule, timing_store, timeit_accumulate, iter_store, iter_accumulate

from ._shortest_path_c import compute_shortest_path

def retrieve_min_reduced_costs(F, part_duals, decisions, job_idx, powers, sequence, duration, cost_t, dual_feas_tol):
    min_red_cost_job = F[-1][-1] - part_duals[job_idx]
    path_job = recover_path(decisions)
    job_sch = JobSchedule()
    job_sch.job_id = job_idx
    job_sch.compl_times = [0 for _ in powers]  # i.e. range(n_machines)
    for (idx, op) in enumerate(sequence[job_idx]):
        job_sch.compl_times[op - 1] = path_job[idx][1]
    job_sch.start_times = job_sch.comp_start_times(duration[job_idx])
    job_sch.time_steps = job_sch.comp_time_steps()
    flat_steps = set().union(*job_sch.time_steps)
    job_sch.time_steps_flat = flat_steps
    job_sch.time_steps_flat_list = list(flat_steps)
    
    pairs = set()
    bits = 0 
    T_SIZE = len(cost_t)
    for m, t_steps in enumerate(job_sch.time_steps):
        base = m * T_SIZE
        add = pairs.add
        for t in t_steps:
            add((m, t))
            bits |= 1 << (base + t)
    job_sch.mt_pairs = pairs
    job_sch.mt_pairs_bitset = bits
    
    job_sch.cost = np.sum([powers[m]*cost_t[t] for (m, t) in job_sch.mt_pairs])
    return job_sch, min_red_cost_job



@timeit_accumulate(timing_store["check_red_cost"])
def compute_pattern_red_cost(pattern_pairs, rc_term, part_dual_term):
    """
    Computes pattern reduced cost wrt. to input dual vectors
    """
    idx = np.array(list(pattern_pairs), dtype=int)
    return rc_term[idx[:, 0], idx[:, 1]].sum() - part_dual_term

@timeit_accumulate(timing_store["array_prep"])
def prepare_arrays(cost_vec, farkas_bool, KP_ineq, res_duals, disj_duals, n_machines, n_KP, C_max):
    cost_vec = np.asarray(cost_vec) * (1 - farkas_bool)
    KP_ineq_arr = np.array([[KP_ineq[i].coefficients[m] for m in range(n_machines)] for i in range(n_KP)])

    res_duals_arr = np.zeros((C_max, n_KP))
    for tau in range(C_max):
        for i in range(n_KP):
            res_duals_arr[tau, i] = res_duals[tau][i]

    disj_duals_arr = np.zeros((n_machines, C_max))
    for (m, tau), val in disj_duals.items():
        disj_duals_arr[m, tau] = val

    return cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr

@iter_accumulate(iter_store["pricing_solver"])
@timeit_accumulate(timing_store["pricing_solver"])
def pricing_solver(jobs, powers, KP_ineq, duration, sequence, cost_t, dual_sols, 
                   forbidden_arcs, forced_arcs, 
                   farkas_bool, root_bool,
                   rc_term=None,
                   offset_cost=None, shift_range=None, partial_bool=False, additional_paths=False, dual_feas_tol=0):
    """
    Solve a shortest path in a DAG for the pricing problem, accounting for branching constraints
    Parameters:

    Returns: the minimum reduced cost and the resulting pattern

    TODO
        - params + types
    """

    part_duals, disj_duals, res_duals = dual_sols
    n_machines = len(powers)
    C_max = len(cost_t)
    n_KP = len(KP_ineq)

    all_schedules = {i: [] for i in jobs}
    all_min_red_costs = {i: 0 for i in jobs}
    if additional_paths:
        all_schedules_add = {i: [] for i in jobs}
        all_min_red_costs_add = {i: 0 for i in jobs}

    # TODO: move to attribute creation step
    d0 = defaultdict(list, {i: [] for i in jobs})
    d = defaultdict(list, {i: [] for i in jobs})
    d1 = defaultdict(list, {i: [] for i in jobs})

    for arc in forbidden_arcs:
        d[arc[0]].append(arc)
    for arc in forced_arcs:
        d1[arc[0]].append(arc)

    if partial_bool and not root_bool:
        # pricing_counter = 0
        jobs = np.random.permutation(jobs)

    if offset_cost is not None:
        cost_vec = offset_cost
    else:
        cost_vec = cost_t

    if rc_term is None:
        cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr = prepare_arrays(cost_vec, farkas_bool, KP_ineq, res_duals, disj_duals, n_machines, n_KP, C_max)
        rc_term = compute_rc_term_njit(powers, cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr)

    for job_idx in jobs:
        job_arcs0 = d0[job_idx]
        job_arcs = d[job_idx]
        djob_arcs = d1[job_idx]

        F, decisions = compute_shortest_path(
            n_machines=n_machines,
            C_max=C_max,
            dur=duration[job_idx],
            seq=sequence[job_idx],
            rc_term=rc_term,
            branch_operations=job_arcs,
            branch_operations1=djob_arcs,
            branch_operations0=job_arcs0,
        )

        if F[-1][-1] < float('inf'):
            all_schedules[job_idx], all_min_red_costs[job_idx] = retrieve_min_reduced_costs(F, part_duals, decisions, job_idx, powers, sequence, duration, cost_t, dual_feas_tol=dual_feas_tol)

            cross_check = np.sum([rc_term[m,t] for m,t in all_schedules[job_idx].mt_pairs]) - part_duals[job_idx]
            assert (abs(cross_check - all_min_red_costs[job_idx]) <= 1e-7)  # TODO : to remove eventually

            # check if pattern is indeed feasible # TODO : to remove eventually
            patt_check = all_schedules[job_idx]
            job_check = patt_check.job_id
            dur_check = duration[job_check]
            seq_check = sequence[job_check]
            assert (math.prod([dur_check[m] == len(patt_check.time_steps[m]) for m in range(n_machines)]))
            assert (list(np.argsort(patt_check.compl_times) + 1) == seq_check)

            for (_, m, t) in job_arcs:
                assert (t < patt_check.time_steps[m][0] or t > patt_check.time_steps[m][-1])
            for (_, m, t) in djob_arcs:
                assert (patt_check.time_steps[m][0] <= t <= patt_check.time_steps[m][-1])

            # pricing_counter += 1
            # if pricing_counter == pricing_rounds:  # if not partial_bool, pricing_counter = - n_jobs (+ potentially n_jobs solved) < low = 1  -> not partial
            #     return all_min_red_costs, all_schedules
            if additional_paths:
                all_schedules_add[job_idx], all_min_red_costs_add[job_idx] = retrieve_min_reduced_costs(F, part_duals, decisions, job_idx, powers, sequence, duration, cost_t, dual_feas_tol=dual_feas_tol, last=False)

    if additional_paths:
        return all_min_red_costs, all_schedules, all_schedules_add, all_min_red_costs_add

    return all_min_red_costs, all_schedules


@timeit_accumulate(timing_store["cumsum_rc"])
@njit
def compute_rc_term_njit(powers, cost_vec, KP_ineq_arr, res_duals_arr, disj_duals_arr):
    n_machines = powers.shape[0]
    C_max = cost_vec.shape[0]
    n_KP = KP_ineq_arr.shape[0]

    rc_term = np.zeros((n_machines, C_max))

    for m in range(n_machines):
        p = powers[m]
        for tau in range(C_max):
            c = cost_vec[tau]
            dot_sum = 0.0
            for i in range(n_KP):
                dot_sum += KP_ineq_arr[i, m] * res_duals_arr[tau, i]
            rc_term[m, tau] = p * c - dot_sum - disj_duals_arr[m, tau]

    return rc_term


@iter_accumulate(iter_store["cumsum_rc"])
@timeit_accumulate(timing_store["cumsum_rc"])
def compute_cumsum(sum_rc_term):
    cumsum_rc = np.cumsum(sum_rc_term, axis=1)
    cumsum_rc = np.pad(cumsum_rc, ((0, 0), (1, 0)), mode='constant')  # shape: (M, T+1)
    return cumsum_rc


def recover_path(path):
    n = len(path) - 1
    t = len(path[0]) - 1
    steps = []
    while n > 0 and t >= 0:
        prev_n, prev_t = path[n][t]
        if prev_n == n:  # skipped current machine
            t = prev_t
        else:  # used machine n at time t
            steps.append((n, t))
            n, t = prev_n, prev_t
    return steps[::-1]


def compute_opt_schedule(M, C_max, dur, seq, W_m, C_t, val_bool=True):
    """
    """
    F = [[float('inf') for _ in range(C_max)] for _ in range(M + 1)]
    path = [[None for _ in range(C_max)] for _ in range(M + 1)]

    q = dur

    for t in range(C_max):
        F[0][t] = 0  # base case

    for i in range(1, M + 1):
        m = seq[i - 1] - 1
        q_i = q[m]
        for t in range(C_max):
            first_term = F[i][t - 1] if t > 0 else float('inf')

            if t - q_i < 0:
                second_term = float('inf')
            else:
                sum_2nd_term = sum(W_m[m] * C_t[tau] for tau in range(t - q_i + 1, t + 1))  # TODO MODIFY THIS TO ACCOUNT FOR MULTIPLE KP INEQS
                second_term = F[i - 1][t - q_i] + sum_2nd_term

            if first_term <= second_term:
                F[i][t] = first_term
                path[i][t] = (i, t - 1)
            else:
                F[i][t] = second_term
                path[i][t] = (i - 1, t - q_i)

    if val_bool:
        return F[-1][-1]
    else:
        return recover_path(path)
