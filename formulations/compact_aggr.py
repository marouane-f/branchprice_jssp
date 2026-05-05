import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pyscipopt import Model, quicksum
import numpy as np

def build_kpjsspmodel(J, M, W_m, dur, prec, T, Cost_t, KP_ineq):
    model = Model("kp_jssp")
    model.setParam('limits/time', 3600)
    model.setParam('limits/gap', 1e-5)      # TODO: add as argument
    model.setParam("display/freq", 100)
    model.setMinimize()
    X = {}
    T = range(1, len(T)+1)

    for j in range(J):
        for m in range(M):
            for t in T:
                X[j, m, t] = model.addVar(name=f"X_{j}_{m}_{t}", lb=0, ub=1, vtype="B")

    # Objective function: Minimize cost
    model.setObjective(
        quicksum(
            Cost_t[t] * quicksum(W_m[m] * quicksum(
                quicksum(X[j, m, t1] for t1 in T if t <= t1 <= t + dur[j][m] - 1) for j in range(J)) for m in range(M)) for t in T),
        "minimize"
    )

    # Disjunction constraint
    for t in T:
        for m in range(M):
            model.addCons(quicksum(quicksum(X[j, m, t1] for t1 in T if t <= t1 <= t + dur[j][m] - 1) for j in range(J)) <= 1)

    # # Each job must be executed exactly once
    for j in range(J):
        for m in range(M):
            model.addCons(quicksum(X[j, m, t] for t in T if t >= dur[j][m]) == 1)
            model.addCons(quicksum(X[j, m, t] for t in T if t <= dur[j][m] - 1) == 0)

    # Precedence constraint

    for (k, n, j, m) in prec:
        model.addCons(
            sum(t1 * X[k, n, t1] for t1 in T) <=
            sum(t1 * X[j, m, t1] for t1 in T) - dur[j][m]
        )

    # Maximum power constraint
    for kp_ineq in KP_ineq:
        coeffs = kp_ineq.coefficients
        rhs = kp_ineq.rhs
        for t in T:
            model.addCons(quicksum(coeffs[m] * quicksum(X[j, m, t1] for t1 in T if t <= t1 <= t + dur[j][m] - 1) for j in range(J) for m in range(M)) <= rhs)
    model.setObjIntegral()

    return model

def write_stats(inst_char, start, end, ps_time, mod, log_file):
    with open(log_file, "a") as f:

        sol_status = mod.getStatus()

        tot_time = np.round(end - start, decimals=3)

        f.write(f"(dim, pow, rhs, lamb, c_t)  {inst_char}\n")
        f.write(f"Solution status: {sol_status}\n")

        f.write('\n________________________________________________\n\n')
        f.write(f"{'time total':<9} {tot_time:>9.3f}\n")
        f.write(f"{'  presolve':<9} {ps_time:>9.3f}\n\n")
        f.write(f"Nodes#  \t {mod.getNNodes()}\n")
        f.write(f"Obj     \t {np.round(mod.getObjVal(), decimals=1)}\n")
        if sol_status != 'optimal':
            f.write(f"Gap \t {mod.getGap()}\n")
            f.write(f"LB \t {mod.getLowerbound()}\n")
        f.write('\n________________________________________________________________________________________________\n\n')
