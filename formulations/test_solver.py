"""
Test script for JSSP-KP solver.
Runs small instances and verifies objective values.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Instance, compute_solver_params
from utils import timing_store, reset_store
import numpy as np


def run_solver(instance_name, solver_type, cost_func_type="rand", w_m_type="lowvar",
               rhs_diff=-1, lambd=1.1):
    """
    Run the solver and return the objective value.
    """
    instance = Instance.load(instance_name)
    w_m, KP_ineq, cost_t, c_max = compute_solver_params(
        w_m_type, rhs_diff, lambd, cost_func_type, instance
    )

    reset_store(timing_store)

    if solver_type == 'bnp':
        from formulations.extended import extended_kpjssp
        kp_model, _, _ = extended_kpjssp(
            sg_algo_bool=False,
            cg_bool=False,
            disj_branch=True,
            partial_bool=False,
            early_branch_int=True,
            early_branch_tailoff=False,
            early_branch_lars=False,
            wentges_bool=True,
            heur_bool=True,
            comp_pricing_bool=False,
            Cost_t=cost_t,
            pwc_bool=False,
            log_file_name=None,
            pb_data=[instance.j, instance.m, np.array(w_m), instance.dur, instance.seq, cost_t, KP_ineq],
            rounding_mode="UNFIX",
            verbose=0,
            reduc_bool=False,
            strong_br_bool=True,
            reli_budget=5,
            n_pricing_iters=2,
        )
    elif solver_type == 'aggr':
        from formulations.compact_aggr import build_kpjsspmodel as build_kpjsspmodel_aggr
        prec = [[j, instance.seq[j][m - 1] - 1, j, instance.seq[j][m] - 1]
                for j in range(instance.j) for m in range(1, instance.m)]
        T = range(c_max)
        kp_model = build_kpjsspmodel_aggr(
            J=instance.j, M=instance.m, W_m=w_m, dur=instance.dur,
            prec=prec, Cost_t=cost_t, KP_ineq=KP_ineq, T=T
        )
        kp_model.optimize()
    elif solver_type == 'disaggr':
        from formulations.compact_disaggr import build_kpjsspmodel as build_kpjsspmodel_disaggr
        prec = [[j, instance.seq[j][m - 1] - 1, j, instance.seq[j][m] - 1]
                for j in range(instance.j) for m in range(1, instance.m)]
        T = range(c_max)
        kp_model = build_kpjsspmodel_disaggr(
            J=instance.j, M=instance.m, W_m=w_m, dur=instance.dur,
            prec=prec, Cost_t=cost_t, KP_ineq=KP_ineq, T=T
        )
        kp_model.optimize()
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    status = kp_model.getStatus()
    obj_val = kp_model.getObjVal() if status == 'optimal' else None

    return {
        'status': status,
        'obj_val': obj_val,
        'model': kp_model
    }


class TestSolver6x6:
    """Test cases for 6x6 instance."""

    def test_bnp_rand(self):
        """Test branch-and-price with random cost function."""
        result = run_solver('6x6', 'bnp', cost_func_type='rand')
        assert result['status'] == 'optimal'
        assert result['obj_val'] == pytest.approx(104444.0, rel=1e-3)

    def test_bnp_tou12(self):
        """Test branch-and-price with tou12 cost function."""
        result = run_solver('6x6', 'bnp', cost_func_type='tou12')
        assert result['status'] == 'optimal'
        assert result['obj_val'] == pytest.approx(14519.0, rel=1e-3)

    def test_bnp_tou11(self):
        """Test branch-and-price with tou11 cost function."""
        result = run_solver('6x6', 'bnp', cost_func_type='tou11')
        assert result['status'] == 'optimal'
        assert result['obj_val'] == pytest.approx(16096.0, rel=1e-3)

    def test_aggr_rand(self):
        """Test aggregated formulation with random cost function."""
        result = run_solver('6x6', 'aggr', cost_func_type='rand')
        assert result['status'] == 'optimal'
        assert result['obj_val'] == pytest.approx(104444.0, rel=1e-3)

    def test_disaggr_rand(self):
        """Test disaggregated formulation with random cost function."""
        result = run_solver('6x6', 'disaggr', cost_func_type='rand')
        assert result['status'] == 'optimal'
        assert result['obj_val'] == pytest.approx(104444.0, rel=1e-3)


class TestSolverConsistency:
    """Test that different solver types produce consistent results."""

    def test_all_solvers_same_objective(self):
        """All solver types should find the same optimal objective."""
        results = {}
        for solver_type in ['bnp', 'aggr', 'disaggr']:
            results[solver_type] = run_solver('6x6', solver_type, cost_func_type='rand')

        obj_values = [r['obj_val'] for r in results.values()]
        assert all(v == pytest.approx(obj_values[0], rel=1e-3) for v in obj_values), \
            f"Objective values differ: {results}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
