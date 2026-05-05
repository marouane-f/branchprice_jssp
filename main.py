import runpy
from dataclasses import dataclass
from typing import List

from formulations.extended import write_stats as write_stats_bnp, extended_kpjssp
from formulations.compact_aggr import build_kpjsspmodel as build_kpjsspmodel_aggr
from formulations.compact_disaggr import build_kpjsspmodel as build_kpjsspmodel_disaggr

import numpy as np

from utils import FacetIneq, timing_store, iter_store, reset_store, gen_cost_func


@dataclass
class Instance:
    c_max_base: int
    m: int
    j: int
    seq: List[List[int]]
    dur: List[List[int]]

    @staticmethod
    def load(jssp_name: str):
        data = runpy.run_path('formulations/jssp_instances/' + jssp_name + '.py')
        return Instance(c_max_base=data['Cmax_base'], m=data['M'], j=data['J'], seq=data['seq'], dur=data['dur'])


def solve(
        instance_name: str,
        type: str = 'bnp',  # 'disaggr' or 'aggr'
        cost_func_type: str = 'rand',
        w_m_type: str = 'lowvar',
        rhs_diff: float = -1,
        lambd: float = 1.1,
        verbose: int = 0,
        reduc_bool: bool = False,
        sg_bool: bool = False,
        pwc_bool: bool = False,
        wentges_bool: bool = True,
        partial_bool: bool = False,
        cg_bool: bool = False,
        heur_bool: bool = True,
        rounding: str = None,
        comp_bool: bool = True,
        str_branch: bool = True,
        dom_propag: bool = True,
        early_branch_int: bool = True,
        early_branch_tail: bool = False,
        early_branch_lars: bool = False,
        reli_budget: int = 5,
        n_pricing_iters: int = 0,
        output_dir: str = "./",
):
    """
    Parameters
    ----------
    instance_name : str
        the name of the instance
    cost_func_type : str
        the type of cost function, supported values are ['rand', 'tou12']
    w_m_type : str
        the type of distribution
    """

    ALL_INSTANCE_NAMES = ['4x4', '6x6', '6x6str', '5x10_25', '5x10_25str', '7x7_50', '7x7_100', '5x10_50', '5x10_100', '4x12_50', '4x12_100', '8x8']
    COST_FUNC_TYPES = ['tou_pyr', 'tou1231', 'rand', 'tou12', 'tou11', 'v', 'bar_v']
    W_M_TYPES = ['lowvar', 'incr', 'sl', 'wsi']

    assert instance_name in ALL_INSTANCE_NAMES, 'instance_name must be one of {}'.format(ALL_INSTANCE_NAMES)
    assert cost_func_type in COST_FUNC_TYPES, 'cost_func_type must be one of {}'.format(COST_FUNC_TYPES)
    assert w_m_type in W_M_TYPES, 'W_m_type must be one of {}'.format(W_M_TYPES)

    if 'str' in instance_name:  # stretched/scaled Cmax intances
       instance_name = instance_name[:-3]
       lambd = np.round((lambd - 1) * 25,decimals=2)

    instance = Instance.load(instance_name)
    w_m, KP_ineq, cost_t, c_max = compute_solver_params(w_m_type, rhs_diff, lambd, cost_func_type, instance)

    reset_store(timing_store)
    reset_store(iter_store)

    log_file_name = output_dir + "full_" + instance_name + w_m_type + "_KP--" + str(len(w_m) + rhs_diff) + "_lmbd--" + str(lambd) + "_Ct--" + cost_func_type + ".log"

    if type == 'bnp':
        kp_model, kp_pricer, _, start, end = extended_kpjssp(
            sg_algo_bool=sg_bool,
            cg_bool=cg_bool,
            early_branch_int=early_branch_int,
            early_branch_tailoff=early_branch_tail,
            early_branch_lars=early_branch_lars,
            wentges_bool=wentges_bool,
            heur_bool=heur_bool * (1 - cg_bool),
            comp_pricing_bool=comp_bool,
            Cost_t=cost_t,
            log_file_name=log_file_name,
            pb_data=[instance.j, instance.m, np.array(w_m), instance.dur, instance.seq, cost_t, KP_ineq],
            rounding_mode=rounding,
            verbose=verbose,
            reduc_bool=reduc_bool,
            strong_br_bool=str_branch,
            dom_propag_bool=dom_propag,
            reli_budget=reli_budget,
            n_pricing_iters=n_pricing_iters,
        )
        write_stats_bnp([instance_name, w_m_type, rhs_diff, lambd, cost_func_type], kp_pricer, start, end, timing_store, kp_model,
                        instance.j,
                        log_file=log_file_name, output_dir=output_dir, print_mode=True)

    elif type == 'aggr':
        prec = [[j, instance.seq[j][m - 1] - 1, j, instance.seq[j][m] - 1] for j in range(instance.j) for m in range(1, instance.m)]
        T = range(c_max)
        kp_model = build_kpjsspmodel_aggr(
            J=instance.j,
            M=instance.m,
            W_m=w_m,
            dur=instance.dur,
            prec=prec,
            Cost_t=cost_t,
            KP_ineq=KP_ineq,
            T=T
        )
        kp_model.optimize()
        kp_model.printStatistics()

    elif type == 'disaggr':
        prec = [[j, instance.seq[j][m - 1] - 1, j, instance.seq[j][m] - 1] for j in range(instance.j) for m in range(1, instance.m)]
        T = range(c_max)
        kp_model = build_kpjsspmodel_disaggr(
            J=instance.j,
            M=instance.m,
            W_m=w_m,
            dur=instance.dur,
            prec=prec,
            Cost_t=cost_t,
            KP_ineq=KP_ineq,
            T=T
        )
        kp_model.optimize()
        kp_model.printStatistics()


def compute_solver_params(
        dist_type: str,
        rhs_diff: float,
        lambd: float,
        cost_func_type: str,
        instance: Instance,
        pwc: bool = False,
):
    """
    Parameters
    ---------
    dist_type : str
    the type of the distribution
    m : int
    the number of machines
    """

    if dist_type == 'lowvar':
        np.random.seed(123)
        w_m = 5 + np.random.randint(-1, 2, instance.m)
    elif dist_type == 'incr':
        w_m = [_ for _ in range(instance.m)]
    elif dist_type == 'sl':
        np.random.seed(124)
        w_1 = 3 + np.random.randint(-2, 3, instance.m // 2)
        np.random.seed(125)
        w_2 = 10 + np.random.randint(-2, 3, instance.m - instance.m // 2)
        w_m = np.concatenate((w_1, w_2))
    elif dist_type == 'wsi':
        w_m = [np.power(2, i) for i in range(instance.m)]
    else:
        raise ValueError('W_m_type must be lowvar, incr, sl or wsi')

    KP_ineq = [FacetIneq(coeffs=[1 for _ in range(instance.m)], rhs=instance.m + rhs_diff)]
    # c_max_base_kp = instance.c_max_base * (1 - rhs_diff / 10)
    c_max_base_kp = instance.c_max_base[rhs_diff]
    c_max = int(lambd * c_max_base_kp)
    if pwc:
        cost_t, c_p, s_p = gen_cost_func(cost_func_type, c_max, pwc=pwc)
    else:
        cost_t = gen_cost_func(cost_func_type, c_max)
    cost_t = np.concatenate([[0], cost_t])
    if pwc:
        return w_m, KP_ineq, cost_t, c_p, s_p, c_max
    return w_m, KP_ineq, cost_t, c_max


# python main.py 6x6 --type bnp|aggr|disaggr --cost-func-type rand|tou12 --w-m-type lowvar|incr|sl --rhs-diff -1 --lambd 1.1
if __name__ == "__main__":
    import typer

    try:
        typer.run(solve)
    except Exception as e:
        typer.echo(f"Error: {e}")
