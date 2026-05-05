# Branch and Price for Job-Shop Scheduling with Time-Dependent Costs and Cardinality Resource Constraints

Implementation accompanying:

> M. Felloussi, M. Ghannam, J. Dionísio, P. Gianessi, X. Delorme.
> **Branch and Price for Job-Shop Scheduling with Time-Dependent Costs and Cardinality Resource Constraints.**
> *Proceedings of International Symposium on Combinatorial Optimization (ISCO), 2026.*

We study a discrete-time job-shop scheduling problem (JSSP) with time-varying operating costs and per-time-step limits on the number of simultaneously active machines (cardinality resource constraint). The objective is to minimize total operating cost. The implementation is an exact branch-and-price algorithm over an extended formulation: jobs are represented by feasible schedules (columns), and the pricing subproblem is solved by a forward dynamic program.

Key components:
- **Pricing oracle**: shortest-path in DAG dynamic program over `(operation, completion-time)` states (Cython, in `shortest_path/`).
- **Dual stabilization**: [Automatic directional smoothing (Pessoa et al., 2018)](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0784) (`pricer_directionalsmoothing.py`).
- **Hierarchical branching**: most-dispersed selection, reliable pseudo-costs with strong branching, structure-aware propagation (`origvar_branching.py`).
- **Primal heuristics**: exact tree search with LP-free bounds over the restricted column pool plus large-neighborhood search (`primal_heur/`).

## Installation

The build uses Cython, so a C compiler is required. We use a custom fork of PySCIPOpt that exposes SCIP's probing-with-pricing interface (one extra commit on top of upstream `master`).

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd branchprice_jssp
```

Create the conda environment (installs Python 3.11, SCIP 9.2.3, the PySCIPOpt fork, and Python deps):

```bash
conda env create -f env.yml
conda activate branchprice_jssp_env
```

Build the Cython extensions in place:

```bash
# 1. Shortest-path pricing oracle
python setup.py build_ext --inplace

# 2. Primal-heuristic helpers
cd primal_heur
python setup.py build_ext --inplace
cd ..
```

## Usage

Run a single instance with default parameters:

```bash
python main.py 6x6
```

List all parameters:

```bash
python main.py --help
```

Override key parameters:

```bash
python main.py 6x6 --cost-func-type tou12 --w-m-type incr
```

The `--verbose` flag is a bitmask (default `0`, silent):

| Bit | Binary | Decimal | Output |
|---|---|---|---|
| 0 | `001` | `1` | Column generation progress (RMP, Lagrangian bound, etc.) |
| 1 | `010` | `2` | Branching info: early branching, strong branching, reliability-branching calls |
| 2 | `100` | `4` | RMP heuristic tree exploration and LNS log |

e.g., branching + heuristics = `--verbose 110` or `--verbose 6`.

## Reproducing experiments

The benchmark set extends base JSSP instances with time-dependent costs (504 instances total; see paper §6 for the construction and setup).

## Citation

If you use this code, please cite the paper. A `CITATION.cff` is provided so GitHub renders a "Cite this repository" widget. BibTeX entry:

```bibtex
@inproceedings{felloussijssp2026,
  title     = {Branch and Price for Job-Shop Scheduling with Time-Dependent Costs and Cardinality Resource Constraints},
  author    = {Felloussi, Marouane and Ghannam, Mohammed and Dion{\'\i}sio, Jo{\~a}o and Gianessi, Paolo and Delorme, Xavier},
  booktitle="Combinatorial Optimization",
  year="2026",
  publisher="Springer Nature Switzerland",
}
```
## Questions

For bugs or reproducibility issues, please open a GitHub issue. Questions about the algorithm or the paper are welcome. Contact information is on the corresponding author's personal page <https://marouane-f.github.io>.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
