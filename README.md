# Source code for FMA
This repo includes the implementation of FMA Algorithm described in the paper *Navigation with Time Limits in Transportation
Networks: A Fourth Moment Approach*, and the benchmarks, namely [DOT, Pruning](https://www.sciencedirect.com/science/article/pii/S0191261520303271), [ILP](https://ieeexplore.ieee.org/document/8543229), and [PLM](https://ieeexplore.ieee.org/abstract/document/7273960?casa_token=rMAE3kIG0xkAAAAA:I6GYS4_RNCLbgSXtUE1kJg5e0opekcn9eFL9Z6HQli33LOEg6YpBjqJmeskW9nyDKT9oQN6MM-uV) used for comparison.

## Dataset
The network data used for conducting simulations is available in the *Networks* folder.
The dataset includes 5 networks. Each of them consists of a `.csv` file:
- `.csv`, nodes and links of the network, and mean cost of each link.

For Sioux Falls Network, we also provide a `variance.npy` file which consists of the variance used in the experiment.

## Dependencies
- Python 3.6+
- NumPy
- SciPy
- Pandas
- NetworkX
- gurobipy (a license might be needed)

## Description
The source code includes the following files:
- `main.py`, sample codes for testing FMA and benchmarks on Simple, Sioux Falls, Anaheim, Winnipeg and Chicago-Sketch Networks described in the paper. Parameters are also specified here.
- `fma.py`, implementation of FMA Algorithm.
- `cao.py`, implementation of ILP and PLM Algorithms.
- `prakash.py`, implementation of DOT and Pruning Algorithms.
- `evaluation.py`, function that evaluates the on-time-arrival probability of a path.
- `func.py`, tool functions used throughout the project.
