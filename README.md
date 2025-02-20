# MRAS Bandits

## Project Proposal and Report

Links:

1. [Proposal](proposal.pdf)
2. [Report](report.pdf) 

## TODO

1. [ ] Fix MRAS-Categorical-Subset reproducibility
2. [ ] Tune MRAS-Dirchlet-Subset

## Install and Run

Use `python 3.6`.
`pip install -r requirements.txt`

To run:
```
export PYTHONPATH=.
python sim.py
```

## Benchmark Algorithms

* UCB
* Thomson Sampling
* Aysm-UCB
* KL-UCB (Needs to be sped up)

## Simulation Experiments

We try the following parameter distributions:

1. [x] Categorical
2. [x] Dirchlet
3. [x] Gaussian

We also experiment with the following:

1. Increasing function `H`.
2. Exploitation param `lambda`.
3. Simulation allocation `M_k`.
4. Population size `N_o`

## Adding a new algorithm

* Add under `bandits` folder.
* Add a unit test to verify its working.
* Import it under sim.py

## Working with configs

We use sacred for configs and capturing arguments. Read more [here](https://sacred.readthedocs.io/en/latest/).
Modify base-config.yaml for more games.

If you need very different args, create a new config file and run as:
`python sim.py with configs/new-config.yaml`

## Critical Checks

1. [x] Is regret being computed correctly? Right now we are accumulating (best_mean - reward). This could be negative, but averaged over experiments is positive.
2. [x] Is the UCB implementation correct?