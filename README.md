# NQS-Dynamics
Python code on Neural Quantum States (NQS) for quantum dynamics, using Stochastic Reconfiguration (SR) for ground state optimization and McLachlan's Time-Dependent Variational Principle (TDVP) for real-time evolution of the 3D Quantum Harmonic Oscillator (QHO) and the deuteron using an effective field theory Hamiltonian.

## Thesis information

This repository contains the code developed for the master's thesis:

**Time-Dependent Neural Quantum States for Quantum Dynamics**

**Author:** Miguel Fernández Suárez  
**Supervisors:** Arnau Rios Huguet and Javier Rozalén Sarmiento

## Overview

The code applies NQS to two quantum systems:

1. the three-dimensional QHO;
2. the deuteron, described with a leading-order pionless effective field theory Hamiltonian.

The workflow consists of two main stages:

1. ground-state optimization using SR;
2. real-time evolution using McLachlan's TDVP.

The wavefunction is parametrized by two real-valued feed-forward neural networks, representing separately the logarithm of the amplitude and the phase.

## Requirements

```text
numpy
scipy
matplotlib
tqdm
torch
```

They can be installed with `pip install -r requirements.txt`. 

## Important computational note

The default parameters in this repository are computationally expensive and are expected to be run on a computing cluster or high-performance computing environment.

Running the full scripts on a regular personal computer may take a very long time and may fail because of memory or computational limitations.

The .npz files used as exact numerical reference data for the deuteron cases are not included in this repository. Before running the main deuteron scripts, `Exact/exact_deuteron.py` should be run first with the specific parameters of the desired run.

In order to run the simulations, execute the main Python file of the corresponding case.

## Quantum Harmonic Oscillator
```text
main_QHO.py
ground_state_QHO.py
time_evolution_QHO.py
```

## Deuteron
```text
main.py
ground_state.py
time_evolution.py
```

## Deuteron in a Periodic Box
```text
main_PBC.py
ground_state_PBC.py
time_evolution_PBC.py
```
