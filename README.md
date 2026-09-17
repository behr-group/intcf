This repository contains implementation and files for simulations of [Splitting the Difference: Interpretable Causal Forests for Treatment Effect Heterogeneity and Bias](https://arxiv.org/abs/2609.16971).

# Introduction

# Package
An implementation of Interpretable Causal Forest (IntCF) is contained in directory `package`. This implementation is also required for the simulations.

# Simulations
## CausalPFN
To evaluate IntCF against established baselines, we used the framework of [CausalPFN](https://github.com/vdblm/CausalPFN/). The files in directory `CausalPFN` need to be combined with their files to enable evaluation of IntCF and Causal Forest. Files `CausalPFN/causal_forest.py` and `CausalPFN/intcf.py` should be placed in `benchmarks/baselines` from the repository CausalPFN, while `CausalPFN/causal_effect_add.ipynb` should be placed and run in `notebooks` from this repository.

## Synthetic data
To evaluate predictions and interpretation of IntCF, synthetic data is used. The files for these simulations are contained in directory `synthetic_data`.
The code for running the simulation, including data generation, training, predictions and feature importance evaluation, can be executed by running file `synthetic_data/parallel_eval.py`. The conda-environment defined by `synthetic_data/environment.yaml` should be used for this.
Evaluation of these simulations and creation of plots are possible with `synthetic_data/summary_creation.ipynb`, where necessary packages are listed in `synthetic_data/requirements.txt`.

## NHEFS
This simulation used data from the NHANES I Epidemiologic Follow-up Study (NHEFS), more information about this study can be found at https://wwwn.cdc.gov/Nchs/Nhanes/nhefs/default.aspx. Here we are using the data set available at https://miguelhernan.org/whatifbook, which we include as `nhefs/nhefs.csv`. The notebook `nhefs/nhefs.ipynb` contains the code for training IntCF, making predictions and evaluating the feature importance scores.
