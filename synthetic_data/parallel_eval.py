import concurrent.futures
import time

import numpy as np
from datamodels import data_models
from interface import (
    GRF,
    CausalForest,
    EconmlForestDML,
    GRFconstOrtho,
    IntCF,
    IntCFunvalidated,
)

seed = 3141592  # seed for rng and trees
N_eval = 100_000  # number of features used for evaluation of MSE
iters = 20  # Number of training (and evaluation) runs
num_trees = 100  # number of trees trained
importance_max_depth = None  # depth for feature importance, None for no limit

def single_simulation(model, s, iter, estimation_method):
    N = 500 * s  # Number of training samples
    # reseed for every parameter to make results only depend on parameter, not position in list
    rng = np.random.default_rng(seed)
    eval_covariates, _, _, eval_treatment_effect = model(N_eval, rng)
    
    rng = np.random.default_rng(seed + iter)
    covariates, treatment, outcomes, _ = model(N, rng)
    # Dummy estimator for comparison
    dummy_te = outcomes[treatment].mean() - outcomes[~treatment].mean()
    mse_0 = ((dummy_te - eval_treatment_effect)**2).mean()
    # Treatment effect estimation
    cf = estimation_method(n_estimators=num_trees, random_state=seed)
    cf.fit(X=covariates, T=treatment, y=outcomes)
    mse = ((cf.predict(eval_covariates) - eval_treatment_effect)**2).mean()
    feat_importance_het = cf.feature_importances(
        heterogeneity=True,
        max_depth=importance_max_depth
    )
    return f'{s},{iter},{model.name},{estimation_method.name},{mse},{mse_0},"{feat_importance_het.tolist()}"\n'

def simulation(s, iter, estimation_method):
    return "".join(single_simulation(model, s, iter, estimation_method) for model in data_models(s))

def main():
    save_path = "/home/ihn29298/CausalForest/"
    # save_path = "/Users/ihn29298/Desktop/CausalForest/simulations/"
    s_params = range(1, 11)  # parameter for number of signal features
    with open(save_path + f"results-depth={importance_max_depth}+{time.strftime('%Y-%m-%d+%H%M%S')}.csv", "w", buffering=1) as file:
        # file.write("s,iter,model,Estimation method,MSE,MSE_0,Heterogeneity feature importance,Confounder feature importance\n")
        file.write("s,iter,model,Estimation method,MSE,MSE_0,Feature importance\n")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fs = []
            for method in [IntCF, IntCFunvalidated, EconmlForestDML, CausalForest, GRF, GRFconstOrtho]:
                for s in s_params:
                    for iter in range(iters):
                        fs.append(executor.submit(simulation, s, iter, method))
            for future in concurrent.futures.as_completed(fs):
                file.write(future.result())

if __name__ == '__main__':
    main()
