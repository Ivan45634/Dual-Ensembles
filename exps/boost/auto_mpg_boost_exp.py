import sys
sys.path.append("/Users/admin/Desktop/diploma")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.optimize import minimize
from utils.utilites import preprocess
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold
from scipy.optimize import minimize
import numpy.ma as mask
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample, check_random_state

#algos
from models.runc import RuncDualizer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from models.boost_models import BoostingElementaryPredicates, BoostingElementaryPredicatesv2

from tqdm import tqdm
from math import sqrt
import math
import itertools
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from utils.own_forest import *
from utils.utilites import *
from utils.own_forest import *
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from ucimlrepo import fetch_ucirepo 

from time import time
from ucimlrepo import fetch_ucirepo 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# def hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name, n_iter=10, cv=5):
#     scoring_rmse = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
#     scoring_r2 = make_scorer(r2_score)

#     randomized_search = RandomizedSearchCV(estimator=model,
#                                            param_distributions=params,
#                                            n_iter=n_iter, 
#                                            cv=cv, 
#                                            random_state=42)
#     randomized_search.fit(X_train, y_train) 

#     best_model = randomized_search.best_estimator_
    
#     cv_rmse_scores = cross_val_score(best_model, X_test, y_test, cv=cv, scoring=scoring_rmse)
#     cv_r2_scores = cross_val_score(best_model, X_test, y_test, cv=cv, scoring=scoring_r2)

#     mean_rmse = np.mean(-cv_rmse_scores)
#     std_rmse = np.std(cv_rmse_scores)

#     mean_r2 = np.mean(cv_r2_scores)
#     std_r2 = np.std(cv_r2_scores)
    
#     print(f"{model_name} on {dataset_name}: RMSE: {mean_rmse:.3f} (+/- {std_rmse:.3f}), R2: {mean_r2:.3f} (+/- {std_r2:.3f})")

#     return best_model, randomized_search.best_params_, mean_rmse, std_rmse, mean_r2, std_r2


auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

data = pd.concat([X, y], axis=1)

data = data.dropna()

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 'mpg')

dataset_name = 'Auto_MPG_1'

lb_params =  {
    "num_iter" : np.linspace(700, 1500, 10).astype(int),
    "m" : np.linspace(2, 20, 10).astype(int),
    "max_cov": [500],
    'learning_rate': [0.1, 0.5, 0.9]
}

lgbm_params = {
    'num_leaves': [x for x in range(2, 32, 8)], 
    'max_depth': [x for x in range(3, 7)],
    'learning_rate': [0.1]
}

cb_params = {
    'depth': [x for x in range(3, 11)], 
    'learning_rate': [0.1],
    'grow_policy': ['SymmetricTree', 'Depthwise'], 
    'score_function': ['Cosine', 'L2']
}

gb_params = {
    'n_estimators': [50, 100, 150, 300, 500], 
    'max_depth': [3, 5, 7, 9]
}

models = [
    ("BoostingElementaryPredicates", BoostingElementaryPredicatesv2(max_cov=500), lb_params),
    ("LightGBM", LGBMRegressor(num_threads=1, verbose=-1), lgbm_params),
    ("CatBoost", CatBoostRegressor(thread_count=1), cb_params),
    ("GBRegressor", GradientBoostingRegressor(), gb_params)
]

filename = f"results_{dataset_name}.txt"

results = []

with open(filename, 'w') as file:
    for model_name, model, params in models:
        _, best_params, train_rmse, test_rmse, train_r2, test_r2 = hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name)
        results.append((dataset_name, model_name, train_rmse, test_rmse, train_r2, test_r2))
        # Запись результатов в файл
        file.write(f"Best hyperparameters for {model_name} on {dataset_name}: {best_params}")
        file.write(f"{model_name} Results:\n")
        file.write(f"Train RMSE: {train_rmse}\n")
        file.write(f"Test RMSE: {test_rmse}\n")
        file.write(f"Train R2: {train_r2}\n")
        file.write(f"Test R2: {test_r2}\n")
        file.write("-----------------------------------\n")

    results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'Train RMSE', 'Test RMSE', 'Train R2', 'Test R2'])
    file.write(f"\nSummary Results:\n")
    file.write(results_df.to_string(index=False))

print(f"Results have been saved to {filename}")

# filename = f"results_{dataset_name}.txt"

# with open(filename, 'w') as file:
#     for model_name, model, params in models:
#         _, best_params, mean_rmse, std_rmse, mean_r2, std_r2 = hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name)
        
#         results.append((dataset_name, model_name, f"{mean_rmse:.3f} +- {std_rmse:.3f}", f"{mean_r2:.3f} +- {std_r2:.3f}"))
        
#         file.write(f"Best hyperparameters for {model_name} on {dataset_name}: {best_params}\n")
#         file.write(f"{model_name} Results:\n")
#         file.write(f"Train/Test RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}\n")
#         file.write(f"Train/Test R2: {mean_r2:.3f} ± {std_r2:.3f}\n")
#         file.write("-----------------------------------\n")

#     results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'RMSE (mean ± std)', 'R2 (mean ± std)'])
#     file.write(f"\nSummary Results:\n")
#     file.write(results_df.to_string(index=False))

# print(f"Results have been saved to {filename}")
