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
from models.bagging_trees import BaggingElementaryTrees

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


data = pd.read_csv('datasets/02_yachts.csv', sep=',', header=None)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 6)

dataset_name = 'Yachts'

lb_params =  {
    "num_iter" : np.linspace(300, 600, 5).astype(int),
    "m" : np.linspace(3, 20, 8).astype(int),
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

results = []

filename = f"results_{dataset_name}.txt"

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
