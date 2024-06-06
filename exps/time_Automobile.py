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
from models.bagging_trees import BaggingElementaryPredicates

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

from time import time
from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Fetch and preprocess data
automobile = fetch_ucirepo(id=10)
X = automobile.data.features
y = automobile.data.targets
X = X[~X['price'].isna()].reset_index()



# Define best hyperparameters
best_params = {
    'BoostingElementaryPredicatesv2': {'num_iter': 300, 'm': 5, 'max_cov': 500, 'learning_rate': 0.1},
    'LightGBM': {'num_leaves': 10, 'max_depth': 4, 'learning_rate': 0.1, 'num_threads': 1, 'verbose': -1},
    'CatBoost': {'score_function': 'Cosine', 'learning_rate': 0.1, 'grow_policy': 'SymmetricTree', 'depth': 3, 'thread_count': 1}
}

results = {}
i = 0
for algorithm, params in best_params.items():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_random(X, 'price', i)
    if algorithm == 'BoostingElementaryPredicatesv2':
        avg_time, std_time = get_avg_time(BoostingElementaryPredicatesv2, params, X_train, y_train, X_test, y_test)
    elif algorithm == 'LightGBM':
        avg_time, std_time = get_avg_time(LGBMRegressor, params, X_train, y_train, X_test, y_test)
    elif algorithm == 'CatBoost':
        avg_time, std_time = get_avg_time(CatBoostRegressor, params, X_train, y_train, X_test, y_test)
    results[algorithm] = {'Average Time': avg_time, 'Std Dev Time': std_time}
    i += 1

results_df = pd.DataFrame(results).T

results_df.to_csv(f'time_Automobile.csv')

print(results_df)
