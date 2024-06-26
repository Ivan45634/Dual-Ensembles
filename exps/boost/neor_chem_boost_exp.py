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

from time import time
from ucimlrepo import fetch_ucirepo 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def preprocess(data, target_name, drop=True, index_col=None):
    if isinstance(data, (str, )):
        if index_col is not None:
            df = pd.read_csv(data, index_col=index_col).reset_index()
        else:
            df = pd.read_csv(data)

    else:
        df = data.copy()
    
    if drop and 'index' in df.columns:
        df = df.drop(columns='index')

    # Splitting the dataset into features and target
    X_raw = df.drop(columns=[target_name])
    
    y = df[target_name].to_numpy()
    # y = df[target_name]

    y = np.log(df[target_name].to_numpy()) # Берем логарифм от целевого признака 

    # Normalizing numerical features
    numeric_cols = X_raw.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X_raw[numeric_cols] = scaler.fit_transform(X_raw[numeric_cols])

    # Encoding categorical variables
    X_raw = pd.get_dummies(X_raw, drop_first=True)

    # Splitting data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, train_size=0.8, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=42)

    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train, y_val, y_test


def hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name, n_iter=10):
    scoring_fnc = make_scorer(rmse, greater_is_better=False)
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=params, 
                                           n_iter=n_iter, scoring=scoring_fnc, cv=5, random_state=42)
    randomized_search.fit(X_train, y_train)

    print(f"Best hyperparameters for {model_name} on {dataset_name}: {randomized_search.best_params_}")
    best_model = randomized_search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Переводим предсказания обратно через exp после логарифмирования target'а
    y_pred_train = np.exp(y_pred_train)
    y_pred_test = np.exp(y_pred_test)
    y_train = np.exp(y_train)
    y_test = np.exp(y_test)

    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n--Train RMSE for {model_name} on {dataset_name}--\n{train_rmse}")
    print(f"--Test RMSE for {model_name} on {dataset_name}--\n{test_rmse}\n")
    print(f"--Train R2 for {model_name} on {dataset_name}--\n{train_r2}")
    print(f"--Test R2 for {model_name} on {dataset_name}--\n{test_r2}\n")

    return best_model, randomized_search.best_params_, train_rmse, test_rmse, train_r2, test_r2 


data = pd.read_csv("neor_him_meh.csv")

chemical_data = data.apply(parse_chemical_composition, axis=1)
data = pd.concat([data, chemical_data], axis=1)

data = data.drop(columns=['HIM_NAME_STR', 'EL_ZN_STR'])

# data.to_csv('neor_him_meh_transformed.csv', index=False)

data.drop(columns='N_OBR', inplace=True)

data.drop(columns='SOOTV_ID', inplace=True)

le = LabelEncoder()
data['NPLAV'] = le.fit_transform(data['NPLAV'])

column = 'MFG_ORDER_ID'
data[column] = data[column].fillna('-200').astype(str)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 'VR_SOPR')

dataset_name = 'Neor_him_meh_2'

lb_params =  {
    "num_iter" : np.linspace(500, 1200, 10).astype(int),
    "m" : np.linspace(2, 20, 5).astype(int),
    "max_cov": [500],
    'learning_rate': [0.1, 0.5, 0.9]
}

# lb_params =  {
#     "num_iter" : np.linspace(800, 2000, 10).astype(int),
#     "m" : np.linspace(2, 20, 10).astype(int),
#     "max_cov": [500],
#     'learning_rate': [0.1, 0.5, 0.9]
# }

lgbm_params = {
    'num_leaves': [x for x in range(2, 20, 8)], 
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
    'n_estimators': [50, 100, 150], 
    'max_depth': [3, 5, 7]
}

models = [
    ("BoostingElementaryPredicates", BoostingElementaryPredicatesv2(max_cov=500), lb_params),
    ("LightGBM", LGBMRegressor(num_threads=1, verbose=-1), lgbm_params),
    ("CatBoost", CatBoostRegressor(thread_count=1, logging_level='Verbose'), cb_params),
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
