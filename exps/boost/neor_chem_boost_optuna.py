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

from optuna import Trial, Study
from optuna.samplers import TPESampler
import optuna

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 'OTN_SUZ')

dataset_name = 'Neor_him_meh_optuna_otn_suz'

def objective(trial):
    num_iter = trial.suggest_int("num_iter", 400, 1500)
    m = trial.suggest_int("m", 2, 20)
    max_cov = trial.suggest_int("max_cov", 500, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.9)
    model = BoostingElementaryPredicatesv2(num_iter, m, max_cov, learning_rate)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    return r2



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

with open(f'results_{dataset_name}.txt', 'w') as f:
    f.write(f'Best trial: {study.best_trial}\n')
    f.write(f'Best score: {study.best_value}\n')
    f.write(f'Best params: {study.best_params}\n')
