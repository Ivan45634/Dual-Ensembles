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
from ucimlrepo import fetch_ucirepo 

from time import time
from ucimlrepo import fetch_ucirepo 
import optuna

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


data = pd.read_csv('datasets/02_yachts.csv', sep=',', header=None)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 6)

dataset_name = 'Yachts_optuna'

combinations = [BoostingElementaryPredicatesv2(num_iter=i, m=j, max_cov=500, learning_rate=1.0)
                for i in np.linspace(300, 600, 5).astype(int)
                for j in range(2, 21)]

# Создаем словарь для отображения индексов в объекты
combinations_dict = {str(index): combo for index, combo in enumerate(combinations)}

def objective(trial):
    params = {
        'base_estimator': combinations_dict[trial.suggest_categorical('base_estimator', list(combinations_dict.keys()))],
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'sub_sample_size': trial.suggest_float('sub_sample_size', 0.3, 0.7),
        'sub_feature_size': trial.suggest_float('sub_feature_size', 0.5, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True]),
        'bootstrap_features': trial.suggest_categorical('bootstrap_features', [False]),
        'n_jobs': trial.suggest_categorical('n_jobs', [1]),
    }
    # модель инициализация/обучение/валидация
    model = BaggingElementaryPredicates(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)


