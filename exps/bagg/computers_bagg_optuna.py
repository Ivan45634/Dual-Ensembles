import numpy as np
from ucimlrepo import fetch_ucirepo
from models.runc import RuncDualizer
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from models.boost_models import *
from utils.utilites import *
from utils.own_forest import *
from models.bagging_trees import BaggingElementaryPredicates
import optuna


computer_hardware = fetch_ucirepo(id=29) 
  
# data (as pandas dataframes) 
X = computer_hardware.data.features 
y = computer_hardware.data.targets 
  
# # metadata 
# print(computer_hardware.metadata) 
  
# # variable information 
# print(computer_hardware.variables) 

# X = X.drop(columns='VendorName', axis=1)

# X = X.drop(columns='ERP', axis=1)

X = X.drop(columns='ModelName', axis=1)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, 'PRP')

dataset_name = 'Computer_Hardware_bagg'


combinations = [BoostingElementaryPredicatesv2(num_iter=i, m=j, max_cov=500, learning_rate=1.0)
                for i in np.linspace(300, 700, 5).astype(int)
                for j in range(2, 21)]

# Создаем словарь для отображения индексов в объекты
combinations_dict = {str(index): combo for index, combo in enumerate(combinations)}

def objective(trial):
    params = {
        'base_estimator': combinations_dict[trial.suggest_categorical('base_estimator', list(combinations_dict.keys()))],
        'n_estimators': trial.suggest_int('n_estimators', 70, 200),
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

with open(f'results_{dataset_name}.txt', 'w') as f:
    f.write(f'Best trial: {study.best_trial}\n')
    f.write(f'Best score: {study.best_value}\n')
    f.write(f'Best params: {study.best_params}\n')
