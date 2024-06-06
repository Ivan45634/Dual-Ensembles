import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.optimize import minimize

from models.bagging_trees import BaggingElementaryTrees

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
from models.boost_models import BoostingElementaryPredicates, BoostingElementaryPredicates1
import time
from tqdm import tqdm
from math import sqrt
import math
import itertools
from sklearn.datasets import make_regression

from joblib import Parallel, delayed
from utils.own_forest import *
from utils.utilites import *
from utils.own_forest import *
from sklearn.base import clone


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

X_train, X_test, y_train, y_test = preprocess(dataset, 'MPG')
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

n_estimators_list = [10, 15, 20, 25] 
rmse_values = []

for n_estimators in n_estimators_list:
    model = BaggingElementaryTrees(base_estimator=BoostingElementaryPredicates(num_iter=11, m=40),
                                   n_estimators=n_estimators,
                                   sub_sample_size=1.0,
                                   sub_feature_size=1.0,
                                   n_jobs=-1,
                                   random_state=42)
    
   
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    current_rmse = mean_squared_error(y_val, predictions, squared=False)
    
    rmse_values.append(current_rmse)
    print(f"n_estimators: {n_estimators}, RMSE: {current_rmse}")

# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_list, rmse_values, marker='o', linestyle='-')
# plt.title('Зависимость RMSE от числа деревьев в бэггинге')
# plt.xlabel('Количество деревьев')
# plt.ylabel('RMSE')
# plt.grid(True)
# plt.show()