import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from models.runc import RuncDualizer
import time
import math
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from models.boost_models import BoostingElementaryPredicates, BoostingElementaryPredicates1
from own_forest import *


class BoostingElementaryPredicates(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter=None, m=None, max_cov=None):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []
        # self.input_rows = []

    def fit(self, X, y):
        self.base_value = y_hat = y.mean()
        n = X.shape[1]

        for _ in range(self.num_iter):
            self.input_rows = []
            residuals = y - y_hat
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer() # create new Dualizer

            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X[max_residual_idx]

            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j+n)  # Больше
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)
                    # self.input_rows.append(comp_row)  # Добавление строки в список
            
            # self.save_input_rows()

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0

            while True:
                covers = self.runc.enumerate_covers()
                # print(covers)

                if self.max_cov is not None and len(covers) > self.max_cov: #TODO: добавить стохастику в построение и выбор покрытий
                    covers = covers[:self.max_cov]

                if len(covers) == 0:
                    break

                for cover in covers[:5000]:
                    h_mask_l = np.isin(np.arange(n), cover)
                    h_mask_g = np.isin(np.arange(n, 2*n), cover)
                    H_l = np.where((X[:, h_mask_l] >= X[max_residual_idx][h_mask_l]).all(axis=1), 1, 0)
                    H_g = np.where((X[:, h_mask_g] <= X[max_residual_idx][h_mask_g]).all(axis=1), 1, 0)
                    base_estimator = residuals[max_residual_idx] * H_l * H_g
                    residual_sum_maybe = ((y - y_hat - base_estimator) ** 2).mean()
                    if residual_sum_maybe < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, residual_sum_maybe
                        best_cover = cover
            self.h.append(h_m)
            gamma_m = self.optimize(y, y_hat, h_m)[0]
            self.gamma.append(gamma_m)
            self.covers.append(best_cover)
            self.key_objects.append(X[max_residual_idx])
            self.est_res.append(residuals[max_residual_idx])
            y_hat += gamma_m * h_m

        # del self.runc

        return self
    
    # def save_input_rows(self):
    #     with open('/Users/admin/Desktop/diploma/input_rows.txt', 'w') as file:
    #         for row in self.input_rows:
    #             file.write(','.join(map(str, row)) + '\n')

    def optimize(self, y, y_hat, h):
        loss = lambda gamma: ((y - y_hat - gamma * h) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = self.est_res[i] * H_l * H_g
            y_pred += self.gamma[i] * base_estimator.reshape(-1)

        return y_pred

#--------------------------#

class BoostingElementaryPredicates1(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter, m):
        self.num_iter = num_iter
        self.m = m
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []

    def fit_predict(self, X_train, y_train, X_test, y_test):
        self.base_value = y_hat = y_train.mean()
        n = X_train.shape[1]

        print(self.num_iter)
        for _ in range(self.num_iter):
            residuals = y_train - y_hat
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer()  # create new Dualizer

            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X_train[max_residual_idx]

            # print(residuals[min_m_residual_indices])
            # print(residuals[max_residual_idx])

            # for row_idx in min_m_residual_indices:
            #     row = X[row_idx]
            #     non_zero_indices = np.where(row != X[max_residual_idx])[0]
            #     if len(non_zero_indices) > 0:
            #         self.runc.add_input_row(list(row))
            # comp_rows=[]
            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(n):
                    # if X[idx, j] == key_object[j]:
                    #     comp_row.append(j)  # Равно
                    if X_train[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X_train[idx, j] > key_object[j]:
                        comp_row.append(j+n)  # Больше
                # comp_rows.append(comp_row)
                # print(comp_row)
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0
            while True:
                covers = self.runc.enumerate_covers()
                if len(covers) == 0:
                    break

                for cover in covers:
                    # n = X_train.shape[1]
                    h_mask_l = np.isin(np.arange(n), cover)
                    h_mask_g = np.isin(np.arange(n, 2*n), cover)
                    H_l = np.where((X_train[:, h_mask_l] >= X_train[max_residual_idx][h_mask_l]).all(axis=1), 1, 0)
                    H_g = np.where((X_train[:, h_mask_g] <= X_train[max_residual_idx][h_mask_g]).all(axis=1), 1, 0)
                    base_estimator = residuals[max_residual_idx] * H_l * H_g
                    residual_sum_maybe = ((y_train - y_hat - base_estimator) ** 2).mean()
                    if residual_sum_maybe < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, residual_sum_maybe
                        best_cover = cover

            self.h.append(h_m)
            gamma_m = self.optimize(y_train, y_hat, h_m)[0]
            self.gamma.append(gamma_m)
            self.covers.append(best_cover)
            self.key_objects.append(X_train[max_residual_idx])
            self.est_res.append(residuals[max_residual_idx])
            y_hat += gamma_m * h_m

            # Calculate train and test loss for current iteration
            train_loss = ((y_train - self.predict(X_train)) ** 2).mean()
            test_loss = ((y_test - self.predict(X_test)) ** 2).mean()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        # Plot the train and test loss
        x = np.arange(self.num_iter) + 1
        plt.plot(x, self.train_losses, label='Train Loss')
        plt.plot(x, self.test_losses, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return self

    def optimize(self, y, y_hat, h):
        loss = lambda gamma: ((y - y_hat - gamma * h) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            # h_mask = np.isin(np.arange(X.shape[1]), self.covers[i])
            # res = np.unique(self.h[i][self.h[i] != 0.])
            # base_estimator = res * np.where((X[:, h_mask] == self.key_objects[i][h_mask]).all(axis=1), 1, 0)
            # y_pred += base_estimator.reshape(-1)
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            # res = np.unique(self.h[i][self.h[i] != 0.])
            base_estimator = self.est_res[i] * H_l * H_g
            y_pred += self.gamma[i] * base_estimator.reshape(-1)

        return y_pred
    
class BoostingElementaryPredicates2(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter=None, m=None, max_cov=None):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []

    def fit(self, X, y):

        def loss(params):
                a, b = params
                prediction = y_hat + a * base_estimator + b  # Обновленный предикт
                return ((y - prediction) ** 2).mean()
        self.base_value = y_hat = y.mean()
        n = X.shape[1]

        for _ in range(self.num_iter):
            residuals = y - y_hat
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer()

            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X[max_residual_idx]

            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j+n)  # Больше
                # print(comp_row)
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0

            while True:
                covers = self.runc.enumerate_covers()
                # print(covers)

                if self.max_cov is not None and len(covers) > self.max_cov: #TODO: добавить стохастику в построение и выбор покрытий
                    covers = covers[:self.max_cov]

                if len(covers) == 0:
                    break

                for cover in covers[:5000]:
                    h_mask_l = np.isin(np.arange(n), cover)
                    h_mask_g = np.isin(np.arange(n, 2*n), cover)
                    H_l = np.where((X[:, h_mask_l] >= X[max_residual_idx][h_mask_l]).all(axis=1), 1, 0)
                    H_g = np.where((X[:, h_mask_g] <= X[max_residual_idx][h_mask_g]).all(axis=1), 1, 0)
                    base_estimator = residuals[max_residual_idx] * H_l * H_g
                    # residual_sum_maybe = ((y - y_hat - base_estimator) ** 2).mean()
                    initial_guess = [0.0, 0.0]  # Начальное предположение для a и b
                    result = minimize(loss, x0=initial_guess)
                    a_opt, b_opt = result.x
                    if loss(result.x) < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, loss(result.x)
                        best_cover, best_params = cover, result.x
            self.h.append(h_m)
            # gamma_m = self.optimize(y, y_hat, h_m)[0]
            # self.gamma.append(gamma_m)
            self.gamma.append(best_params) 
            self.covers.append(best_cover)
            self.key_objects.append(X[max_residual_idx])
            self.est_res.append(residuals[max_residual_idx])
            # y_hat += gamma_m * h_m
            a_m, b_m = best_params
            y_hat += a_m * h_m + b_m

        # del self.runc

        return self

    def optimize(self, y, y_hat, h):
        loss = lambda gamma: ((y - y_hat - gamma * h) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = self.est_res[i] * H_l * H_g
            # print(self.gamma[i])
            a_i, b_i = self.gamma[i]
            y_pred += a_i * base_estimator.reshape(-1) + b_i

        return y_pred

class BoostingElementaryPredicatesv2(BaseEstimator, RegressorMixin):
    """
    Модель бустинга на элементарных предикатах. Алгоритм использует итеративный подход
    для улучшения предсказаний, минимизируя остатки на тренировочных данных.
    
    Параметры:
    - num_iter: количество итераций алгоритма
    - m: количество объектов с наименьшими остатками для участия в матрице сравнения
    - max_cov: максимальное количество покрытий
    """
    
    def __init__(self, num_iter=None, m=None, max_cov=None, learning_rate=1.0):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n = X.shape[1]
        self.base_value = y_hat = y.mean()

        for _ in range(self.num_iter):
            residuals = y - y_hat
            min_residual_idx = np.argmin(residuals)
            key_object = X[min_residual_idx]

            self.runc = RuncDualizer()

            # Отбор объектов
            if self.m > 0 and self.m < len(residuals):
                candidates = np.argsort(-residuals)[:self.m]
            else:
                candidates = np.arange(len(residuals))

            # Создание матрицы сравнения опорного объекта с остальными
            for idx in candidates:
                # if residuals[idx] <= 0:  #   нам не надо занулять предикат на тех слагаемых, которые будут уменьшать сумму \sum B(S_i) r_i
                #     break
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j + n)  # Больше
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)
            
            min_residual_sum = float('inf')
            num_processed_covers = 0

            while True:
                covers = self.runc.enumerate_covers()

                if len(covers) == 0:
                    break  # Если покрытий больше нет, выход из цикла

                if self.max_cov is not None and num_processed_covers >= self.max_cov:
                    break
                
                for cover in covers:
                    # Прежде чем обработать покрытие, проверяем, не превышен ли лимит
                    if num_processed_covers >= self.max_cov:
                        break
                    a_opt, b_opt, residual_sum, base_estimator = self.evaluate_coefs_and_residual_sum(cover, X, residuals, min_residual_idx)
                    if residual_sum < min_residual_sum:
                        best_cover = cover
                        best_gamma = a_opt, b_opt
                        min_residual_sum = residual_sum
                        h_m = base_estimator
                    
                    num_processed_covers += 1
                else:
                    continue
                # Если внутренний цикл прерван из-за достижения лимита, прерываем и внешний цикл
                if num_processed_covers >= self.max_cov:
                    break
            
            best_gamma = best_gamma[0] * self.learning_rate, best_gamma[1] * self.learning_rate
            
            self.h.append(best_cover)
            self.gamma.append(best_gamma)
            self.covers.append(best_cover)
            self.key_objects.append(X[min_residual_idx])
            self.est_res.append(residuals[min_residual_idx])
            # y_hat += gamma_m * h_m
            a_m, b_m = best_gamma
            y_hat += a_m * h_m + b_m
        
        return self


    def evaluate_coefs_and_residual_sum(self, cover, X, residuals, min_residual_idx):

        def loss(params):
            a, b = params
            return ((a * base_estimator + b - residuals)**2).mean()
        
        
        n = X.shape[1]

        h_mask_l = np.isin(np.arange(n), cover)
        h_mask_g = np.isin(np.arange(n, 2*n), cover)
        H_l = np.where((X[:, h_mask_l] >= X[min_residual_idx][h_mask_l]).all(axis=1), 1, 0)
        H_g = np.where((X[:, h_mask_g] <= X[min_residual_idx][h_mask_g]).all(axis=1), 1, 0)
        base_estimator = H_l * H_g
        # result = minimize(loss, x0=np.array([1.0, 0.1]), method='BFGS') - численный поиск опт. коэфф.
        # a_opt, b_opt = result.x
        # min_residual_sum = loss(result.x)

        B = base_estimator 
        r = residuals
        sum_r_B = np.sum(r * B)  # Сумма r_i * B(S_i)
        sum_B = np.sum(B)        # Сумма B(S_i)

        sum_r_B1 = np.sum(r * (B - 1))  # Сумма r_i * (B(S_i) - 1)
        sum_B1 = np.sum(B - 1)          # Сумма (B(S_i) - 1)

        if sum_B == 0 or sum_B1 == 0:
            raise ValueError("Division by zero in formula calculation for a_opt or b_opt.")

        a_opt = (sum_r_B / sum_B) - (sum_r_B1 / sum_B1)
        b_opt = sum_r_B1 / sum_B1

        params = [a_opt, b_opt]

        min_residual_sum = loss(params)

        return a_opt, b_opt, min_residual_sum, base_estimator

    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = H_l * H_g
            # print(self.gamma[i])
            a_i, b_i = self.gamma[i]
            y_pred += a_i * base_estimator + b_i

        return y_pred


class BoostingElementaryPredicatesv3(BaseEstimator, RegressorMixin):
    """
    Модель бустинга на элементарных предикатах. Алгоритм использует итеративный подход
    для улучшения предсказаний, минимизируя остатки на тренировочных данных, 
    вместе с критерием останова early stopping.
    
    Параметры:
    - num_iter: количество итераций алгоритма
    - m: количество объектов с наименьшими остатками для участия в матрице сравнения
    - max_cov: максимальное количество покрытий
    """
    
    def __init__(self, num_iter=None, m=None, max_cov=None, learning_rate=1.0, patience=10, validation_fraction=0.1, random_state=42):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.est_res = []
        self.learning_rate = learning_rate
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.val_losses = []

    def fit(self, X, y):
        n = X.shape[1]
        # self.base_value = y_hat = y.mean()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_fraction, random_state=self.random_state)
        self.base_value = y_hat_train = y_train.mean()
        y_hat_val = self.base_value
        
        best_val_loss = float('inf')
        no_improvement_count = 0

        for iteration in range(self.num_iter):
            # residuals = y - y_hat
            residuals = y_train - y_hat_train
            residuals_val = y_val - y_hat_val

            min_residual_idx = np.argmin(residuals)
            min_residual_idx_val = np.argmin(residuals_val)

            key_object = X_train[min_residual_idx]

            self.runc = RuncDualizer()

            # Отбор объектов
            if self.m > 0 and self.m < len(residuals):
                candidates = np.argsort(-residuals)[:self.m]
            else:
                candidates = np.arange(len(residuals))

            for idx in candidates:
                # Создание матрицы сравнения опорного объекта с остальными
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j + n)  # Больше
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)
            
            min_residual_sum = float('inf')
            num_processed_covers = 0

            while True:
                covers = self.runc.enumerate_covers()

                if len(covers) == 0:
                    break  # Если покрытий больше нет, выход из цикла

                if self.max_cov is not None and num_processed_covers >= self.max_cov:
                    break
                
                for cover in covers:
                    # Прежде чем обработать покрытие, проверяем, не превышен ли лимит
                    if num_processed_covers >= self.max_cov:
                        break
                    a_opt, b_opt, residual_sum, base_estimator = self.evaluate_coefs_and_residual_sum(cover, X_train, residuals, min_residual_idx)
                    if residual_sum < min_residual_sum:
                        best_cover = cover
                        best_gamma = a_opt, b_opt
                        min_residual_sum = residual_sum
                        h_m = base_estimator
                    
                    num_processed_covers += 1
                else:
                    continue
                # Если внутренний цикл прерван из-за достижения лимита, прерываем и внешний цикл
                if num_processed_covers >= self.max_cov:
                    break
            
            best_gamma = best_gamma[0] * self.learning_rate, best_gamma[1] * self.learning_rate
            
            self.h.append(best_cover)
            self.gamma.append(best_gamma)
            self.covers.append(best_cover)
            self.key_objects.append(X[min_residual_idx])
            self.est_res.append(residuals[min_residual_idx])
            a_m, b_m = best_gamma
            y_hat_train += a_m * h_m + b_m

            # Предполагаем, что для валидационной выборки то же покрытие даст наименьшую ошибку
            # a_val, b_val, residual_sum, base_estimator = self.evaluate_coefs_and_residual_sum(best_cover, X_val, residuals_val, min_residual_idx_val)
            # y_hat_val += a_m * base_estimator + b_m 
            #TODO: имплементировать train/eval режим для регулировки коэффициентов базовой модели

            y_hat_val = self.predict(X_val)

            current_val_loss = np.mean((y_val - y_hat_val) ** 2)
            self.val_losses.append(current_val_loss)
            
            # Проверка на early stopping
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= self.patience:
                print(f"Early stopping triggered after {iteration + 1} iterations with validation loss {best_val_loss:.4f}")
                break
        
        return self


    def evaluate_coefs_and_residual_sum(self, cover, X, residuals, min_residual_idx):

        def loss(params):
            a, b = params
            return ((a * base_estimator + b - residuals)**2).mean()
        
        
        n = X.shape[1]

        h_mask_l = np.isin(np.arange(n), cover)
        h_mask_g = np.isin(np.arange(n, 2*n), cover)
        H_l = np.where((X[:, h_mask_l] >= X[min_residual_idx][h_mask_l]).all(axis=1), 1, 0)
        H_g = np.where((X[:, h_mask_g] <= X[min_residual_idx][h_mask_g]).all(axis=1), 1, 0)
        base_estimator = H_l * H_g

        B = base_estimator 
        r = residuals
        sum_r_B = np.sum(r * B)  # Сумма r_i * B(S_i)
        sum_B = np.sum(B)        # Сумма B(S_i)

        sum_r_B1 = np.sum(r * (B - 1))  # Сумма r_i * (B(S_i) - 1)
        sum_B1 = np.sum(B - 1)          # Сумма (B(S_i) - 1)

        if sum_B == 0 or sum_B1 == 0:
            raise ValueError("Division by zero in formula calculation for a_opt or b_opt.")

        a_opt = (sum_r_B / sum_B) - (sum_r_B1 / sum_B1)
        b_opt = sum_r_B1 / sum_B1

        params = [a_opt, b_opt]

        min_residual_sum = loss(params)

        return a_opt, b_opt, min_residual_sum, base_estimator

    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = H_l * H_g
            # print(self.gamma[i])
            a_i, b_i = self.gamma[i]
            y_pred += a_i * base_estimator + b_i

        return y_pred


class BaggingElementaryPredicates(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, n_estimators=10, sub_sample_size=1.0, sub_feature_size=1.0, bootstrap=True, bootstrap_features=True, n_jobs=None, random_state=None):
        self.base_estimator = base_estimator if base_estimator is not None else BoostingElementaryPredicates()
        self.base_params = base_estimator.get_params() if base_estimator is not None else {}
        self.n_estimators = n_estimators
        self.sub_sample_size = sub_sample_size
        self.sub_feature_size = sub_feature_size
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.random_state_ = np.random.RandomState(self.random_state)
        seeds = self.random_state_.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        self.estimators_ = []
        self.estimators_features_ = []

        if self.n_jobs is None:
            for seed in seeds:
                estimator, features = self._fit_estimator(X, y, seed)
                self.estimators_.append(estimator)
                self.estimators_features_.append(features)
        else:
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_estimator)(X, y, seed) for seed in seeds)
            self.estimators_, self.estimators_features_ = zip(*results)

        return self

    def _fit_estimator(self, X, y, seed):
        rng = np.random.default_rng(seed)
        samples = generate_bagging_indices(rng, X.shape[0], self.sub_sample_size, self.bootstrap)
        if self.bootstrap_features:
            features = generate_bagging_indices(rng, X.shape[1], self.sub_feature_size, self.bootstrap)
        else:
            features = np.arange(X.shape[1])

        estimator = self.base_estimator.__class__(**self.base_params)
        estimator.fit(X[samples][:, features], y[samples])
        
        return estimator, features

    def predict(self, X):
        predictions = np.mean([est.predict(X[:, features]) for est, features in zip(self.estimators_, self.estimators_features_)], axis=0)
        return predictions
