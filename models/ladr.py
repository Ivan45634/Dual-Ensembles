import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from models.runc import RuncDualizer
from sklearn.tree import DecisionTreeClassifier

class LADRegression(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter=10, n_clusters=10, discretization_method='kmeans'):
        self.num_iter = num_iter # число итераций
        self.discretization_method = discretization_method
        self.n_clusters = n_clusters
        self.elemental_clauses = []  # эл.кл.
        self.gamma = []  # веса
        
    def fit(self, X, y):
        thresholds = self._discretize_responses(y)
        
        y_hat = np.zeros(len(y))
        
        for iter_num in range(self.num_iter):
            for threshold in thresholds:
                binary_labels = self._create_binary_labels(y, threshold)
                
                elemental_clauses = self._generate_elemental_clauses(X, binary_labels)
                
                # elemental_clauses = self._preprocess_clauses(elemental_clauses)

                gamma = self._compute_optimal_coefficient(X, y, y_hat, elemental_clauses)
                
                self.elemental_clauses.append(elemental_clauses)
                self.gamma.append(gamma)
                
                y_hat += gamma * self._apply_clauses(X, elemental_clauses)
                
        return self
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for clauses, gamma in zip(self.elemental_clauses, self.gamma):
            y_pred += gamma * self._apply_clauses(X, clauses)
        return y_pred
    
    def _discretize_responses(self, y):
        """
        Дискретизация ответов.
        
        В зависимости от выбранного метода дискретизации (например, равные интервалы), 
        метод определяет пороговые значения для разбиения непрерывных ответов на категории, 
        что позволяет более эффективно работать с данными.
        
        TODO: Реализовать поддержку различных методов дискретизации (K-means, %STD, Quantile)
        """
        if self.discretization_method == 'equal_width':
            max_val = np.max(y)
            min_val = np.min(y)
            range_val = max_val - min_val
            interval_width = range_val / 10  # creating 10 intervals
            thresholds = [min_val + i * interval_width for i in range(1, 10)]
            return thresholds
        elif self.discretization_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = kmeans.fit_predict(y.reshape(-1, 1))
            
            # Сортировка индексов вместе с метками для обеспечения последовательности
            sorted_indices = np.argsort(labels)
            
            # Вычисление порогов как первое значение следующего кластера
            thresholds = []
            current_label = labels[sorted_indices[0]]
            for i in range(1, len(labels)):
                if labels[sorted_indices[i]] != current_label:
                    thresholds.append(y[sorted_indices[i]])
                    current_label = labels[sorted_indices[i]]
                
            return thresholds

        return []
        
    
    def _create_binary_labels(self, y, threshold):
        """
        Создание бинарных меток на основе порогового значения.
        
        Все значения выше порога получают метку 1, остальные - 0. 
        Это позволяет переформулировать задачу регрессии в 
        формат бинарной классификации для каждого порога.
    
        TODO: Изучить влияние различных способов бинаризации на качество модели.
        """
        return np.where(y > threshold, 1, 0)
    
    def _generate_elemental_clauses(self, X, binary_labels):
        """
        Генерация эл.кл. на основе бинарных меток.
        
        Использует метод дуализации задачи для генерации эл.кл., которые 
        описывают классификацию для текущего набора бинарных меток. 
        
        TODO: Оптимизировать поиск клауз для ускорения процедуры.
        """
        runc = RuncDualizer()
        num_features = X.shape[1]
        
        for idx, label in enumerate(binary_labels):
            if label == 1:
                columns_with_ones = list(np.where(X[idx] == 1)[0])
                runc.add_input_row(columns_with_ones)
        
        elemental_clauses = runc.enumerate_covers()
        print(elemental_clauses)
        return elemental_clauses
    
    def _generate_elemental_clauses(self, X, binary_labels):
        clf = DecisionTreeClassifier(max_depth=1)  
        clf.fit(X, binary_labels)
        return clf

    
    def _preprocess_clauses(self, elemental_clauses):
        """
        Предобработка эл.кл.: удаление дубликатов и коррелирующих предикатов.
        
        Этот шаг необходим для устранения избыточности и 
        мультиколлинеарности среди клауз, что улучшает обобщающую способность модели.
        
        TODO: Расширить для более сложного анализа корреляций между эл.кл.
        """
        unique_clauses = []
        for clause in elemental_clauses:
            if clause not in unique_clauses:
                unique_clauses.append(clause)
        return unique_clauses
    
    def _compute_optimal_coefficient(self, X, y, y_hat, elemental_clauses):
        """
        Вычисление оптимальных коэффициентов для эл.кл.
        
        Метод рассчитывает корреляцию между текущими ошибками и применением эл.кл.,
        используя это в качестве оценки коэффициента перед клаузой.
        
        TODO: Использовать более сложные методы оптимизации для повышения точности.
        """
        errors = y - y_hat 
        applied_clauses = self._apply_clauses(X, elemental_clauses) 
        gamma = np.corrcoef(errors, applied_clauses)[0, 1]
        return gamma
    
    def _apply_clauses(self, X, elemental_clauses):
        """
        Применение эл.кл. к данным.
        
        Каждая клауза проверяет выполнение определенного условия в данных. 
        Полученные сигналы от каждой клаузы комбинируются для получения 
        итоговых предикций модели.
        
        TODO: Рассмотреть более гибкие подходы к применению эл.кл.
        """
        signals = np.ones(X.shape[0])
        for clause in elemental_clauses:
            if len(clause) != 2:
                raise ValueError(f"Expected clause to have 2 elements, but got {len(clause)}: {clause}")
        
        for clause in elemental_clauses:
            feature_index, threshold = clause
            signals *= X[:, feature_index] > threshold 

        return signals
