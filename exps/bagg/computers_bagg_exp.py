import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from models.boost_models import *
from utils.utilites import *
from utils.own_forest import *
from models.bagging_trees import BaggingElementaryPredicates


# Fetch the dataset
computer_hardware = fetch_ucirepo(id=29)

# Extract features and targets
X = computer_hardware.data.features
y = computer_hardware.data.targets

# Drop unnecessary columns
X = X.drop(columns='ModelName', axis=1)

# Preprocess the dataset
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, 'PRP')

dataset_name = 'Computer_Hardware_bagg'

# Generate combinations
combinations = [BoostingElementaryPredicatesv2(num_iter=i, m=j, max_cov=500, learning_rate=0.9)
                for i in np.linspace(300, 600, 5).astype(int)
                for j in range(2, 21)]

# Создаем словарь для отображения индексов в объекты
combinations_dict = {str(index): combo for index, combo in enumerate(combinations)}

# Обратное отображение для использования в RandomizedSearchCV
rev_combinations_dict = {v: k for k, v in combinations_dict.items()}

# Set up the parameter grid for RandomizedSearchCV
param_distributions = {
    'base_estimator': list(combinations_dict.values()),
    'n_estimators': sp_randint(10, 100),
    'sub_sample_size': uniform(0.3, 0.4),  # uniform distribution within [0.3, 0.7]
    'sub_feature_size': uniform(0.5, 0.5),  # uniform distribution within [0.5, 1.0]
    'bootstrap': [True],
    'bootstrap_features': [False],
    'n_jobs': [1],
}

# Initialize the BaggingElementaryPredicates model
model = BaggingElementaryPredicates()

# Initialize the RandomizedSearchCV with 10 iterations
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, scoring='r2', n_jobs=1, cv=3, random_state=42)

# Fit the model on training data
random_search.fit(X_train, y_train)

# Testing the model on test data
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
test_score = r2_score(y_test, y_pred)

# Save results to file
with open(f'results_{dataset_name}.txt', 'w') as f:
    f.write(f'Best trial: {random_search.best_params_}\n')
    f.write(f'Best score: {random_search.best_score_}\n')
    f.write(f'Test score: {test_score}\n')
