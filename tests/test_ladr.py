import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from models.ladr import LADRegression

@pytest.fixture
def sample_data():
    """Набор данных для тестирования."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0.1, 0.4, 0.35, 0.8])
    return X, y

def test_fit_predict(sample_data):
    """Тестирование методов fit и predict."""
    X, y = sample_data
    model = LADRegression(num_iter=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray), "Predict должен возвращать np.ndarray"
    assert predictions.shape == y.shape, "Форма предсказаний должна совпадать с формой y"

def test_discretize_responses(sample_data):
    """Проверка дискретизации ответов."""
    _, y = sample_data
    model = LADRegression(discretization_method='equal_width')
    thresholds = model._discretize_responses(y)
    assert len(thresholds) > 0, "Должен возвращать список порогов"
    assert all(isinstance(t, (int, float)) for t in thresholds), "Пороги должны быть числами"

def test_create_binary_labels(sample_data):
    """Тестирование бинарных меток."""
    _, y = sample_data
    model = LADRegression()
    threshold = 0.2
    binary_labels = model._create_binary_labels(y, threshold)
    assert len(binary_labels) == len(y), "Длина бинарных меток должна совпадать с длиной y"
    assert set(binary_labels).issubset({0, 1}), "Метки должны быть 0 или 1"

@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.7])
def test_binary_labels_threshold(sample_data, threshold):
    """тест для различных порогов."""
    _, y = sample_data
    model = LADRegression()
    binary_labels = model._create_binary_labels(y, threshold)
    expected = np.where(y > threshold, 1, 0)
    np.testing.assert_array_equal(binary_labels, expected, "Бинарные метки некорректно рассчитаны")

# TODO: Дополнительные тесты для оставшихся методов и их крайних случаев

def main():
    # -v для подробного вывода.
    pytest.main(['-v'])

if __name__ == "__main__":
    main()