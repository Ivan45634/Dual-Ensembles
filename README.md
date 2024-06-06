# Описание проекта
Проект представляет реализацию алгоритмов, таких как бустинг/бэггинг над элементарными классификаторами, для прогнозирования непрерывных величин с высокой точностью.

# Описание используемых технологий, моделей, алгоритмов
- Язык программирования: Python3
- Фреймворки машинного обучения - scikit-learn, optuna
- Модель - BoostingElementaryPredicates, BaggingElementaryPredicates
- Визуализация - seaborn, matplotlib
- Алгоритмы регрессии, ансамблевые модели

# Описание используемых датасетов
Для обучения и тестирования моделей использовались датасеты из репозитория UCI и кастомный датасет для решения задачи прогнозирования химико-механических свойств металлоизделий:
https://archive.ics.uci.edu/datasets

# Запуск
Рекомендуется использовать python последней стабильной версии.
Перед запуском приложения необходимо установить требуемые для работы библиотеки, прописав в терминале: <br />
pip install -r requirements.txt <br />
В некоторых случаях замечено, что библиотеки не установливаются с первого раза, и тогда необходимо выполнить повторную установку библиотек, прописав команду выше или установив вручную с помощью pip install (conda install). <br />

# Описание структуры проекта
models - папка для хранения моделей
utils -  папка с кодом для работы с моделью и обработки данных
datasets - папка содержит используемые задачи в виде таблиц
exps - папка с кодом для проведения настройки гиперпараметров моделей, осуществления прогнозирования и оценивания качества работы
tests - папка с кодом для тестирования корректности исполнения кода моделей (TODO)
dual - папка с кодом алгоритма монотонной дуализации
librunc.dylib - содержит C++ бибилиотеку для использования RuncDualizer
runc.py - реализация класса, осуществляющего перечисление неприводимых покрытий для заданной матрицы сравнения "объект-признак"
own_forest.py - приведены некоторые методы для построения прогнозирующего ансамбля в виде случайного леса над кастомными моделями.
