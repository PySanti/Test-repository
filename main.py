from utils.basic_preprocess import basic_preprocess
import pandas as pd
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report



t1 = time.time()
[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"),"HeartDisease")
print(f"Tiempo de preprocesamiento {time.time() - t1}")



svc = SVC()

# Definición de los hiperparámetros a buscar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=10, n_jobs=3)

t1 = time.time()
grid_search.fit(df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"])
print(f"Tiempo de entrenamiento total {time.time() - t1}")

# Resultados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)


