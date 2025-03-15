from numpy import diff
from pandas.core.common import random_state
from utils.basic_preprocess import basic_preprocess
import pandas as pd
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report





t1 = time.time()
[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"),"HeartDisease")
print(f"Tiempo de preprocesamiento : {time.time()-t1}")


opciones = {
    94: {"max_depth":None, "min_samples_leaf":1,"min_samples_split":2, "n_estimators":400},
    46: {"max_depth":None, "min_samples_leaf":1,"min_samples_split":2, "n_estimators":200},
    72: {"max_depth":None, "min_samples_leaf":1,"min_samples_split":2, "n_estimators":300},
}

xs = []

for t,hp in opciones.items():
    print(f"Hiperparametros a estudiar : {hp}")
    print(f"Segundos en pc de santiago: {t}")
    t1 = time.time()
    RandomForestClassifier(random_state=42, **hp).fit(df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"])
    t2 = time.time()
    print(f"Segundos en pc de samuel {t2-t1}")
    xs.append(t/(t2-t1))

print(f"La pc de samuel es aproximadamente {sum(xs)/3} veces mas rapida que la de santiago")

