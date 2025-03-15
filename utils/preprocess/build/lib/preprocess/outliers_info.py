from sklearn.ensemble import IsolationForest
import pandas as pd

def outliers_info(X_data):
    """
        Funcion creada para detectar casos anomalos en conjuntos de datos
        y mejorar la calidad general de los datasets.
    """
    alg = IsolationForest(n_estimators=400, contamination="auto", max_samples=0.7, max_features=0.5, bootstrap=True)
    y_pred = pd.DataFrame(
        {"outliers_" : alg.fit_predict(X_data)},
        index=X_data.index
    )
    return pd.concat([X_data, y_pred], axis=1)

