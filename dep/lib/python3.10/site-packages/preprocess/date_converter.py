import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class DateConverter(BaseEstimator, TransformerMixin):
    """
        Tranformador creado para convertir todas las columnas de tipo date a float.

        Recibe la lista de columnas de date en el constructor.

        Considerar que para lograr convertir a float todos los elementos (incluyendo nan),
        se debio convertir nans a 0 para despues convertir 0 a nans de nuevo.
    """
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):  
        X = X.copy()  
        def converter(col):  
            dt_col = pd.to_datetime(col, errors='coerce')  
            float_col = dt_col.apply(lambda x: x.timestamp() if pd.notna(x) else 0)
            return float_col.replace(0, np.nan)
        for col in self.attributes:
            X[col] = converter(X[col])
            X[col] = X[col].astype(float)
        return X

