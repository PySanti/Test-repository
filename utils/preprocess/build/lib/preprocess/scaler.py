import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    """
        Recibe las columnas numéricas del dataframe. Escala sus valores usando RobustScaler.
    """
    def __init__(self, attributes):
        self.attributes = attributes
        self.scaler_ = RobustScaler()  # Inicializa el escalador

    def fit(self, X, y=None):
        # Ajusta el escalador solo con los datos de entrenamiento
        scale_cols = X[self.attributes]
        self.scaler_.fit(scale_cols)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        # Usa el escalador ya ajustado para transformar los datos
        X_scaled = self.scaler_.transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy
