import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """
        Implementa el SimpleImputer pero, a diferencia del mismo, retorna
        dataframes y no matrices de numpy
    """
    def __init__(self, strategy : str, attributes) -> None:
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        self.attributes = attributes
    def fit(self, X, y=None):
        self.imputer_.fit(X[self.attributes], y)
        return self

    def transform(self, X, y=None):
        transformed_data = self.imputer_.transform(X.copy()[self.attributes])
        transformed_data = pd.DataFrame(transformed_data, columns=self.attributes, index=X[self.attributes].index)
        X[self.attributes] = transformed_data
        return X
