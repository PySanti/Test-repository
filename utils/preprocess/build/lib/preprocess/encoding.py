import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    """
        Busca todas las columnas no numericas del dataframe (.dtypes(include=["object"]))
        Y las codifica usando el metodo OneHotEncoder de sk
    """
    def __init__(self, attributes):
        self.oh_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.columns_ = None
        self.attributes = attributes
    def fit(self, X, y=None):
        X_cat = X[self.attributes]
        self.columns_ = pd.get_dummies(X_cat).columns
        self.oh_.fit(X_cat)
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy[self.attributes]
        X_cat_oh = self.oh_.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh, 
                                columns=self.columns_, 
                                index=X_copy.index)
        X_copy.drop(self.attributes, axis=1, inplace=True)
        return pd.concat([X_copy, X_cat_oh], axis=1)

class FrecuencyEncoding(BaseEstimator, TransformerMixin):
    """
        Busca las columnas categoricas y las reemplaza
        utilizando Frecuency Encoding
    """
    def __init__(self, attributes):
        self.frequencies_ = {}
        self.attributes = attributes

    def fit(self, X, y=None):
        for col in self.attributes:
            self.frequencies_[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        df_encoded = X.copy()
        for col, freq in self.frequencies_.items():
            df_encoded[col] = df_encoded[col].map(freq).fillna(0)
        return df_encoded
