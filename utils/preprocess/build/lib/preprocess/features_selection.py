import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin




def features_selection(X_data, Y_data, classification):
    """
        Funcion creada para automatizar el proceso de seleccion
        de caracteristicas usando random forest

        Retorna un dataframe con las caracteristicas y si importancia
    """
    estimators = 250
    alg = RandomForestRegressor(n_estimators=estimators) if not classification else  RandomForestClassifier(n_estimators=estimators)
    alg.fit(X_data, Y_data)
    importances_df = pd.DataFrame({
        'Feature': X_data.columns  , 
        'Importance': alg.feature_importances_  })  
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    return importances_df

class BestFeatures(BaseEstimator, TransformerMixin):
    """
        Transformador creado para reducir dataframes
        a su version mas relevante
    """
    def __init__(self, threshold : float, classification : bool) -> None:
        self.threshold = threshold
        self.classification = classification
        self.best_features_ = []
    def fit(self, X_data, Y_data):
        select = features_selection(X_data, Y_data, self.classification)
        self.best_features_ = select[select["Importance"] >= self.threshold]["Feature"].tolist()
        return self
    def transform(self, X_data, Y_data=None):
        return X_data[self.best_features_]

