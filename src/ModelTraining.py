from zenml import step
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
import pandas as pd
import mlflow
import optuna
from abc import ABC, abstractmethod
from config import ModelNameConfig
import logging
import pickle
# Import your models here
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import scipy
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV


@step
def train_model(X: scipy.sparse._csr.csr_matrix, y: pd.Series) -> RegressorMixin:
    """
    This part of the code is for training the model

    args : text data which is converted using Count vectorizer & y as a label
    returns : it returns the trained model
    """

    parameters = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [5, 10, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
    }

    grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(X, y)
    var = grid_search.best_params_

    rfc = RandomForestClassifier(**grid_search.best_params_)

    # rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
    #                              max_depth=grid_search.best_params_['max_depth'],
    #                              n_estimators=grid_search.best_params_['n_estimators'],
    #                              min_samples_split=grid_search.best_params_['min_samples_split'],
    #                              min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    #                              bootstrap=grid_search.best_params_['bootstrap'])

    rfc.fit(X, y)
    # Specify the file path where you want to save the pickle file
    file_path = "random_forest_model.pkl"

    # Save the `rfc` object to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(rfc, f)

    return rfc
