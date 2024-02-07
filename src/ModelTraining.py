import tensorflow as tf
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



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow



@step
def train_model(X_train: tf.Tensor, y_train: pd.Series) -> RegressorMixin:
    """
    This part of the code is for training the model

    args : text data which is converted using Count vectorizer & y as a label
    returns : it returns the trained model
    """

    # # Hyperparameters
    # embedding_dim = 100  # Adjust as needed based on your desired embedding dimension
    # max_seq_length = X_train_dense.shape[1]
    # Hyperparameters
    embedding_dim = 300  # Adjust as needed based on your desired embedding dimension
    max_seq_length = X_train.shape[1]  # Maximum sequence length based on the number of features

    # Create LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=max_seq_length))
    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model summary
    print(model.summary())

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)  # Adjust epochs and batch_size as needed

    # Specify the file path where you want to save the pickle file
    file_path = "LSTM.pkl"

    # Save the `rfc` object to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    return model
