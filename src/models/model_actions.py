import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error)


def train_model(
    features: pd.DataFrame, target: pd.Series, training_params: dict):
    """
    Trains the model.
    :param features: features to train on
    :param target: target labels to train on
    :param training_params: training parameters

    :return: save model via pickle

    """
    if training_params['model_type'] == 'XGBRegressor':
        model = XGBRegressor(
            n_estimators=training_params['n_estimators'],
            min_child_weight=training_params['min_child_weight'],
            max_depth=training_params['max_depth'],
            learning_rate=training_params['learning_rate'],
            random_state=training_params['model_random_state'],
            n_iter=20,
            n_jobs=-1,
            verbose=1
        )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_model(model, features: pd.DataFrame) -> [np.ndarray, np.ndarray]:
    """
    Makes predictions using model.
    :param model: the model to predict with
    :param features: the features to predict on

    :return: prediction

    """
    prediction=model.predict(features)
    return prediction


def evaluate_model(target: pd.Series, predicts: np.array) -> dict:
    """
    Evaluates model predictions and returns the metrics.
    :param target: actual target labels
    :param predicts: pipeline hard predictions

    :return: a dict of metrics in format {'metric_name': value}

    """
    r2 = round(r2_score(target, predicts), 3)
    mae = round(mean_absolute_error(target, predicts), 3)
    mse = round(mean_squared_error(target, predicts), 3)

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse
    }
