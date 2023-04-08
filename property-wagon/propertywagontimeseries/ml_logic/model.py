

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from neuralprophet import NeuralProphet
import numpy as np


def initialize_model():
    model = NeuralProphet(seasonality_mode="multiplicative", learning_rate=0.1,n_lags=5,n_forecasts=60)

    return model
