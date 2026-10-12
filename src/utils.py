from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import sys

import pandas as pd

import pickle

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

class SensorFeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds domain-informed interaction features to the raw sensor readings."""

    def __init__(self, eps=1e-5):
        self.eps = eps

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X):        
        X = X.copy()        
        X["Environmental_Stress"] = X["VolatileOrgComp"] * X["AirQuality"]
        X["VOC-to-USS"] = X["VolatileOrgComp"] / (X["UltrSensData"] + self.eps)
        X["log_footfall"] = np.log1p(X["footfall"])
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ["Environmental_Stress",
                                 "VOC-to-USS", "log_footfall"])

    
def build_pipeline(model, scale):
    steps = [("feature_engineering", SensorFeatureEngineering())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", model))
    return Pipeline(steps)