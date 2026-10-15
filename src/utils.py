import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.utils.validation import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



# def build_pipeline(model=None, scale=False):
#     """
#     Builds a pipeline containing feature engineering, 
#     an optional conditional scaler, and an optional classifier model.
#     """
#     steps = [("feature_engineering", SensorFeatureEngineering())]
    
#     if scale:
#         steps.append(("scaler", StandardScaler()))
        
#     if model is not None:
#         steps.append(("classifier", model))
        
#     return Pipeline(steps)



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

    


def get_candidates(class_ratio):
    """Defines and returns all candidate models and their scaling requirements."""   
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            True,  # Requires scaling
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=42, n_jobs=-1),
            False, # No scaling needed
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=300, learning_rate=0.09, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=class_ratio, random_state=42,
                            n_jobs=-1, eval_metric="logloss"),
            False, # No scaling needed
        ),
        "CatBoost": (
            CatBoostClassifier(iterations=300, learning_rate=0.09, depth=6,
                                auto_class_weights='Balanced', random_seed=42, verbose=0),
            False, # No scaling needed
        ),
    }




def get_parameter_distributions():
    """Returns comprehensive hyperparameter search spaces using randint and uniform for all models."""    
    return {
        "Logistic Regression": {
            "classifier__C": uniform(0.01, 10.0),
            "classifier__solver": ["lbfgs", "saga"]
        },
        "Random Forest": {
            "classifier__n_estimators": randint(100, 600),
            "classifier__max_depth": [None, 5, 10, 20, 30],
            "classifier__min_samples_split": randint(2, 11),
            "classifier__min_samples_leaf": randint(1, 10),
            "classifier__max_features": ["sqrt", "log2", None]
        },
        "XGBoost": {
            "classifier__n_estimators": randint(100, 600),
            "classifier__max_depth": randint(3, 10),
            "classifier__learning_rate": uniform(0.01, 0.2),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4)
        },
        "CatBoost": {
            "classifier__iterations": randint(100, 600),
            "classifier__depth": randint(4, 10),
            "classifier__learning_rate": uniform(0.01, 0.2)
        }
    }    


def evaluate_candidates(candidates, fitted_pipelines, X_train, y_train, cv_splits, scoring):
    """Trains and cross-validates all candidates, returning a sorted performance DataFrame."""
    try:
        comparison_rows = []
        for name, (model, scale) in candidates.items():
            pipe = build_pipeline(model, scale)  # Assumes build_pipeline is defined globally or imported
            fitted_pipelines[name] = pipe
            cv_results = cross_validate(pipe, X_train, y_train, cv=cv_splits, scoring=scoring, n_jobs=-1)
            row = {"Model": name}
            row.update({metric: cv_results[f"test_{metric}"].mean() for metric in scoring})
            comparison_rows.append(row)            
        results_df = pd.DataFrame(comparison_rows).sort_values("average_precision", ascending=False)
        return results_df

    except Exception as e:
            raise CustomException(e, sys)



def tune_top_models(top_models, fitted_pipelines, param_distributions, X_train, y_train, cv_splits):
    """Performs RandomizedSearchCV on the dynamically selected top-performing models."""
    try:
        tuning_rows = []
        tuned_estimators = {}
        for name in top_models:
            search = RandomizedSearchCV(
                fitted_pipelines[name],
                param_distributions=param_distributions[name],
                n_iter=20,
                scoring="average_precision",
                cv=cv_splits,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            tuned_estimators[name] = search.best_estimator_
            tuning_rows.append({
                "Model": name,
                "Best CV average precision": search.best_score_,
                "Best params": search.best_params_,
            })
        tuning_df = pd.DataFrame(tuning_rows).sort_values("Best CV average precision", ascending=False)
        best_model_name = tuning_df.iloc[0]["Model"]
        return tuned_estimators[best_model_name]

    except Exception as e:
        raise CustomException(e, sys)
    


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)