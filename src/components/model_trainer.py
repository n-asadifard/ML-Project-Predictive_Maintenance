import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve
)

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    get_candidates, 
    get_parameter_distributions, 
    evaluate_candidates, 
    tune_top_models, 
    save_object
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting target feature from train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
            cv_splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

            candidates = get_candidates(class_ratio)
            fitted_pipelines = {}

            # 1. Evaluate candidates (idempotency handles any redundant transformer calls safely)
            logging.info("Starting training and cross-validation loop for all candidates")
            results_df = evaluate_candidates(candidates, fitted_pipelines, X_train, y_train, cv_splits, scoring)
            
            # 2. Extract top 2 models and tune hyperparameters
            top_2_models = results_df["Model"].head(2).tolist()
            logging.info(f"Top 2 models selected for tuning: {top_2_models}")

            param_distributions = get_parameter_distributions()
            final_pipeline, model_name = tune_top_models(top_2_models, fitted_pipelines, param_distributions, X_train, y_train, cv_splits)

            # 3. Optimize decision threshold using Out-Of-Fold predictions
            logging.info("Optimizing decision threshold using OOF predictions")
            oof_proba = cross_val_predict(final_pipeline, X_train, y_train, cv=3, method="predict_proba", n_jobs=-1)[:, 1]
            oof_precisions, oof_recalls, oof_thresholds = precision_recall_curve(y_train, oof_proba)
            oof_f1_scores = 2 * oof_precisions * oof_recalls / (oof_precisions + oof_recalls + 1e-12)
            best_idx = np.nanargmax(oof_f1_scores[:-1])
            best_threshold = oof_thresholds[best_idx]

            # 4. Final fit on training data and save model artifact
            final_pipeline.fit(X_train, y_train)
            logging.info(f"Saving final model pipeline to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_pipeline
            )

            # 5. Evaluate on test data using robust classification metrics
            y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= best_threshold).astype(int)

            test_metrics = {
                "model": model_name, 
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, zero_division=0),
                "recall": recall_score(y_test, y_test_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_test_proba),
                "average_precision": average_precision_score(y_test, y_test_proba),
                "optimal_threshold": best_threshold
            }

            logging.info(f"Final Test Metrics: {test_metrics}")
            return test_metrics

        except Exception as e:
            raise CustomException(e, sys)