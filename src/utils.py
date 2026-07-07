# src/utils.py
# -------------------------------------------------
# This module provides utility functions for the application.
# It includes functions for common tasks that can be reused across different modules.   
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# OS Library for interacting with the operating system
# SYS Library for system-specific parameters and functions
# Pickle for object serialization
# Sklearn's metrics for evaluating model performance
# Sklearn's model_selection for hyperparameter tuning
# src.exception for custom exception handling
# =============================================
import os
import pickle
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# =============================================
# save_object Function
# ---------------------------------------------
# This function saves a Python object to a specified file path using pickle serialization.
# It creates the necessary directories if they do not exist.
# =============================================
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# =============================================
# evaluate_models Function
# ---------------------------------------------
# Tunes each model with GridSearchCV (cv=3) and records, per model:
#   - cv_score   : mean cross-validated R2 (used for MODEL SELECTION, so we
#                  never pick a model by looking at the held-out test set)
#   - train_r2   : R2 on the training set (to spot over-fitting)
#   - test_r2    : R2 on the held-out test set (reported once, for the winner)
#   - best_params: the hyperparameters chosen by the grid search
# Each tuned model instance is left fitted in `models`, so the caller can reuse
# the winner directly without an extra refit.
# =============================================
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            report[name] = {
                "cv_score": gs.best_score_,
                "train_r2": r2_score(y_train, model.predict(X_train)),
                "test_r2": r2_score(y_test, model.predict(X_test)),
                "best_params": gs.best_params_,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

# =============================================
# load_object Function
# ---------------------------------------------
# This function loads a Python object from a specified file path using pickle deserialization.
# =============================================    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
