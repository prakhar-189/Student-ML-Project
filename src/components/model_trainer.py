# src/Components/model_trainer.py
# =================================================
# This module handles model training for the machine learning pipeline.
# It includes methods for training various regression models, evaluating their performance,
# and saving the best-performing model to a specified file path.
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# OS Library for interacting with the operating system
# SYS Library for system-specific parameters and functions
# Dataclasses for creating data classes to manage configuration
# Sklearn's ensemble methods for various regression algorithms like RandomForest, AdaBoost, GradientBoosting
# Sklearn's linear_model for Linear Regression
# Sklearn's metrics for evaluating model performance
# Sklearn's neighbors for K-Nearest Neighbors regression
# Sklearn's tree for Decision Tree regression
# src.exception for custom exception handling
# src.logger for logging information during execution
# src.utils for utility functions like saving objects and evaluating models
# =============================================
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# MLflow is an optional training-time dependency. Import it defensively so the
# deployed prediction path (which never trains) does not require it installed.
try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# =============================================
# ModelTrainerConfig Class
# ---------------------------------------------
# This section defines the configuration for model training and the ModelTrainer class.
# The ModelTrainer class contains methods to train and evaluate multiple regression models.
# =============================================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


# =============================================
# ModelTrainer Class
# ---------------------------------------------
# This class contains methods to train and evaluate multiple regression models.
# It selects the best-performing model based on R2 score and saves it to a specified file path.
# =============================================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # X_train and y_train are the training data and labels, while X_test and y_test are the testing data and labels.
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # models is a dictionary that contains the names of the regression models as keys and their corresponding instantiated objects as values.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # params is a dictionary that contains the hyperparameters for each regression model.
            params={
                # Decision Tree is used to select the best method to train the model 
                # and to get the best hyperparameters for the model.
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },

                # Random Forest trains many decision trees (using default squared_error)
                # and averages their outputs to improve performance and control over-fitting.
                # The n_estimators parameter specifies the number of trees in the forest.
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },

                # Gradient Boosting builds an additive model in a sequential manner, 
                # optimizing for a loss function (default friedman_mse) and using decision trees as weak learners.
                # The learning_rate parameter controls the error contribution of each tree,
                # while subsample specifies the fraction of samples to be used for fitting the individual base learners.
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                # Linear Regression is a simple linear approach to modeling the relationship between a dependent variable and one or more independent variables.
                # Creates an default model with no hyperparameters to tune, as it is a straightforward linear approach.
                "Linear Regression":{},

                # XGBRegressor(eXtreme Gradient Boosting) is an optimized gradient boosting library that does same thing as Gradient Boosting but is designed for speed and performance.
                # We use both Gradient Boosting and XGBRegressor to compare their performance and select the best one for our dataset.
                # Same as XGBRegressor, we also use CatBoosting Regressor and AdaBoost Regressor to compare their performance and select the best one for our dataset.
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            # Tune every model and collect cv_score / train_r2 / test_r2 / best_params.
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params,
            )

            self._log_to_mlflow(model_report)

            # Select the winner by cross-validated score -- NOT by the test set,
            # which must stay untouched until the final, one-time evaluation.
            best_model_name = max(model_report, key=lambda name: model_report[name]["cv_score"])
            best_info = model_report[best_model_name]
            best_model = models[best_model_name]  # already fitted inside evaluate_models

            if best_info["test_r2"] < 0.6:
                raise CustomException("No best model found with R2 score >= 0.6", sys)

            logging.info(
                f"Best model: {best_model_name} "
                f"(cv_r2={best_info['cv_score']:.4f}, test_r2={best_info['test_r2']:.4f})"
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Report the winner's held-out test R2 (recomputed here for clarity).
            return r2_score(y_test, best_model.predict(X_test))

        except Exception as e:
            raise CustomException(e, sys)

    # =============================================
    # _log_to_mlflow
    # ---------------------------------------------
    # Logs one MLflow run per model (params + cv/train/test R2) so experiments
    # are tracked and comparable. No-op if MLflow isn't installed.
    # =============================================
    def _log_to_mlflow(self, model_report):
        if not _MLFLOW_AVAILABLE:
            logging.info("MLflow not installed; skipping experiment tracking.")
            return
        try:
            mlflow.set_experiment("student-math-score")
            for name, info in model_report.items():
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model", name)
                    mlflow.log_params(info["best_params"])
                    mlflow.log_metric("cv_r2", info["cv_score"])
                    mlflow.log_metric("train_r2", info["train_r2"])
                    mlflow.log_metric("test_r2", info["test_r2"])
        except Exception:
            # Tracking must never break training.
            logging.exception("MLflow logging failed; continuing without it.")