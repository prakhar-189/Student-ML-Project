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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


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
            
            # model_report is a dictionary that contains the R2 scores of each regression model on the test dataset.
            # 'params = params' is passed to the evaluate_models function to perform hyperparameter tuning for each model using GridSearchCV or RandomizedSearchCV.
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            best_model.fit(X_train,y_train)

            if best_model_score<0.6:
                raise CustomException("No best model found with R2 score >= 0.6",sys)
            logging.info(f"Best found model on both training and testing dataset")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
                
        except Exception as e:
            raise CustomException(e,sys)