# src/Pipeline/predict_pipeline.py
# =================================================
# This module handles the prediction pipeline for the machine learning model.
# It defines the PredictPipeline class, which loads the trained model and preprocessor, and makes predictions based on the input features.
# It also defines the CustomData class, which is used to structure the input data for prediction and convert it into a DataFrame format suitable for the model.
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# SYS Library for system-specific parameters and functions
# Pandas for data manipulation and analysis
# src.exception for custom exception handling
# src.utils for utility functions like loading objects
# =============================================
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


# =============================================
# PredictPipeline Class
# ---------------------------------------------
# This class is responsible for loading the trained model and preprocessor, and making predictions based on the input features.
# The predict method takes the input features, applies the preprocessor, and returns the predicted values
# =============================================
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(base_dir, "artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)


# =============================================
# CustomData Class
# ---------------------------------------------
# This class is used to structure the input data for prediction.
# It takes the input features as parameters and has a method to convert them into a DataFrame format suitable for the model.
# =============================================
class CustomData:
    def __init__(self,
                    gender: str,
                    race_ethnicity : str,
                    parental_level_of_education : str,
                    lunch : str,
                    test_preparation_course : str,
                    writing_score : int,
                    reading_score : int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score" : [self.writing_score],
                "reading_score" : [self.reading_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)