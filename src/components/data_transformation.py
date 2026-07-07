# src.components.data_transformation.py
# =================================================
# This module handles data transformation for the machine learning pipeline.
# It includes methods for preprocessing the data, such as handling missing values.
# It also prepares the data for model training by applying necessary transformations.
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# OS Library for interacting with the operating system
# SYS Library for system-specific parameters and functions
# dataclasses for creating data classes to manage configuration
# Numpy for numerical operations
# Pandas for data manipulation and analysis
# Sklearn's ColumnTransformer for applying different transformations to different columns
# Sklearn's SimpleImputer for handling missing values
# Sklearn's Pipeline for creating a sequence of data transformations
# Sklearn's OneHotEncoder for converting categorical variables into a format that can be provided to ML algorithms
# Sklearn's StandardScaler for feature scaling
# src.exception for custom exception handling
# src.logger for logging information during execution
# =============================================
import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# =============================================
# @Data Transformation Configuration and Class
# ---------------------------------------------
# This section defines the configuration for data transformation and the DataTransformation class.
# The DataTransformation class contains methods to create a data transformer object.
# =============================================
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


# =============================================
# get_data_transformer_object Function
# ---------------------------------------------
# This function creates and returns a data transformer object.
# It defines pipelines for numerical and categorical data preprocessing.
# The numerical pipeline handles missing values and scales the data.
# The categorical pipeline handles missing values, encodes categorical variables, and scales the data.
# Finally, it combines both pipelines into a ColumnTransformer.
# =============================================
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown = "ignore")),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


# =============================================
# initiate_data_transformation Function
# ---------------------------------------------
# This function initiates the data transformation process.
# It reads the training and testing data, applies the preprocessing object,
# and returns the transformed training and testing arrays along with the preprocessor file path.
# =============================================
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)