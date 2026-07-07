# src/components/data_ingestion.py
# =================================================
# This module handles data ingestion for the machine learning pipeline.
# It reads the dataset, splits it into training and testing sets.
# It also saves the raw, training, and testing data to specified file paths.
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# OS Library for interacting with the operating system
# SYS Library for system-specific parameters and functions
# src.exception for custom exception handling
# src.logger for logging information during execution
# Pandas for data manipulation and analysis
# Sklearn's train_test_split for splitting the dataset into training and testing sets
# Dataclasses for creating data classes to manage configuration
# src.components.data_transformation for data transformation configuration and class
# src.components.model_trainer for model training configuration and class
# =============================================
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


# =============================================
# @Data Ingestion Configuration and Class
# ---------------------------------------------
# This section defines the configuration for data ingestion and the DataIngestion class.
# The DataIngestion class contains methods to read the dataset, split it, and save the files.
# =============================================
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:            
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, "Notebook", "Dataset", "Student.csv")
            df = pd.read_csv(data_path)

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
