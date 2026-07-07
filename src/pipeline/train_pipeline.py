# src/Pipeline/train_pipeline.py
# =================================================
# This module handles the training pipeline for the machine learning model.
# It orchestrates the data ingestion, data transformation, and model training processes.
# It imports the necessary components for each step of the pipeline and executes them in sequence.
# Creates the pickled model and preprocessor files in the artifacts directory after training is completed.
# =================================================


# =============================================
# Importing the Necessary Libraries and Components
# ---------------------------------------------
# Importing the DataIngestion component for data ingestion
# Importing the DataTransformation component for data transformation 
# Importing the ModelTrainer component for model training
# =============================================
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Training completed successfully. R2 Score: {r2}")
