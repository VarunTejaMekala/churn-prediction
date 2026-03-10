import os
import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


if __name__ == "__main__":
    try:
        # ----------------------------------------------------------------
        # Pipeline config
        # ----------------------------------------------------------------
        training_pipeline_config = TrainingPipelineConfig()

        # ----------------------------------------------------------------
        # Step 1: Data Ingestion
        # ----------------------------------------------------------------
        logging.info("=" * 60)
        logging.info("STEP 1: Data Ingestion")
        logging.info("=" * 60)
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion complete: {data_ingestion_artifact}")

        # ----------------------------------------------------------------
        # Step 2: Data Validation
        # ----------------------------------------------------------------
        logging.info("=" * 60)
        logging.info("STEP 2: Data Validation")
        logging.info("=" * 60)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation complete: {data_validation_artifact}")

        # ----------------------------------------------------------------
        # Step 3: Data Transformation
        # ----------------------------------------------------------------
        logging.info("=" * 60)
        logging.info("STEP 3: Data Transformation")
        logging.info("=" * 60)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation complete: {data_transformation_artifact}")

        # ----------------------------------------------------------------
        # Step 4: Model Training
        # ----------------------------------------------------------------
        logging.info("=" * 60)
        logging.info("STEP 4: Model Training")          # FIX: was "sstared" (typo)
        logging.info("=" * 60)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model Training complete: {model_trainer_artifact}")

        logging.info("=" * 60)
        logging.info("PIPELINE COMPLETE ✓")
        logging.info("=" * 60)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
