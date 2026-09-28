from src.textSummariser.logging import logger
from src.textSummariser.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline 
from src.textSummariser.pipeline.stage_2_data_transformation_pipeline import DataTransformationPipeline
from src.textSummariser.pipeline.stage_3_model_trainer import ModelTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} started")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f"stage {STAGE_NAME} started")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f"stage {STAGE_NAME} started")
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.initiate_model_trainer()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

