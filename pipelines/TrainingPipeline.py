from zenml import pipeline
from src.DataIngestion import data_ingestion
from src.DataProcessing import text_main_preprocess
from src.ModelTraining import train_model

@pipeline(enable_cache=True)
def training_pipeline():

    train_df = data_ingestion()
    X, y = text_main_preprocess(train_df)
    rfc = train_model(X, y)