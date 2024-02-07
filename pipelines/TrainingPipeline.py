import pickle  # Import the pickle module

from zenml import pipeline
from src.DataIngestion import data_ingestion
from src.DataProcessing import text_main_preprocess
from src.ModelTraining import train_model

# Define a ZenML pipeline for training
@pipeline(enable_cache=True)
def training_pipeline():
    # Step 1: Data Ingestion
    train_df = data_ingestion()  # Ingest data from source
    # Step 2: Text Data Preprocessing
    X, y = text_main_preprocess(train_df)  # Preprocess text data
    # Step 3: Model Training
    rfc = train_model(X, y)  # Train the model

    # Save the trained model using pickle
    with open("LSTM.pkl", "wb") as f:
        pickle.dump(rfc, f)

# End of pipeline definition
