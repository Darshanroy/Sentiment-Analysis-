# Import the training pipeline from the defined module
from pipelines.TrainingPipeline import training_pipeline

# Check if this script is being executed as the main program
if __name__ == '__main__':
    # If so, execute the training pipeline
    training_pipeline()

# End of script
