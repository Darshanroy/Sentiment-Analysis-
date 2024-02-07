from zenml import step
import pandas as pd
from typing import Tuple

@step
def data_ingestion() -> pd.DataFrame:
    """
    This function reads the train data, test data, merges both, and gives the final output.
    
    Returns:
        DataFrame: Merged DataFrame containing train and test data.
    """
    # Read train and test data
    df_train = pd.read_csv("data/train.txt", delimiter=';', names=['text', 'label'])
    df_val = pd.read_csv("data/val.txt", delimiter=';', names=['text', 'label'])

    # Merge train and test data
    df = pd.concat([df_train, df_val])
    df.reset_index(inplace=True, drop=True)

    return df
