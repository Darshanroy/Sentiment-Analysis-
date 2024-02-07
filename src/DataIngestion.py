from zenml import step
import pandas as pd
import os
from typing import Tuple

# path = '../data'
# os.chdir(path=path)




@step
def data_ingestion() -> pd.DataFrame:
    # Tuple[pd.DataFrame, pd.DataFrame]
    """
    This function reads the train data , test data --> merges both and gives final output
    args : None
    returns : DataFrame
    
    """

    df_train = pd.read_csv("data/train.txt",delimiter=';',names=['text','label'])
    df_val = pd.read_csv("data/val.txt",delimiter=';',names=['text','label'])
    # test_df = pd.read_csv('data/test.txt',delimiter=';',names=['text','label'])

    df = pd.concat([df_train,df_val])
    df.reset_index(inplace=True,drop=True)
    # ,test_df
    return df 

