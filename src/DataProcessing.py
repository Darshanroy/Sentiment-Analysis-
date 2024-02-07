import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from zenml import step
from typing import Tuple, Union
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

# Initialize WordNetLemmatizer and CountVectorizer
lm = WordNetLemmatizer()
cv = CountVectorizer(ngram_range=(1, 2))

def custom_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode sentiment labels as binary values:
    1. Positive Sentiment – “joy”, “love”, “surprise”
    2. Negative Sentiment – “anger”, “sadness”, “fear”

    Args:
        df (pd.DataFrame): DataFrame of the dataset

    Returns:
        pd.DataFrame: Modified DataFrame with encoded labels
    """
    df.replace(to_replace=["surprise", "love", "joy"], value=1, inplace=True)
    df.replace(to_replace=["fear", "anger", "sadness"], value=0, inplace=True)
    return df

def text_transformation(df_col: pd.Series) -> list:
    """
    Perform text transformation on input text.

    Args:
        df_col (pd.Series): Input text column in DataFrame

    Returns:
        list: Transformed text corpus
    """
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower().split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

@step
def text_main_preprocess(df: pd.DataFrame) -> Tuple[tf.Tensor, pd.Series]:
    """
    Main preprocessing step for text data.

    Args:
        df (pd.DataFrame): Input DataFrame containing text and label columns

    Returns:
        Tuple[tf.Tensor, pd.Series]: Processed input features (X) and labels (y)
    """
    try:
        df = custom_encoder(df)
        corpus = text_transformation(df['text'])

        # Create TextVectorization layer
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=1000, output_mode='int', output_sequence_length=1000, pad_to_max_tokens=True
        )
        vectorizer.adapt(corpus)

        # Convert text data to integer indices
        vectorized_data = vectorizer(corpus)

        # Save the pre-trained vectorizer
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        X = vectorized_data
        y = df.label

        return X, y
    except Exception as e:
        return e
