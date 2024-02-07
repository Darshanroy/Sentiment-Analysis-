
import re
import pickle
import pandas as pd
import tensorflow as tf
from typing import Tuple
from zenml import step
from zenml import pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords




#object of WordNetLemmatizer
lm = WordNetLemmatizer()

def text_transformation(text : str ):

    corpus = []

    new_item = re.sub('[^a-zA-Z]',' ',str(text))
    new_item = new_item.lower()
    new_item = new_item.split()
    new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(str(x) for x in new_item))
    return corpus


def expression_check(prediction_input):
    if prediction_input == 0:
        return "Input statement has Negative Sentiment."
    elif prediction_input == 1:
        return "Input statement has Positive Sentiment."
    else:
        return "Invalid Statement."


@pipeline(enable_cache=True)
def pred_pipeline(input_text: str) -> str:

    with open('LSTM.pkl', 'rb') as f:
        model = pickle.load(f)
    ouput = model(input_text)

    return ouput

    