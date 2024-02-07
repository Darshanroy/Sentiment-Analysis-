import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from zenml import step
from typing import Tuple
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from typing import Union ,Tuple
from typing_extensions import Annotated
import pandas

#object of WordNetLemmatizer
lm = WordNetLemmatizer()
cv = CountVectorizer(ngram_range=(1,2))


def custom_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """
      
    1. Positive Sentiment – “joy”,”love”,”surprise”
    2. Negative Sentiment – “anger”,”sadness”,”fear”
    
    args: Dataframe of the dataset
    return : Modifed Dataframe
  
    """
    df.replace(to_replace ="surprise", value =1, inplace=True)
    df.replace(to_replace ="love", value =1, inplace=True)
    df.replace(to_replace ="joy", value =1, inplace=True)
    df.replace(to_replace ="fear", value =0, inplace=True)
    df.replace(to_replace ="anger", value =0, inplace=True)
    df.replace(to_replace ="sadness", value =0, inplace=True)

    return df

def text_transformation(df_col : pd.Series ):

    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus


@step
def text_main_preprocess( df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series]:
    try:
        df = custom_encoder(df)

        corpus = text_transformation(df['text'])

        traindata = cv.fit_transform(corpus)

        X = traindata
        y = df.label
        # print(X,y

        return X , y

    except Exception as e:
        return e

