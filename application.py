from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, classification_report
from scikitplot.metrics import plot_confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('models', 'random_forest_model.joblib')

rfc = joblib.load(model_path)

# Pre-processing function
def text_transformation(df_col):
    corpus = []
    lm = WordNetLemmatizer()
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

# Function to predict sentiment
def predict_sentiment(input_text):
    input_text = text_transformation([input_text])
    transformed_input = cv.transform(input_text)
    prediction = rfc.predict(transformed_input)
    if prediction == 0:
        return "Negative Sentiment"
    elif prediction == 1:
        return "Positive Sentiment"
    else:
        return "Invalid Statement"

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/EDA')
def EDA():
    return render_template('output.html')


# Route for custom input form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        result = predict_sentiment(input_text)
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
