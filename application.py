from flask import Flask, render_template, request
import pickle
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
from src.DataProcessing import text_transformation
from wordcloud import WordCloud

# Load the dataset
df_train = pd.read_csv(r"data/train.txt", delimiter=';', names=['text', 'label'])
df_val = pd.read_csv(r"data/val.txt", delimiter=';', names=['text', 'label'])
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

corpus = text_transformation(df['text'])
custom_colors = ['#434247', '#5c5066', '#febc68', '#fcdb8c', '#149589', '#41c0b5']

# Generate WordCloud
word_cloud = WordCloud(width=1000, height=500, background_color='white', min_font_size=10).generate(' '.join(corpus))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result_page():
    if request.method == 'POST':
        # Get the text input from the form
        input_text = request.form.get('text_input')
        
        # Load the trained model
        with open('artifacts\LSTM.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Use the loaded model to make predictions on the input text
        final_output = model(input_text)
    
    # Pass the text input and the model's prediction to the result.html template
    return render_template('result.html', text_input=final_output)


# Route to serve HTML page
@app.route('/EDA', methods=['POST','GET'])
def EDA():
    # Create Plotly Count Plot
    fig = px.histogram(df, x='label', title='Count Plot', color='label',
                       color_discrete_sequence=custom_colors,
                       labels={'label': 'Categories'}, width=600, height=400)
    plot_data = fig.to_json()

    # Convert WordCloud image to HTML string
    word_cloud_data = plt.imshow(word_cloud).to_html()
    
    return render_template('EDA.html', plot_data=plot_data, word_cloud_data=word_cloud_data)


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True, threaded=False)
