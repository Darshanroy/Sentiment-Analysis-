from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result_page():
    if request.method == 'POST':
        # Get the text input from the form
        input_text = request.form.get('text_input')
        with open('artifacts\LSTM.pkl', 'rb') as f:
            model = pickle.load(f)
        final_output = model(input_text)
    # Pass the text input to the result.html template
    return render_template('result.html', text_input=final_output)

if __name__ == '__main__':
    app.run(debug=True)
