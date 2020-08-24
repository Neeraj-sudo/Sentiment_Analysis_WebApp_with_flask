

from flask import Flask,render_template,url_for,request
import pickle
from text_processing import text_process

app = Flask(__name__)
sentiment_model = pickle.load(open('model.pkl', 'rb'))
pipe_model = pickle.load(open('pipe_model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction = pipe_model.predict([message])
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)