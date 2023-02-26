from flask import Flask, render_template, request
from utils import *
import os 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def classify():
    datafile = request.files['datafile']

    # data_path = datafile.filename
    # datafile.save(data_path)

    classification = preprocessClassify(datafile)
    

    return render_template('/results.html', classification = classification)


if (__name__ == "__main__"):
     app.run(debug = True, port = 8888)