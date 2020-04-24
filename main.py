import json
import pandas as pd
import os
from flask import Flask
from flask import render_template, request, jsonify
import tempfile
from werkzeug.utils import secure_filename
from classify import predict_image
     

UPLOAD_FOLDER = './images/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/go', methods = ['GET', 'POST'])
def go():
    # save user input in query

    # use model to predict classification for query
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        print(filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        res = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        result=res
    )

def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()