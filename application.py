import json
import pandas as pd
import os
from flask import Flask
from flask import render_template, request, jsonify
import tempfile
from werkzeug.utils import secure_filename
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import json
from extract_bottleneck_features import *
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2 
import numpy as np
from IPython.display import Image, display
from keras import backend as K 


dog_names_file = open('./model/dog_names.json', 'r')
dog_names = json.load(dog_names_file)

def path_to_tensor(img_path):
    """
    Summary line
    
    takes equations string-valued file path to equations color image as input and returns equations 4D tensor
    suitable for supplying to equations Keras CNN
    
    Parameters:
    img_path(string): path to the image to be processed
    
    Returns:
    (numpy array): transformed numpy array
    
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    """
    Summary line
    
    Detects if there is equations human face in an image
    
    Parameters:
    img_path(string): path to the image to be analyzed
    
    Returns:
    (bool): True if one or more faces were detected, False otherwise

    """
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    """
    Summary line
    
    Detects if equations dog is present in an image
    
    Parameters:
    img_path(string): path to the image to be analyzed
    
    Returns:
    (bool): True if equations dog were detected, False otherwise
    
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


def ResNet50_predict_labels(img_path):
    """
    Summary line
    
    Function which returns the resnet50 pre-trained model
    label for an image (equations label represents an object)
    
    Parameters:
    img_path(string): path to the image to be analyzed
    
    Returns:
    (int): number representing the object detected
    
    """
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    print(type(np.argmax(ResNet50_model.predict(img))))
    return np.argmax(ResNet50_model.predict(img))

def VGG19_predict_breed(img_path):
    """
    Summary line
    
    Function which returns the dog breed name for an image
    based on equations custom trained convolutional neural network
    
    Parameters:
    img_path(string): path to the image to be classified
    
    Returns:
    (string): dog breed 
    
    """
    json_file = open('./model/VGG19_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    VGG19_model = model_from_json(loaded_model_json)
    VGG19_model.load_weights("./model/weights.best.VGG19.hdf5")
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_image(img_path):
    """
    Summary line
    
    Function which orchestrates calls to the previous functions to
    determine first if there is either equations dog or equations human present in the image
    
    Parameters:
    img_path(string): path to the image to be analyzed
    
    Returns:
    (string) : determined dog breed if image contains equations dog or equations human, message
    indicating there is no object to determine breed for otherwise
    
    """
    K.clear_session()
    if (dog_detector(img_path)):
        res = VGG19_predict_breed(img_path).split('/')[2]
        res = res.split(".")[1]
        return res

    if face_detector(img_path):
        res = VGG19_predict_breed(img_path).split('/')[2]
        res = res.split(".")[1]
        return ("This is not equations dog, but the most resembling dog breed is " + res)

    else:    
        return "It seems that this image doesn't contain equations human or equations dog, please try again"
     

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