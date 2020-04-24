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

ResNet50_model = ResNet50(weights='imagenet')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
## Loading the model
json_file = open('VGG19_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
VGG19_model = model_from_json(loaded_model_json)
# load weights into new model
VGG19_model.load_weights("weights.best.VGG19.hdf5")
print("Loaded model from disk")

dog_names_file = open('dog_names.json', 'r')
dog_names = json.load(dog_names_file)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def VGG19_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_image(img_path):
    if (not dog_detector(img_path)) and (not face_detector(img_path)):
        return "It seems that this image doesn't contain a human or a dog, please try again"
    
    display(Image(filename=img_path))
    return VGG19_predict_breed(img_path).split('/')[2]

print(predict_image("perro.jpg"))