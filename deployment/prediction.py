import streamlit as st
import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import cv2

# Load All Files

best_model = keras.models.load_model('./best_model.keras')
class_names = ['Cats', 'Dogs']

st.header('Cats or Dogs Prediction')

def run():
    
    upload= st.file_uploader('Cats or Dogs?', type=['jpg'])
    c1, c2= st.columns(2)
    if upload is not None:
        im = Image.open(upload)
        img = np.asarray(im)
        image = cv2.resize(img, (150, 150))
        img = tf.keras.applications.resnet50.preprocess_input(image)
        img = np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        preds = best_model.predict(img_array)
        pred_classes = np.argmax(preds, axis=1)
        c2.header('Output')
        c2.subheader('Predicted class:')
        c2.subheader(class_names[pred_classes[0]])

if __name__ == '__main__':
    run()