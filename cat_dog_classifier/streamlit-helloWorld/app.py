import streamlit as st
#from tensorflow.keras.models import Sequential
from tensorflow import keras
import numpy as np 
import cv2

model = keras.models.load_model('./cats_dogs_RGB_CNN-1613859215.h5')
CATEGORIES = ['cat','dog']

st.write("""
    # Image Classifier
    """)
IMAGE_DIMENSION = 150
def prediction(image_data):
    try:
        #image_array = cv2.imread(filename,cv2.IMREAD_COLOR)
        print('prediction method')
        img = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
        resized_array = cv2.resize(img, (IMAGE_DIMENSION,IMAGE_DIMENSION))
        resized_array = np.array(resized_array).reshape(-1, IMAGE_DIMENSION,IMAGE_DIMENSION,3)
        #print(resized_array)
        predicted_class = model.predict([resized_array])
        print(predicted_class)
        st.markdown(CATEGORIES[int (predicted_class[0][0])])
    #return resized_array
    except Exception as e:
        print(e)
# FILE_NAME = 'unknown.jpg'
# image = Image.open(FILE_NAME)
# image.show()      
#print(input_data)      

image = st.file_uploader('File uploader',type=['jpg','jpeg'])
if image is not None:
    file_details = {"FileName":image.name,"FileType":image.type,"FileSize":image.size}
    image_data = image.read()
    st.image(image_data)
    st.write(file_details)
    prediction(image_data)



