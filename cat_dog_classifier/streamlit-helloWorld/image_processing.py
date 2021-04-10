
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess(image1): 
    image = load_img(image1, target_size=(50, 50))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((3, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    return image
