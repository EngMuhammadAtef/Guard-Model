import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("..")

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
CLASSES_LIST = [True, False]
PATH = r'violence_model/image/image_violence_model'

MoBi_model = tf.saved_model.load(PATH)

# Define a function to preprocess the image
def preprocess_image(image_path, target_size):
    # Load the image with OpenCV
    img = cv2.imread(image_path)
    # Resize the image to the target size
    img = cv2.resize(img, target_size)
    # Convert the image to a numpy array and normalize it
    img_array = np.array(img) / 255.0
    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Define a function to predict the label of a new image
def predict_image(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, target_size)
    # Make a prediction
    predictions = MoBi_model(preprocessed_image, training=False)[0].numpy()
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Get the class label
    predicted_class_label = CLASSES_LIST[predicted_class_index]
    return predicted_class_label, predictions[predicted_class_index]
