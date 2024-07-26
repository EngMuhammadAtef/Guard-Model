# import libraries
import cv2
from pytesseract import image_to_string

# config settings
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
LANG = 'ara+eng'

# Define a function to preprocess the image
def preprocess_image(image_path, target_size):
    # Load the image with OpenCV
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to the target size
    resized_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img

# Define a function to extract the text
def image2text(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    image = preprocess_image(image_path, target_size)
    string = image_to_string(image, lang=LANG)    
    return string
