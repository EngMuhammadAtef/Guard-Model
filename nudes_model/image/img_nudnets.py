# import libraries
import numpy as np
from onnxruntime import InferenceSession # onnx onnxruntime
from PIL import Image
import cv2
import sys
sys.path.append("..")

model_path = r'nudes_model/classifier_model.onnx'
nude_model = InferenceSession(model_path)
categories=[True, False]

# to load image in PIL format
def load_img(path, target_size=None, data_format="channels_last"):
    """Loads an image into PIL format."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            img = img.resize(width_height_tuple, Image.Resampling.NEAREST)
    
    img_arr = np.asarray(img, dtype="float32")
    
    if len(img_arr.shape) == 3:
        if data_format == "channels_first":
            img_arr = img_arr.transpose(2, 0, 1)
    elif len(img_arr.shape) == 2:
        if data_format == "channels_first":
            img_arr = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1]))
        else:
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1], 1))
    else:
        raise ValueError("Unsupported image shape: %s" % (img_arr.shape,))

    return img_arr/255

# function to classify image
def classify_image(image_path, image_size=(256, 256)):
    loaded_image = load_img(image_path, image_size)
    model_preds = nude_model.run(
        [nude_model.get_outputs()[0].name],
        {nude_model.get_inputs()[0].name: [loaded_image]},
    )[0]
    
    return categories[np.argmax(model_preds)], np.max(model_preds)