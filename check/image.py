# image moderator
from nudes_model.image.img_nudnets import classify_image
from violence_model.image.Image_Violence import predict_image
from caption_model.image.OCR_model import image2text
# text moderator
from caption_model.text.Offensive_model import predict_offensive
import sys
sys.path.append("..")

def check_image(media_path):
    # Make predictions
    is_violence, conv1 = predict_image(media_path)
    is_pornographic, conv2 = classify_image(media_path)
    text = image2text(media_path)
    is_offensive = predict_offensive(text)
    ressons = []
    if is_violence:
        ressons.append("The Media contains violence")
    if is_pornographic:
        ressons.append("The Media contains +18 scene")
    if is_offensive:
        ressons.append("The Media contains offensive words or hate speech")
    result  =  {"is_safe":True if not is_violence and not is_pornographic and not is_offensive else False, "resson": ressons}
    return result