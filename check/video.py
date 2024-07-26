# video moderator
from nudes_model.video.vid_nudnets import classify_video
from violence_model.video.Video_Violence import predict_video
from caption_model.video.OCR_model import video2text
# text moderator
from caption_model.text.Offensive_model import predict_offensive
import sys
sys.path.append("..")

def check_video(media_path):
    # Make predictions
    is_violence, conv1 = predict_video(media_path)
    is_pornographic, conv2 = classify_video(media_path)
    text = video2text(media_path)
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