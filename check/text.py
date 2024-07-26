# text moderator
from caption_model.text.Offensive_model import predict_offensive
import sys
sys.path.append("..")

def check_text(text):
    is_offensive = predict_offensive(text)
    result = {'is_safe': True if not is_offensive else False, 'resson': f'The text contains offensive words: {text}' if is_offensive else ''} 
    return result
