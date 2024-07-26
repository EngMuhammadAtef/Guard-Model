import tensorflow as tf
import cv2
import numpy as np
import sys
sys.path.append("..")

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = [False, True]
PATH = r'violence_model/video/Video_Violence_Guard'

# Load the TensorFlow kinetics classification model
MoBiLSTM_model = tf.saved_model.load(PATH)

def predict_video(video_path):
    try:
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
        frames_list = []
        predicted_class_name = ''
        
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)

        frames_array = np.expand_dims(frames_list, axis=0).astype(np.float32)
        predicted_labels_probabilities = MoBiLSTM_model(frames_array, training=False)[0].numpy()
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]

        return predicted_class_name, predicted_labels_probabilities[predicted_label]
    
    finally:
        video_reader.release()
