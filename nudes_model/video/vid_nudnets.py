# import libraries
import numpy as np
from onnxruntime import InferenceSession # onnx onnxruntime
import cv2 
import sys
sys.path.append("..")

model_path = r'nudes_model/classifier_model.onnx'
nude_model = InferenceSession(model_path)
categories=[True, False]

# function to classify video
def classify_video(video_path, SEQUENCE_LENGTH=4, image_size=(256, 256)):
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
            resized_frame = cv2.resize(frame, image_size)
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        # Convert frames list to a numpy array
        input_data = np.array(frames_list, dtype=np.float32)

        # Run inference with the ONNX model
        predicted_labels_probabilities = nude_model.run(
            [nude_model.get_outputs()[0].name], 
            {nude_model.get_inputs()[0].name: input_data})[0]
        predicted_label = np.argmax(predicted_labels_probabilities, axis=1)[0]
        predicted_class_name = categories[predicted_label]

        return predicted_class_name, np.max(predicted_labels_probabilities)
    
    finally:
        video_reader.release()
