import cv2
from pytesseract import image_to_string
import sys
sys.path.append("..")

# Config settings
FRAME_WIDTH, FRAME_HEIGHT = target_size = 224, 224
SEQUENCE_LENGTH = 4
LANG = 'ara+eng'

def video2text(video_path):
    try:
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
        full_text = ''

        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_size)
            frame_text = image_to_string(resized_frame, lang=LANG)
            full_text += f"Frame({frame_counter+1}:{frame_counter+1+SEQUENCE_LENGTH}): {frame_text}\n" if frame_text else ''

        return full_text

    finally:
        video_reader.release()
