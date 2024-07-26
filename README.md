# Guard model

## Introduction:

## [What is a Content Moderation Model (Guard Model) in AI?]{.mark}

## [A content moderation model in AI is a sophisticated framework designed to ensure that online content adheres to community standards and legal guidelines. It leverages machine learning and AI technologies to automatically detect and manage inappropriate content, thereby creating a safer and more respectful online environment.]{.mark}

## [Why This Matters]{.mark}

## [Traditional content supervision methods were labor-intensive and time-consuming. Our new system addresses these challenges by significantly reducing the time required for content moderation and enhancing the accuracy and consistency of content evaluation.]{.mark}

## [Key Technologies and Models Used]{.mark}

## [To build this robust system, we employed a variety of cutting-edge models and technologies:]{.mark}

## [- Flask App as an API: Multiple endpoints to handle text, images, and videos.]{.mark}

## [- Text Moderation: We used an English profanity words dataset, scraped from Twitter and Wikipedia's talk page comments (dataset link:]{.mark} []{.mark}[English profanity words dataset (kaggle.com)]{.underline}[).]{.mark}

## [- Translation Model:Texts are translated into English using the Google Translator API before checking for offensive language or hate speech.]{.mark}

## [- Image and Video Moderation: Our system supervises media to detect violence, +18 nudity, and offensive words on images.]{.mark}

## [Technologies Used:]{.mark}

## [- Optical Character Recognition (OCR) and Google Translator libraries.]{.mark}

## [- NLP (bag of words) for detecting offensive words (model available on Kaggle:]{.mark} []{.mark}[Offensive Words model (kaggle.com)]{.underline}[)]{.mark}

## [- Computer Vision for detecting violence and nudity.]{.mark}

## [This comprehensive system integrates multiple AI models to provide seamless and efficient content moderation, ensuring our platform remains safe and welcoming for all users.]{.mark}

## [Filename: app.py]{.mark}

## Content for Documentation Text File

## API Documentation for Media Content Moderation

## This API provides functionalities to analyze various types of media content for violence, pornography, and offensive language.

## Authentication

## This API requires HTTP Basic Authentication. You will need to create a user with a password in order to access the endpoints.

## Endpoints

## ● check_text (POST): analyzes a block of text for offensive language.

## ● check_media (POST): analyzes a list of media files (images or videos) for violence, pornography, and offensive language found within extracted text.

## Request Format

## ● check_text:

## ○ Body: JSON with a key named \"text\" containing the text content to be analyzed.

## ● check_media:

## ○ Body: FormData with a key named \"media\" containing a list of media files (images or videos) to be analyzed.

## Response Format

## ● All API responses are returned in JSON format.

## ● check_text:

## ○ On success: JSON object with a key named \"is_offensive\" containing a boolean value indicating the presence of offensive language.

## ○ On error: JSON object with a key named \"error\" containing an error message.

## ● check_media:

## ○ On success: JSON array containing one object for each uploaded media file. Each object contains keys:

## ■ \"is_violence\": boolean indicating presence of violence (for videos only)

## ■ \"confidence\": float representing the confidence score for violence detection (for videos only)

## ■ \"is_pornographic\": boolean indicating presence of pornography

## ■ \"confidence\": float representing the confidence score for pornography detection

## ■ \"is_offensive_text\": boolean indicating presence of offensive language in extracted text

## ○ On error: JSON object with a key named \"error\" containing an error message.

## Error Codes

## ● 400 Bad Request: Missing required data in the request body.

## ● 401 Unauthorized: Invalid authentication credentials.

## Supported Media Formats

## ● Images: jpg, jpeg, png, gif, tiff, WebM

## ● Videos: mp4, avi, mov, wmv, flv, WebM

## Note: This documentation is based on the code provided and may not reflect any external dependencies or specific functionalities not explicitly mentioned in the code.

Content for Documentation Text File

**Text Offensive Language Detection**

This module provides a function to predict if a piece of text contains
offensive language.

**Functionality**

-   The function takes text in any language as input.

-   It translates the text to English using Google Translate.

-   It then performs text cleaning steps including:

    -   Removing special characters and punctuation.

    -   Converting text to lowercase.

    -   Tokenization (splitting text into words).

    -   Removing stop words (common words like \"the\", \"a\", \"an\").

    -   Lemmatization (reducing words to their base form).

-   Finally, it converts the cleaned text into a numerical
    representation and uses a machine learning model to predict if the
    text is offensive.

**Function Usage**

> Python

from offensive_language_detection import predict_offensive\
\
\# Example usage\
text_to_check = \"This is an offensive text.\"\
is_offensive = predict_offensive(text_to_check)\
\
if is_offensive:\
print(\"The text is offensive.\")\
else:\
print(\"The text is not offensive.\")

**Note:**

-   This is a basic implementation and may not be perfect in all
    situations.

-   The accuracy of the model might be affected by the quality of the
    translation and the complexity of the text.

**Dependencies**

-   googletrans (\>= 3.1.0.0a0)

-   joblib

-   nltk

-   numpy (required by some NLP libraries)

**Downloaded Resources**

-   NLTK punkt tokenizer

-   NLTK wordnet for lemmatization

Content for Documentation Text File

**Video Text Extraction**

This module provides a function to extract text from a video using
Optical Character Recognition (OCR).

**Functionality**

-   The function takes a video path as input.

-   It opens the video using OpenCV library.

-   It defines the target frame size for text extraction and the number
    of frames to be processed (SEQUENCE_LENGTH).

-   The function calculates a skip window size based on the total number
    of frames and the desired sequence length.

-   It iterates through the video, reading frames at specific intervals
    based on the skip window.

-   For each frame, it resizes it to the target size and uses Tesseract
    library to perform OCR and extract text.

-   The extracted text for each frame is concatenated and returned as a
    string.

**Function Usage**

> Python

from video_text_extraction import video2text\
\
\# Example usage\
video_path = \"path/to/your/video.mp4\"\
text_from_video = video2text(video_path)\
\
if text_from_video:\
print(\"Extracted text from video:\")\
print(text_from_video)\
else:\
print(\"Failed to extract text from video.\")

**Note:**

-   The accuracy of OCR depends on the quality of the video and the
    clarity of the text within the frames.

-   This function extracts text only from a limited number of frames
    based on the SEQUENCE_LENGTH parameter.

**Dependencies**

-   OpenCV (cv2)

-   pytesseract

## 

Content for Documentation Text File

**Image Text Extraction**

This module provides a function to extract text from an image using
Optical Character Recognition (OCR).

**Functionality**

-   The function takes an image path as input.

-   It defines the target image size for text extraction (IMAGE_WIDTH,
    IMAGE_HEIGHT).

-   It uses OpenCV to load the image in grayscale mode.

-   The function then resizes the image to the target size using
    appropriate interpolation for better text clarity.

-   Finally, it uses Tesseract library to perform OCR and extract text
    from the preprocessed image.

**Function Usage**

> Python

from image_text_extraction import image2text\
\
\# Example usage\
image_path = \"path/to/your/image.jpg\"\
text_from_image = image2text(image_path)\
\
if text_from_image:\
print(\"Extracted text from image:\")\
print(text_from_image)\
else:\
print(\"Failed to extract text from image.\")

**Note:**

-   The accuracy of OCR depends on the quality of the image and the
    clarity of the text within it.

**Dependencies**

-   OpenCV (cv2)

-   pytesseract

## 

Content for Documentation Text File

**Image Nudity Classification**

This module provides a function to classify an image as nude or not
nude.

**Functionality**

-   The function takes an image path as input.

-   It loads the image using OpenCV and converts it to RGB format.

-   The function then uses Pillow (PIL Fork) library to convert the
    image to a NumPy array suitable for model input.

-   It preprocesses the image by resizing it to a specific size
    (default: 256x256) and normalizes pixel values by dividing by 255.

-   The preprocessed image is then fed into an ONNX runtime inference
    session to get predictions.

-   The model outputs a probability score for each category (nude and
    not nude).

-   The function returns the predicted category (nude or not nude) and
    the corresponding confidence score.

**Function Usage**

> Python

from image_nudity_classification import classify_image\
\
\# Example usage\
image_path = \"path/to/your/image.jpg\"\
is_nude, confidence = classify_image(image_path)\
\
if is_nude:\
print(\"The image is classified as nude with confidence:\", confidence)\
else:\
print(\"The image is classified as not nude with confidence:\",
confidence)

**Note:**

-   The accuracy of the model depends on the training data and may not
    be perfect in all cases.

-   The confidence score represents the model\'s certainty in its
    prediction.

**Dependencies**

-   NumPy

-   onnxruntime

-   Pillow (PIL Fork)

-   OpenCV (cv2)

## 

Content for Documentation Text File

**Video Nudity Classification**

This module provides a function to classify a video as nude or not nude.

**Functionality**

-   The function takes a video path as input.

-   It opens the video using OpenCV library.

-   It defines the number of frames to be processed from the video
    (SEQUENCE_LENGTH) and the target frame size for prediction
    (image_size).

-   The function calculates a skip window size based on the total number
    of frames and the desired sequence length.

-   It iterates through the video, reading frames at specific intervals
    based on the skip window.

-   For each frame, it resizes it to the target size and normalizes
    pixel values by dividing by 255.

-   A list of preprocessed frames is created.

-   Finally, the list of frames is converted to a NumPy array suitable
    for model input.

-   The preprocessed video frames are then fed into an ONNX runtime
    inference session to get predictions.

-   The model outputs a probability score for each category (nude and
    not nude) for each frame.

-   The function returns the predicted class for the entire video (nude
    or not nude) based on the frame with the highest confidence score
    and the corresponding confidence value.

**Function Usage**

> Python

from video_nudity_classification import classify_video\
\
\# Example usage\
video_path = \"path/to/your/video.mp4\"\
is_nude, confidence = classify_video(video_path)\
\
if is_nude:\
print(\"The video is classified as nude with confidence:\", confidence)\
else:\
print(\"The video is classified as not nude with confidence:\",
confidence)

**Note:**

-   The accuracy of the model depends on the training data and may not
    be perfect in all cases. The confidence score represents the
    model\'s certainty in its classification for the entire video based
    on the most confident frame.

-   This function processes only a limited number of frames based on the
    SEQUENCE_LENGTH parameter.

**Dependencies**

-   NumPy

-   onnxruntime

-   OpenCV (cv2)

## 

Content for Documentation Text File

**Image Violence Classification**

This module provides a function to classify an image as containing
violence or not.

**Functionality**

-   The function takes an image path as input.

-   It defines the target image size for prediction (IMAGE_HEIGHT,
    IMAGE_WIDTH).

-   It uses OpenCV to load the image.

-   The function then resizes the image to the target size using
    appropriate interpolation for better violence detection.

-   The image is converted to a NumPy array and normalized by dividing
    all pixel values by 255.

-   It expands the dimension of the array to match the model\'s expected
    input shape (typically a batch of images).

-   The preprocessed image is then fed into a pre-trained TensorFlow
    Keras model for violence classification.

-   The model outputs a probability score for each class (violence and
    no violence).

-   The function returns the predicted class (violence or no violence)
    and the corresponding confidence score.

**Function Usage**

> Python

from image_violence_classification import predict_image\
\
\# Example usage\
image_path = \"path/to/your/image.jpg\"\
is_violent, confidence = predict_image(image_path)\
\
if is_violent:\
print(\"The image is classified as containing violence with
confidence:\", confidence)\
else:\
print(\"The image is classified as not containing violence with
confidence:\", confidence)

**Note:**

-   The accuracy of the model depends on the training data and may not
    be perfect in all cases. The confidence score represents the
    model\'s certainty in its prediction.

-   This function only analyzes the content of the image itself and does
    not consider any context.

**Dependencies**

-   NumPy

-   TensorFlow (loaded through Keras)

-   OpenCV (cv2)

## 

Content for Documentation Text File

**Video Violence Classification**

This module provides a function to classify a video as containing
violence or not.

**Functionality**

-   The function takes a video path as input.

-   It defines the target frame size for prediction (IMAGE_HEIGHT,
    IMAGE_WIDTH) and the number of frames to be processed from the video
    (SEQUENCE_LENGTH).

-   It opens the video using OpenCV library.

-   The function calculates a skip window size based on the total number
    of frames and the desired sequence length.

-   It iterates through the video, reading frames at specific intervals
    based on the skip window.

-   For each frame, it resizes it to the target size and normalizes
    pixel values by dividing by 255.

-   A list of preprocessed frames is created.

-   Finally, the list of frames is converted into a NumPy array with an
    extra dimension at the beginning to represent a batch of video
    sequences (typically with shape (1, SEQUENCE_LENGTH, IMAGE_HEIGHT,
    IMAGE_WIDTH, 3)).

-   The preprocessed video frames are then fed into a pre-trained
    TensorFlow Keras model for violence classification.

-   The model outputs a probability score for each class (violence and
    no violence).

-   The function returns the predicted class for the entire video
    (violence or no violence) based on the frame with the highest
    confidence score and the corresponding confidence value.

**Function Usage**

> Python

from video_violence_classification import predict_video\
\
\# Example usage\
video_path = \"path/to/your/video.mp4\"\
is_violent, confidence = predict_video(video_path)\
\
if is_violent:\
print(\"The video is classified as containing violence with
confidence:\", confidence)\
else:\
print(\"The video is classified as not containing violence with
confidence:\", confidence)

**Note:**

-   The accuracy of the model depends on the training data and may not
    be perfect in all cases. The confidence score represents the
    model\'s certainty in its classification for the entire video based
    on the most confident frame.

-   This function processes only a limited number of frames based on the
    SEQUENCE_LENGTH parameter.

**Dependencies**

-   NumPy

-   TensorFlow (loaded through Keras)

-   OpenCV (cv2)

The provided Dockerfile is a good foundation for creating a Python
application container. Here\'s a breakdown of the instructions:

**1. Base Image:**

-   FROM python:3.9-slim-buster - This line specifies the base image for
    your Docker container. It uses the official Python 3.9 slim image
    from Buster (Debian 10) which is a lightweight base image suitable
    for Python applications.

**2. Update and Install Dependencies:**

-   RUN apt-get update && \\

    -   apt-get update - This command updates the package list of
        > available software packages from the repositories configured
        > for your system.

-   apt-get -qq -y install tesseract-ocr && \\

    -   apt-get -qq - This option suppresses the output of apt-get
        > commands, making the build process quieter in the terminal.

    -   -y - This option automatically answers yes to any prompts during
        > the installation process.

    -   tesseract-ocr - This package installs Tesseract, an open-source
        > Optical Character Recognition (OCR) engine used for extracting
        > text from images.

-   apt-get -qq -y install libtesseract-dev && \\

    -   libtesseract-dev - This package installs the development
        > libraries for Tesseract, which might be required if you\'re
        > building applications that interact with Tesseract directly.

-   apt-get -qq -y install ffmpeg libsm6 libxext6

    -   ffmpeg - This is a powerful command-line tool for video and
        > audio processing. It might be needed for functionalities
        > related to video processing or format conversion within your
        > application.

    -   libsm6 and libxext6 - These libraries might be dependencies for
        > ffmpeg depending on the specific functionalities used in your
        > application.

    -   

**3. Working Directory:**

-   WORKDIR /app - This line sets the working directory inside the
    container to /app. This is where your application code and other
    files will be placed.

**4. Copy Requirements:**

-   COPY requirements.txt requirements.txt - This line copies the
    requirements.txt file from your local machine to the /app directory
    inside the container. The requirements.txt file typically specifies
    the external Python libraries your application needs to run.

-   RUN pip3 install -r requirements.txt - This line installs the Python
    libraries listed in the requirements.txt file using pip3, the
    package installer for Python.

**5. Copy Application Code:**

-   COPY . . - This line copies all files and directories from your
    current working directory on the host machine to the /app directory
    inside the container. This will copy your entire application
    codebase.

**6. Run Gunicorn:**

-   CMD \[\"gunicorn\", \"app:app\"\] - This line sets the default
    command to be executed when the container starts. It uses gunicorn,
    a popular WSGI (Web Server Gateway Interface) server for running
    Python web applications. The command instructs Gunicorn to serve
    your application code located at app:app. This likely refers to a
    WSGI application entry point within your codebase (e.g., a file
    named app.py with a wsgi application object).

**Overall, this Dockerfile creates a container with Python 3.9, installs
necessary dependencies for OCR and potentially video processing, copies
your application code, and sets up Gunicorn to serve your web
application.**

**Here are some additional points to consider:**

-   Make sure your requirements.txt file is properly formatted with the
    names and versions of the Python libraries your application needs.

-   If your application uses a different WSGI server or has a different
    entry point, you might need to adjust the CMD instruction
    accordingly.

-   You can expose ports from the container using the EXPOSE instruction
    in the Dockerfile to allow external access to your application.
