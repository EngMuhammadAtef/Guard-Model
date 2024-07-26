# import libraries
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
# checking functions
from check import image, text, video

# Initial Flask object
app = Flask(__name__)

# Authentication setup
auth = HTTPBasicAuth()

VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'WebM'}
IMAGE_ALLOWED_EXTENSIONS = {'jpg', 'jpeg','png', 'gif', 'tiff', 'WebM'}

# In-memory user storage
users = {
    "user1": generate_password_hash("msso3ks24as7fh48fasdakms"),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/check_text', methods=['POST'])
@auth.login_required
def check_text():
    # Get text from request body
    text_from_api = request.get_json().get('text')  # Access 'text' key in JSON

    if text_from_api is None:
        return jsonify({'error': 'Missing text data'}), 400  # Handle missing data

    # Make predictions
    result = text.check_text(text_from_api)
    
    # Return prediction as JSON
    return jsonify(result)

@app.route('/check_media', methods=['POST'])
@auth.login_required
def check_media():
    # Validate request for media file using Flask-Requests
    if 'media' not in request.files:
        return jsonify({'error': 'Missing List of media file'}), 400
    
    media_files = request.files.getlist('media')
    results = []

    for media in media_files:
        # get the path of the media
        media_path = media.filename
        if not media_path:
            results.append({'error': 'Empty media file'})
            continue
        
        extension = media_path.rsplit('.', 1)[1].lower()
        
        # if the media is a video
        if extension in VIDEO_ALLOWED_EXTENSIONS:
            result = video.check_video(media_path)
            results.append(result)
        
        # if the media is a image
        elif extension in IMAGE_ALLOWED_EXTENSIONS:
            result = image.check_image(media_path)
            results.append(result)
        
        # the media in other format
        else:
            results.append({'error': 'Unsupported media format'})

    return jsonify(results)

@app.route('/check_profile', methods=['POST'])
@auth.login_required
def check_profile():
    if 'media' not in request.files:
        return jsonify({'error': 'Missing List of profile media'}), 400

    media_files = request.files.getlist('media')
    profile_texts = dict(request.form.items())
    
    if not isinstance(profile_texts, dict):
        return jsonify({'error': 'Invalid profile_texts format'}), 400

    final_result = {}
    
    # check profile media
    results = []
    for media in media_files:
        # get the path of the media
        media_path = media.filename
        if not media_path:
            results.append({'error': 'Empty media file'})
            continue
        
        extension = media_path.rsplit('.', 1)[1].lower()
        
        # if the media is a video
        if extension in VIDEO_ALLOWED_EXTENSIONS:
            result = video.check_video(media_path)
            results.append(result)
        
        # if the media is a image
        elif extension in IMAGE_ALLOWED_EXTENSIONS:
            result = image.check_image(media_path)
            results.append(result)
        
        # the media in other format
        else:
            results.append({'error': 'Unsupported media format'})
    
    # check profile text [Bio, username, Name]
    for name, text_from_api in profile_texts.items():
        result = text.check_text(text_from_api) # Make predictions
        final_result[name] = result
    
    # save the results of the profile media
    final_result["profile_media"] = results
    return jsonify(final_result)

if __name__ == '__main__':
    app.run()
