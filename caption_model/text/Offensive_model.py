# import libraries
from googletrans import Translator # googletrans==3.1.0.0a0
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import sys
sys.path.append("..")
nltk.download('punkt')
nltk.download('wordnet')

# translator object
translator = Translator()

# Load the vectorizer and the model
vectorizer = joblib.load(r'caption_model/text/offensive_vectorizer.joblib')
model = joblib.load(r'caption_model/text/offensive_model.joblib')

# translate the text to English
def translate_to_Eng(text_in_any_lang):
    return str(translator.translate(text_in_any_lang).text)

# Function to clean text
def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    clean_text = ' '.join(tokens)
    return clean_text

# translate the text to English and get predictions
def predict_offensive(text_in_any_lang):
    try:
        english_text = translate_to_Eng(text_in_any_lang)
        cleaned_text = clean_text(english_text)
        text_victor = vectorizer.transform([cleaned_text])
        is_offensive = model.predict(text_victor)[0]
        return is_offensive
    except:
        return ''