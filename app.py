import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import os
import pickle

BASE_DIR = os.path.dirname(r"C:\Users\shree\Desktop\Lab\AI\Fake News Detector\app.py")  # folder where app.py is
model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("Fake News Detection")
news = st.text_area("Paste the news here:")

if st.button("Predict"):
    clean_news = preprocess(news)
    vect_news = vectorizer.transform([clean_news])
    prediction = model.predict(vect_news)[0]
    st.success("Fake News" if prediction == 1 else "Real News")
