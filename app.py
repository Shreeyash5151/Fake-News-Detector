import streamlit as st
import pickle
import os
import re
import requests
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ------------------------
# Preprocessing Function
# ------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ------------------------
# Load Model & Vectorizer
# ------------------------
BASE_DIR = os.path.dirname(r"C:\Users\shree\Desktop\Lab\AI\Fake News Detector\app.py")
model_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection App")

# User Input Section
st.header("Predict Your Own News")
user_input = st.text_area("Paste your news article or headline here:")

if st.button("Predict User Input"):
    if user_input.strip() != "":
        clean_text = preprocess(user_input)
        vect = vectorizer.transform([clean_text])
        prediction = model.predict(vect)[0]
        confidence = max(model.predict_proba(vect)[0]) * 100
        st.success(f"Prediction: {'Fake' if prediction==1 else 'Real'}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text for prediction.")

st.markdown("---")

# Real-Time News Section
st.header("Real-Time News Detection")
api_key = st.secrets.get("NEWSAPI_KEY", None)

if api_key:
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=5&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            for article in data["articles"]:
                news_text = (article["title"] or "") + " " + (article["description"] or "")
                clean_text = preprocess(news_text)
                vect = vectorizer.transform([clean_text])
                prediction = model.predict(vect)[0]
                confidence = max(model.predict_proba(vect)[0]) * 100
                
                st.subheader(article["title"])
                st.write(article["description"])
                st.write(f"Prediction: {'Fake' if prediction==1 else 'Real'}")
                st.info(f"Confidence: {confidence:.2f}%")
                st.markdown("---")
        else:
            st.error("Failed to fetch news from API.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
else:
    st.warning("Please add your NewsAPI key in Streamlit secrets.")
