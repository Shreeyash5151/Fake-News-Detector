# Fake News Detection

## Overview
**Fake News Detection** is a Machine Learning project that classifies news articles as **Real** or **Fake**.  
This project uses **NLP techniques**, **TF-IDF vectorization**, and a **Logistic Regression model**.  
It includes a **Streamlit web app** for interactive predictions.

---

## Features
- Detect whether a news article is real or fake.
- Interactive web interface using **Streamlit**.
- Preprocessing includes lowercasing, punctuation removal, stopwords removal.
- Easy-to-run Python project with saved ML model and vectorizer.

---

## Dataset
The project uses the **Fake and Real News Dataset** from Kaggle:  
- [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Dataset contains ~20k news articles labeled as `Fake` or `Real`.

> **Note:** Only a small sample CSV is included in the repo; download full dataset separately if needed.

---

## Installation

1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/FakeNewsDetection.git
cd FakeNewsDetection
```
2. Create virtual environment:
```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
3. To install dependencies:
```
pip install -r requirements.txt

```
4. Train the model:
```
python src/train_model.py

```
5. Run the streamlit app:
```
streamlit run app.py

```
6. Folder Structure:
```
FakeNewsDetection/
├─ data/              # Sample CSV or dataset links
├─ models/            # Saved ML model and vectorizer
├─ src/               # Scripts for preprocessing & training
├─ app.py             # Streamlit application
├─ requirements.txt   # Python dependencies
├─ README.md          # Project description
└─ .gitignore         # Ignore large files

