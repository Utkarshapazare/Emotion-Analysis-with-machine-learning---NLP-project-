# 💬 Emotion Analysis using Machine Learning & NLP

An End-to-End Natural Language Processing (NLP) web application that analyzes conversational text and classifies user comments into specific emotional states (such as joy, sadness, anger, fear, and neutral) using Machine Learning and Flask.

---

## 🚀 Features
* **Text Preprocessing Pipeline:** Cleans raw text by removing HTML tags, filtering English stopwords, normalizing emojis/emoticons, and applying Porter Stemming.
* **TF-IDF Vectorization:** Converts unstructured text into meaningful numerical feature vectors using Term Frequency-Inverse Document Frequency.
* **Machine Learning Model:** Trained on conversational datasets using Scikit-Learn.
* **Interactive Web Interface:** A lightweight, user-friendly Flask web application styled with Bootstrap for real-time emotion predictions.

---

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Machine Learning / NLP:** Scikit-Learn, NLTK, Pandas, NumPy
* **Web Framework:** Flask
* **Frontend:** HTML5, Bootstrap 4
* **Serialization:** Pickle

---

## 📊 Dataset Source
* **Dataset Name:** DailyDialog Dataset
* **Source:** [Kaggle DailyDialog Dataset](https://www.kaggle.com/) *(or you can link the exact Kaggle dataset page you used)*
* **Description:** A multi-turn dialog dataset which reflects human daily communications and captures various emotional labels.

---

## 📂 Project Structure
```text
📦 Emotion-Analysis-with-machine-learning-NLP
│
├── static/                   # Static assets / stylesheets
├── templates/
│   └── index.html            # Frontend HTML template for the web app
├── app.py                    # Main Flask application file
├── clf.pkl                   # Trained Logistic Regression classifier model
├── tfidf.pkl                 # Fitted TF-IDF vectorizer
├── requirements.txt          # Python project dependencies
└── README.md                 # Project documentation
