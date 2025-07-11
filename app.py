import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# Streamlit UI Config
# ----------------------------
st.set_page_config(page_title="📰 Fake News Detector", layout="wide")
st.title("📰 Fake News Detection System")


# ----------------------------
# Helper: Text Cleaner
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("news.csv")
        df = df.dropna()
        if "title" in df.columns and "text" in df.columns:
            df["text"] = (df["title"] + " " + df["text"]).apply(clean_text)
        elif "text" in df.columns:
            df["text"] = df["text"].apply(clean_text)
        st.success(f"✅ Data loaded successfully: {df.shape[0]} rows")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()


# ----------------------------
# Train & Cache the Model
# ----------------------------
@st.cache_resource
def train_model():
    df = load_data()
    if df.empty or "text" not in df.columns or "label" not in df.columns:
        st.stop()

    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, vectorizer, acc, cm, report


# Train the model
model, vectorizer, accuracy, conf_matrix, report = train_model()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.markdown(
    "Detect whether a news article is Real or Fake using NLP and Machine Learning.")

st.markdown("## 📝 Input News Article Below")
user_input = st.text_area("Paste the news article here:", height=250)

if st.button("🚀 Predict Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter an article to analyze.")
    else:
        cleaned_input = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_input])
        pred = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input)[0]

        st.markdown("### 🔍 Prediction Result")
        if pred == 1:
            st.success("✅ This article is predicted to be: REAL")
        else:
            st.error("🚨 This article is predicted to be: FAKE")

        st.markdown("### 📊 Prediction Confidence")
        st.progress(prob[1] if pred == 1 else prob[0])
        st.write(
            f"🧠 Confidence → Real: {prob[1]*100:.2f}% | Fake: {prob[0]*100:.2f}%")

with st.expander("📈 View Model Metrics & Explanation"):
    st.write(f"🧮 Model Accuracy: **{accuracy * 100:.2f}%**")
    st.markdown("### 📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[
                "Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
    st.pyplot(fig)
    st.markdown("### 🧾 Classification Report")
    st.json(report)

st.markdown("---")
st.markdown("🔬 Built using: Python, scikit-learn, Streamlit, TF-IDF")
st.markdown("💡 Dataset: Fake & Real News (Kaggle)")
