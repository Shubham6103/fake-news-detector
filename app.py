import streamlit as st
from transformers import pipeline

# Load BERT fake news classifier
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

model = load_model()

# UI
st.title("🧠 BERT Fake News Detector")
st.write("Enter a news article or headline to check if it's real or fake using a pretrained BERT model.")

# Input
news = st.text_area("Paste News Text Here:")

# Predict
if st.button("Analyze"):
    if news.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing with BERT..."):
            prediction = model(news)[0]
            label = prediction['label']
            score = prediction['score']

            if label == 'REAL':
                st.success(f"✅ This appears to be REAL news (confidence: {score:.2%})")
            else:
                st.error(f"🚫 This may be FAKE news (confidence: {score:.2%})")
