import streamlit as st
from transformers import pipeline

# Load BERT model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-mini-finetuned-fake-news")

model = load_model()

st.title("🧠 BERT Fake News Detector")
st.write("Enter a news article or headline to check if it's real or fake using a pretrained BERT model.")

news = st.text_area("Paste News Text Here:")

if st.button("Analyze"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = model(news)[0]
            label = result['label']
            score = result['score']

            if label == 'REAL':
                st.success(f"✅ This appears to be REAL news (confidence: {score:.2%})")
            else:
                st.error(f"🚫 This may be FAKE news (confidence: {score:.2%})")
