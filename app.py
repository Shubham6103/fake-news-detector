import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App interface
st.title("📰 Fake News Detector")
st.write("Enter a news article below, and the AI will tell you if it's real or fake.")

# User input
news = st.text_area("Enter News Article Text Here")

if st.button("Check"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        vect_text = vectorizer.transform([news])
        prediction = model.predict(vect_text)

        if prediction[0] == 1:
            st.success("✅ This looks like Real News.")
        else:
            st.error("🚫 Warning: This might be Fake News.")
