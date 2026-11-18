import streamlit as st
import pickle, re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

st.title("AI-Powered Fake News Detector 📰")
st.write("Paste any news here, AI will tell you if it is Fake or Real")

news = st.text_area("Paste your news here...")

if st.button("Check Now"):
    if news:
        vec = vectorizer.transform([clean(news)])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max() * 100
        if pred == 0:
            st.error(f"FAKE NEWS - {prob:.1f}% sure")
            st.write("This news contains unverified claims and sensational words.")
        else:
            st.success(f"REAL NEWS - {prob:.1f}% sure")
            st.write("The language is formal and the facts look clear.")
    else:
        st.warning("Please enter some news first!")
