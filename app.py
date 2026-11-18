import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.markdown("""
<style>
header {visibility: hidden;}
.stApp {
    background-image: url("https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=2000");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    background: white;
    padding: 40px 40px 30px 40px !important;
    border-radius: 16px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    margin-top: 30px;
    max-width: 750px !important;
}
.stTextArea textarea {
    background-color: white !important;
    color: #222 !important;
    border: 1.5px solid #e5e5e5 !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    line-height: 1.6 !important;
}
.stTextArea label p { font-size: 18px !important; color: #333 !important; font-weight: 500 !important; }
div[data-testid="stColumn"] button {
    background: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
    font-size:
