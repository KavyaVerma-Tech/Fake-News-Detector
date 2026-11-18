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
}
.block-container {
    background: rgba(255,255,255,0.96);
    padding: 35px 35px 25px 35px !important;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    margin-top: 40px;
}
h1 { color: #0a2540 !important; text-align: center; }
p { color: #333 !important; }
.stTextArea textarea {
    background-color: white !important;
    color: black !important;
    border: 1.5px solid #ddd !important;
    font-size: 16px !important;
}
.stButton button {
    background: white !important;
    color: #0a2540 !important;
    border: 1px solid #ddd !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📰 Fake News Detector</h1>")
st.markdown("<p style='text-align:center;'>AI-powered verification for news articles</p>")

col1, col2 = st.columns(2)
with col1:
    if st.button("Try Real Example"):
        st.session_state.news_text = "The Prime Minister of India addressed the nation from the Red Fort in New Delhi on Independence Day. He spoke about the country's economic growth, development plans, and the achievements of Indian scientists, farmers and soldiers."
with col2:
    if st.button("Try Fake Example"):
        st.session_state.news_text = "BREAKING: Government announces that all mobile phones will stop working from tomorrow and everyone will get free gold coins. Share this message with 10 people otherwise your phone will be blocked."

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news = st.text_area("Paste news here...", value=st.session_state.news_text, height=180)

if st.button("Check Now"):
    if news.strip() == "":
        st.warning("Please paste some news first!")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        conf = round(max(prob) * 100, 1)
        if pred == 1 or str(pred).lower() == "fake":
            st.error(f"❌ Fake News - {conf}% confidence")
        else:
            st.success(f"✅ Real News - {conf}% confidence")

st.markdown("<p style='text-align:center; font-size:13px; margin-top:20px;'>Made by Kavya | AI Project</p>")
