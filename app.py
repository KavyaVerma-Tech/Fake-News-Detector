import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Bigger Box + Visible Newspaper Background ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(247,245,240,0.82), rgba(247,245,240,0.82)),
    url("https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=1800&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    max-width: 900px !important;
    padding-top: 40px !important;
}
.main-card {
    background: white;
    padding: 40px 48px !important;
    border-radius: 18px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    border: 1px solid #E8E5DF;
}
h1 { font-size: 34px !important; color: #1A2B4A !important; margin-bottom: 0px !important; }
.subtitle { color: #6B7280; font-size: 15px; margin-top: 6px; margin-bottom: 22px; }
.stTextArea textarea { font-size: 15px !important; }
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #D1D5DB !important;
    background: white !important;
    color: #1A2B4A !important;
    padding: 8px 16px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered verification for news articles</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Try Real Example", use_container_width=True):
        st.session_state.news_text = "India celebrated its 77th Republic Day on 26 January 2026 in New Delhi with a grand parade at Kartavya Path. The event showcased India's military strength, cultural diversity and technological achievements."
with col2:
    if st.button("Try Fake Example", use_container_width=True):
        st.session_state.news_text = "SHOCKING! Government will give 1 lakh rupees to every student if you forward this message to 10 people, offer ends tonight!"

news = st.text_area("Paste news here...", key="news_text", height=200, label_visibility="collapsed")

if st.button("Check Now", use_container_width=True, type="primary"):
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

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6B7280; font-size:12px; margin-top:14px;'>Made by Kavya | AI Project</p>", unsafe_allow_html=True)
