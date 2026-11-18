import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

css = "<style>"
css += "header {visibility: hidden;}"
css += ".stApp { background-image: url(https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=2000); background-size: cover; background-position: center; }"
css += ".block-container { background: white; padding: 40px 40px 30px 40px !important; border-radius: 16px; box-shadow: 0 15px 40px rgba(0,0,0,0.15); margin-top: 30px; }"
css += ".stTextArea textarea { background-color: white !important; color: #222 !important; border: 1.5px solid #e5e5e5 !important; border-radius: 10px !important; font-size: 17px !important; }"
css += "</style>"
st.markdown(css, unsafe_allow_html=True)

st.markdown("<h1 style='color:#0a1931; font-size:48px; margin-bottom:0px;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#444; font-size:22px; margin-top:5px;'>Analyze content for misinformation and bias in seconds</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Try Real Example"):
        st.session_state.news_text = "Local authorities confirmed the city council voted unanimously yesterday to approve the new parks and recreation funding bill. The proposal includes upgrades to playgrounds, walking trails, and community garden space. Residents are invited to attend the public meeting next Thursday at 6pm at City Hall to share feedback."
with col2:
    if st.button("Try Fake Example"):
        st.session_state.news_text = "BREAKING: Government announces that all mobile phones will stop working from tomorrow and everyone will get free gold coins. Share this message with 10 people otherwise your phone will be blocked."

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news = st.text_area("Paste the article, headline, or claim you want to check here...", value=st.session_state.news_text, height=180)

if st.button("Check Now"):
    if news.strip() == "":
        st.warning("Please paste some news first!")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        conf = round(max(prob) * 100, 1)
        if pred == 1 or str(pred).lower() == "fake":
            st.markdown("<div style='background:#f8d7da; padding:15px 20px; border-radius:10px;'><b>Fake News - " + str(conf) + "% confidence</b><br>This content shows signs of misinformation. Please verify from reliable sources.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#d9ead3; padding:15px 20px; border-radius:10px;'><b>Real News - " + str(conf) + "% confidence</b><br>This content appears to be factual and consistent with reliable sources. No misleading language detected.</div>", unsafe_allow_html=True)
