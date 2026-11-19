import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

css = "<style>"
css += "header {visibility: hidden;}"
css += ".stApp { background-image: url(https://images.unsplash.com/photo-1504711434969-e33886168f5c?q=80&w=1800&auto=format&fit=crop); background-size: cover; background-position: center; background-attachment: fixed; }"
css += ".block-container { background: white; padding: 35px 45px 30px 45px !important; border-radius: 12px; box-shadow: 0 15px 40px rgba(0,0,0,0.18); margin-top: 30px; max-width: 780px !important; }"
css += ".stTextArea textarea { background-color: white !important; color: #222 !important; border: 1px solid #e0e0e0 !important; border-radius: 8px !important; font-size: 17px !important; line-height: 1.6 !important; box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important; }"
css += ".stTextArea label p { font-size: 19px !important; color: #333 !important; }"
css += "button[kind='primary'] { background-color: #0a1931 !important; color: white !important; border: none !important; border-radius: 8px !important; font-size: 19px !important; font-weight: 700 !important; padding: 12px !important; }"
css += "button[kind='primary']:hover { background-color: #0a1931 !important; color: white !important; }"
css += "</style>"
st.markdown(css, unsafe_allow_html=True)

st.markdown("<h1 style='color:#0a1931; font-size:46px; margin-bottom:0px;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#444; font-size:22px; margin-left:58px; margin-top:2px;'>Analyze content for misinformation & bias in seconds</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Try Real Example"):
        st.session_state.news_text = "WASHINGTON (Reuters) - The United States said on Tuesday that it will continue to work with its allies to address security challenges in the region, according to White House officials. The announcement was made during a press briefing in Washington."
with col2:
    if st.button("Try Fake Example"):
        st.session_state.news_text = "Pope Francis Shocks World, Endorses Donald Trump for President, Releases Statement. BREAKING: The Pope just endorsed Trump and said Hillary Clinton is going to jail. Share this before they delete it!"

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

news = st.text_area("Paste the article, headline, or claim you want to check here...", value=st.session_state.news_text, height=180)

if st.button("Check Now", type="primary", use_container_width=True):
    if news.strip() == "":
        st.warning("Please paste some news first!")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        conf = round(max(prob) * 100, 1)
        if pred == 0 or str(pred).lower() == "fake":
            st.markdown("<div style='background:#f8d7da; padding:15px 20px; border-radius:8px;'><b style='font-size:19px;'>❌ Fake News - " + str(conf) + "% confidence</b><br><span style='color:#333;'>This content shows signs of misinformation. Please verify from reliable sources.</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#d9ead3; padding:15px 20px; border-radius:8px;'><b style='font-size:19px;'>✅ Real News - " + str(conf) + "% confidence</b><br><span style='color:#333;'>This content appears to be factual and consistent with reliable sources. No misleading language detected.</span></div>", unsafe_allow_html=True)
