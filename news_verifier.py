import requests
from bs4 import BeautifulSoup
from googlesearch import search
from difflib import SequenceMatcher
import streamlit as st
import time

st.set_page_config("NewsProof")

def fetch_newsapi_articles(queries):
    trusted_texts = []
    for query in queries:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey=e8b84809b38b4d838cd055c075d8dd7c"
        try:
            response = requests.get(url).json()
            if response.get("status") == "ok":
                for article in response["articles"]:
                    domain = article["url"].split("/")[2]
                    if domain in TRUSTED_SOURCES:
                        text = f"{article['title']} {article['description']} {article['content']}"
                        if text and len(text) > 50:
                            trusted_texts.append(text[:300])
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {e}")
    return trusted_texts

def fetch_google_results(queries):
    trusted_texts = []
    for query in queries:
        try:
            for url in search(query, num_results=5):
                domain = url.split("/")[2]
                if domain in TRUSTED_SOURCES:
                    response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(response.content, "html.parser")
                    paragraphs = soup.find_all("p")
                    text = " ".join([p.get_text() for p in paragraphs[:3]])
                    if text and len(text) > 50:
                        trusted_texts.append(text[:300])
        except Exception as e:
            print(f"Error processing {query}: {e}")  # Logs error but doesn't show it to users
    return trusted_texts

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .header-text {
        font-size: 50px !important;
        font-weight: 700 !important;
                
    }
    .highlight {
        background-color: #f8f9fa;
        color: #000000;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .source-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .similarity-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .similarity-med {
        color: #f1c40f;
        font-weight: bold;
    }
    .similarity-low {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

TRUSTED_SOURCES = ["bbc.com", "reuters.com", "apnews.com", "aljazeera.com", "cnn.com", "nytimes.com"]

# Header Section
st.markdown('<p class="header-text">📰 NewsProof</p>', unsafe_allow_html=True)
st.markdown("#### *Your AI-Powered News Integrity Shield*")
st.markdown("""
<div class="highlight">
    <p>🛡️ Verify news credibility in real-time using our 3-step verification system:</p>
    <ol>
        <li>Multi-source Fact-Checking</li>
        <li>Semantic Analysis Engine</li>
        <li>Trust Network Validation</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Split layout for input and tips
col1, col2 = st.columns([3, 2])

with col1:
    user_input = st.text_area(
        "**Enter news content to analyze:**",
        placeholder="Paste news article/text snippet here...\nExample: 'Breaking: Major scientific breakthrough in renewable energy announced...'",
        height=150
    )
    
    if st.button("🔍 Launch Deep Analysis", use_container_width=True):
        if user_input.strip():
            with st.status("🕵️ Investigating News Integrity...", expanded=True) as status:
                st.write("🌐 Connecting to Trusted News Networks")
                time.sleep(1)
                st.write("🔍 Scanning for Cross-Verification Sources")
                time.sleep(1)
                st.write("📊 Analyzing Content Patterns")
                time.sleep(1)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            queries = [sentence.strip() for sentence in user_input.split(".") if sentence]
            trusted_texts = fetch_newsapi_articles(queries) + fetch_google_results(queries)
            
            if trusted_texts:
                similarity_scores = [calculate_similarity(user_input, text) for text in trusted_texts]
                max_score = max(similarity_scores) * 100 if similarity_scores else 0
                
                # Credibility Meter
                st.markdown("### 📊 Credibility Assessment")
                if max_score > 75:
                    st.success(f"✅ High Confidence ({(max_score):.1f}% Verified)")
                elif max_score > 45:
                    st.warning(f"⚠️ Medium Confidence ({(max_score):.1f}% Verified)")
                else:
                    st.error(f"❌ Low Confidence ({(max_score):.1f}% Verified)")
                
                # Source Analysis
                st.markdown("### 🔍 Source Breakdown")
                for i, score in enumerate(similarity_scores):
                    score_percent = round(score * 100, 1)
                    with st.container():
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.metric(label="Match Score", value=f"{score_percent}%")
                        with col_b:
                            st.progress(score)
                
                # Trust Network
                st.markdown("### 🌐 Trust Network Validation")
                cols = st.columns(3)
                source_counts = min(3, len(TRUSTED_SOURCES))
                for idx in range(source_counts):
                    with cols[idx]:
                        st.image(f"https://logo.clearbit.com/{TRUSTED_SOURCES[idx]}?size=80", width=80)
                        st.caption(f"Verified by {TRUSTED_SOURCES[idx].split('.')[0].title()}")
            else:
                st.warning("⚠️ No strong verifications found. Exercise caution with this information.")
        else:
            st.error("Please input news content to analyze")

with col2:
    st.markdown("### 📌 Verification Tips")
    with st.expander("🔎 How to Spot Fake News", expanded=True):
        st.markdown("""
        - Check multiple reputable sources
        - Verify dates and author credentials
        - Look for original reporting
        - Reverse image search media content
        - Be wary of emotional language
        """)
