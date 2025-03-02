import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from googlesearch import search
from requests_html import HTMLSession  # For JavaScript rendering

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Verify NumPy
try:
    np.zeros(1)
except ImportError:
    st.error("NumPy is not installed. Please run 'pip install numpy'.")
    st.stop()

# Initialize tools
nlp = spacy.load("en_core_web_sm")
newsapi = NewsApiClient(api_key="e8b84809b38b4d838cd055c075d8dd7c")
TRUSTED_SOURCES = {"bbc.com", "reuters.com", "apnews.com"}
STATIC_TRUSTED = [
    "The Supreme Court often upholds legal doctrines impacting military cases.",
    "Veterans' lawsuits against the government face legal barriers.",
    "Military healthcare accountability remains a debated topic."
]  # Fallback trusted texts

# Load models
@st.cache_resource
def load_models():
    try:
        extractor = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return extractor, similarity_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

extractor, similarity_model = load_models()
if extractor is None or similarity_model is None:
    st.stop()

# Summarize and extract entities
def summarize_and_extract(text):
    try:
        summary = extractor(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        doc = nlp(summary)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]]
        key_points = [sent.text.strip() for sent in doc.sents if len(sent.text) > 15]
        return summary, entities, key_points if key_points else [summary]
    except Exception as e:
        st.warning(f"Extraction failed: {e}")
        return text[:200], [], [text[:200]]

# Fetch trusted texts with enhanced logic
def fetch_trusted_texts(summary, key_points, entities):
    trusted_texts = []
    queries = [summary] + [kp for kp in key_points if kp != summary][:2]
    if entities:
        queries.extend([ent[0] for ent in entities])  # Add entities as queries

    # NewsAPI attempt
    try:
        for query in queries:
            response = newsapi.get_everything(q=query, language="en", page_size=5)
            articles = response.get("articles", [])
            for article in articles:
                domain = article["url"].split("/")[2]
                if domain in TRUSTED_SOURCES:
                    text = article["title"] + " " + article["description"]
                    trusted_texts.append(text)
            if len(trusted_texts) >= 3:
                break
    except Exception as e:
        st.warning(f"NewsAPI failed: {e}")

    # Google fallback with requests-html
    if not trusted_texts:
        session = HTMLSession()
        try:
            for query in queries:
                for url in search(query, num_results=10):  # Increased to 10
                    domain = url.split("/")[2]
                    if domain in TRUSTED_SOURCES:
                        try:
                            response = session.get(url, timeout=5)
                            response.html.render(timeout=10)  # Render JavaScript
                            soup = BeautifulSoup(response.html.html, "html.parser")
                            paragraphs = soup.find_all("p")
                            text = " ".join([p.get_text() for p in paragraphs[:2]])
                            if text and len(text) > 50:
                                trusted_texts.append(text[:200])
                            if len(trusted_texts) >= 3:
                                break
                        except Exception as e:
                            st.warning(f"Scraping {url} failed: {e}")
                    if len(trusted_texts) >= 3:
                        break
                if len(trusted_texts) >= 3:
                    break
        except Exception as e:
            st.warning(f"Google fallback failed: {e}")

    # Static fallback if all else fails
    if not trusted_texts:
        trusted_texts = STATIC_TRUSTED
        st.warning("Using static trusted content due to fetch failure.")

    return trusted_texts

# Similarity analysis
def analyze_similarities(key_points, trusted_texts):
    results = []
    for point in key_points:
        try:
            point_embedding = similarity_model.encode(point, convert_to_tensor=True)
            trusted_embeddings = similarity_model.encode(trusted_texts, convert_to_tensor=True)
            if trusted_embeddings.size(0) == 0:
                results.append({"point": point, "label": "Unverified", "similarity": 0, "color": "orange"})
                continue
            similarities = util.cos_sim(point_embedding, trusted_embeddings)[0]
            max_similarity = float(torch.max(similarities).item())
            if max_similarity > 0.85:
                label, color = "Consistent", "green"
            elif max_similarity > 0.6:
                label, color = "Partially Consistent", "orange"
            else:
                label, color = "Contradictory", "red"
            results.append({"point": point, "label": label, "similarity": max_similarity, "color": color})
        except Exception as e:
            st.warning(f"Similarity analysis failed for '{point[:50]}...': {e}")
            results.append({"point": point, "label": "Error", "similarity": 0, "color": "gray"})
    return results

# Streamlit UI
st.title("News Summary & Similarity Analyzer")
st.write("Summarize text, extract entities, and check consistency with trusted sources.")

user_input = st.text_area("Enter news text:")
if st.button("Analyze"):
    if user_input:
        with st.spinner("Processing..."):
            summary, entities, key_points = summarize_and_extract(user_input)
            
            st.subheader("Summary")
            st.write(summary)
            
            st.subheader("Extracted Entities")
            if entities:
                for entity, label in entities:
                    st.write(f"- {entity} ({label})")
            else:
                st.write("No entities found.")
            
            st.subheader("Key Points")
            for i, point in enumerate(key_points, 1):
                st.write(f"{i}. {point}")

            trusted_texts = fetch_trusted_texts(summary, key_points, entities)
            st.write("Trusted snippets:", trusted_texts[:2] + ["..."] if len(trusted_texts) > 2 else trusted_texts)

            results = analyze_similarities(key_points, trusted_texts)
            st.subheader("Similarity Analysis")
            for r in results:
                st.markdown(f"- **'{r['point'][:50]}...'**: <span style='color:{r['color']}'>{r['label']}</span> (Similarity: {r['similarity']:.2f})", unsafe_allow_html=True)

            if any(r["label"] not in ["Unverified", "No Data", "Error"] for r in results):
                avg_similarity = np.mean([r["similarity"] for r in results if r["label"] not in ["Unverified", "No Data", "Error"]])
                if avg_similarity > 0.85:
                    st.success(f"Overall: Highly Consistent (Avg Similarity: {avg_similarity:.2f})")
                elif avg_similarity > 0.6:
                    st.warning(f"Overall: Partially Consistent (Avg Similarity: {avg_similarity:.2f})")
                else:
                    st.error(f"Overall: Contradictory (Avg Similarity: {avg_similarity:.2f})")
            else:
                st.warning("Overall: Unverified - Insufficient trusted data.")
    else:
        st.error("Please provide text input!")

st.write("Powered by DistilBART (summary/entities) and MiniLM (similarity).")