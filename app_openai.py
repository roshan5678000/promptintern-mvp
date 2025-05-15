
import streamlit as st
import openai
import os
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PromptIntern – OpenAI Powered", layout="wide")
st.title("🎓 PromptIntern – AI Internship Matcher")

# Load OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Helper to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

# Helper to get embedding from OpenAI
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# Upload resumes
st.header("📤 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload one or more resumes", type="pdf", accept_multiple_files=True)

# Prompt input
st.header("💬 Enter Prompt")
prompt = st.text_input("Example: 'Remote intern with Canva and sales skills'", "")

if st.button("🔍 Match Now") and prompt and uploaded_files:
    with st.spinner("Analyzing resumes with AI..."):
        prompt_embedding = get_embedding(prompt)
        if prompt_embedding is None:
            st.stop()

        matches = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if not text:
                continue
            resume_embedding = get_embedding(text)
            if resume_embedding:
                sim = cosine_similarity([prompt_embedding], [resume_embedding])[0][0]
                matches.append((file.name, sim))

        # Show results
        matches.sort(key=lambda x: x[1], reverse=True)
        st.subheader("✅ Top Matches:")
        for name, score in matches:
            st.write(f"📄 {name} — Similarity Score: {round(score*100, 2)}%")
