
import streamlit as st
import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="PromptIntern", layout="wide")
st.title("🎓 PromptIntern – AI Internship Matcher")

# Upload resumes
st.header("📤 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload one or more resumes", type="pdf", accept_multiple_files=True)
os.makedirs("resumes", exist_ok=True)

for file in uploaded_files:
    with open(os.path.join("resumes", file.name), "wb") as f:
        f.write(file.read())

# Prompt input
st.header("💬 Enter Prompt")
prompt = st.text_input("Example: 'Remote intern with React and Canva skills'", "")

# Matching logic
if st.button("🔍 Match Now") and prompt:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    matches = []

    for pdf_file in os.listdir("resumes"):
        pdf_path = os.path.join("resumes", pdf_file)
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                resume_embedding = model.encode(text, convert_to_tensor=True)
                score = util.cos_sim(prompt_embedding, resume_embedding).item()
                matches.append((pdf_file, score))
        except Exception as e:
            matches.append((pdf_file, 0))

    # Sort and display
    matches.sort(key=lambda x: x[1], reverse=True)
    st.subheader("✅ Top Matches:")
    for file, score in matches:
        st.write(f"📄 {file} — Similarity Score: {round(score*100, 2)}%")
