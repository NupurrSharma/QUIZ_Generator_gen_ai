import streamlit as st
from ingest import ingest_documents
from utils.quiz_generator import generate_quiz, format_quiz

# Load documents and create embeddings
def load_data():
    documents = ingest_documents("data")
    return documents

def main():
    st.set_page_config(page_title="AI Quiz Generator", layout="centered")
    st.title("📘 Quiz Generator")

    uploaded_files = st.file_uploader("Upload PDF/CSV files", type=["pdf", "csv"], accept_multiple_files=True)

    if st.button("Ingest Documents") and uploaded_files:
        for uploaded_file in uploaded_files:
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Documents ingested and indexed!")

    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz using AI..."):
            docs = load_data()
            quiz_text = generate_quiz(docs)
            quiz = format_quiz(quiz_text)

        for q in quiz:
            st.markdown(f"**{q['question']}**")
            for opt in q["options"]:
                st.markdown(f"- {opt}")
            st.markdown(f"✅ **Correct Answer: {q['answer']}**")
            st.markdown("---")

if __name__ == "__main__":
    main()
