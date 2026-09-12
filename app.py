import streamlit as st
import os
from rag import (
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    store_chunks,
    search_relevant_chunks,
    generate_answer
)

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")
st.title("📄 RAG Chatbot - Ask Questions About Your PDF")

# Session state init
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Reading and processing PDF..."):
            # Save uploaded file temporarily
            os.makedirs("documents", exist_ok=True)
            temp_path = os.path.join("documents", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # RAG pipeline: extract -> chunk -> embed -> store
            text = extract_text_from_pdf(temp_path)
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            store_chunks(chunks, embeddings)

            st.session_state.processed = True
        st.success(f"PDF processed! Created {len(chunks)} chunks. You can ask questions now.")

if st.session_state.processed:
    st.divider()
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Searching document and generating answer..."):
            relevant_chunks = search_relevant_chunks(query)
            answer = generate_answer(query, relevant_chunks)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved context (for debugging)"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
else:
    st.info("Upload a PDF and click 'Process PDF' to get started.")