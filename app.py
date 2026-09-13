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

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

# ---- CSS: aligns the user's messages to the right, assistant's to the left ----
st.markdown("""
<style>
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    text-align: right;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
    background-color: #2b6cb0;
    border-radius: 15px;
    padding: 10px 15px;
    display: inline-block;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
    background-color: #333844;
    border-radius: 15px;
    padding: 10px 15px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# Session state init
if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ================= SIDEBAR =================
with st.sidebar:
    st.title("📄 RAG Chatbot")
    st.markdown("Upload a PDF and ask questions about it.")
    st.divider()

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Reading and processing PDF..."):
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
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name
            st.success(f"Processed! ({len(chunks)} chunks)")

    if st.session_state.processed:
        st.divider()
        st.caption(f"📎 Active document: **{st.session_state.pdf_name}**")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

# ================= MAIN CHAT AREA =================
st.title("💬 Chat")

if not st.session_state.processed:
    st.info("👈 Upload a PDF from the sidebar and click 'Process PDF' to get started.")
else:
    # Display all previous chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Chat input box - clears automatically after each question
    query = st.chat_input("Ask a question about the PDF...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                relevant_chunks = search_relevant_chunks(query)
                answer = generate_answer(query, relevant_chunks)
            st.write(answer)

            with st.expander("Show retrieved context (for debugging)"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer
        })