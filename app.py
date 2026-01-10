# app.py

import streamlit as st
import tempfile
import os
import base64

from rag_pipeline import build_rag_chain
# ///////////////////////////////////////////////


# ///////////////////////////////////////////////
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ChatPDF - RAG",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# =========================
# PDF PREVIEW FUNCTION
# =========================
def show_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="700"
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


# =========================
# SIDEBAR (PDF UPLOAD + PREVIEW)
# =========================
with st.sidebar:
    st.title("📄 ChatPDF")
    st.markdown("Upload a PDF and chat with it")

    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"]
    )

    if uploaded_file:
        st.session_state.pdf_bytes = uploaded_file.read()

        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(st.session_state.pdf_bytes)
                pdf_path = tmp.name

            retriever, chain = build_rag_chain(pdf_path)

            st.session_state.retriever = retriever
            st.session_state.chain = chain
            st.session_state.messages = []

        st.success("PDF processed successfully ✅")

        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# =========================
# MAIN LAYOUT (2 COLUMNS)
# =========================
left_col, right_col = st.columns([1.1, 1.9])

# =========================
# LEFT: PDF PREVIEW
# =========================
with left_col:
    st.subheader("📘 PDF Preview")

    if st.session_state.pdf_bytes:
        show_pdf(st.session_state.pdf_bytes)
    else:
        st.info("Upload a PDF to preview it here.")

# =========================
# RIGHT: CHAT UI
# =========================
with right_col:
    st.subheader("💬 Chat with your PDF")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 🔍 Show sources if available
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔎 View source chunks"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(src)
                        st.divider()

    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })
        st.chat_message("user").markdown(user_query)

        if st.session_state.retriever is None:
            answer = "⚠️ Please upload a PDF first."
            sources = []
        else:
            with st.spinner("Thinking..."):
                retrieved_docs = st.session_state.retriever.invoke(user_query)

                context_text = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )

                answer = st.session_state.chain.invoke({
                    "context": context_text,
                    "question": user_query
                })

                sources = [doc.page_content for doc in retrieved_docs]

        # Store assistant response with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

            if sources:
                with st.expander("🔎 View source chunks"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(src)
                        st.divider()
