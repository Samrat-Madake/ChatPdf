# rag_pipeline.py

import re
from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



# TEXT CLEANING
# =========================
def clean_pdf_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"●", "\n- ", text)
    text = re.sub(r"-\s+", "- ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# BUILD RAG CHAIN
# =========================
def build_rag_chain(pdf_path: str):

    # 1️⃣ Load PDF
    try:
    loader = PyPDFium2Loader(pdf_path)
    docs = loader.load()
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")


    for doc in docs:
        doc.metadata["source"] = pdf_path
        doc.page_content = clean_pdf_text(doc.page_content)

    # 2️⃣ Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    # 3️⃣ Embeddings + Vector Store
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="my_collection",
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.7}
    )

    # 4️⃣ LLM + Prompt
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4
    )

    prompt = PromptTemplate(
        template="""
You are an academic assistant.

Rules:
- Answer ONLY from the context
- Use bullet points if possible
- If definitions exist, include them
- If comparison is asked, answer in a table

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    chain = prompt | llm | StrOutputParser()

    return retriever, chain



# ASK QUESTION
# =========================
def ask_question(retriever, chain, question: str):

    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    answer = chain.invoke({
        "context": context_text,
        "question": question
    })

    return answer
