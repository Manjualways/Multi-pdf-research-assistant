import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------
# Load Environment Variables
# -----------------------------------

load_dotenv()

# -----------------------------------
# Streamlit Config
# -----------------------------------

st.set_page_config(
    page_title="Multi PDF Research Assistant",
    page_icon="📚"
)

st.title("📚 Multi PDF Research Assistant")

st.write(
    "Upload multiple PDFs and ask questions across all documents."
)

# -----------------------------------
# OpenRouter LLM
# -----------------------------------

def get_llm():

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

# -----------------------------------
# Extract Text
# -----------------------------------

def extract_text(pdf_files):

    text = ""

    for pdf in pdf_files:

        reader = PdfReader(pdf)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text

# -----------------------------------
# Create Chunks
# -----------------------------------

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks

# -----------------------------------
# Create Vector Store
# -----------------------------------

def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    return vectorstore

# -----------------------------------
# Upload PDFs
# -----------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------------
# Process PDFs
# -----------------------------------

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded"
    )

    if st.button("Process PDFs"):

        with st.spinner(
            "Processing PDFs..."
        ):

            text = extract_text(
                uploaded_files
            )

            chunks = create_chunks(
                text
            )

            st.write(
                f"Total Chunks: {len(chunks)}"
            )

            vectorstore = create_vectorstore(
                chunks
            )

            st.session_state[
                "vectorstore"
            ] = vectorstore

        st.success(
            "PDFs Processed Successfully!"
        )

# -----------------------------------
# Ask Questions
# -----------------------------------

if "vectorstore" in st.session_state:

    st.subheader(
        "Ask Questions"
    )

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        try:

            docs = st.session_state[
                "vectorstore"
            ].similarity_search(
                question,
                k=4
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            prompt = f"""
You are a research assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say "I could not find that information in the uploaded PDFs."

Context:
{context}

Question:
{question}

Answer:
"""

            llm = get_llm()

            response = llm.invoke(
                prompt
            )

            st.subheader(
                "Answer"
            )

            st.write(
                response.content
            )

            with st.expander(
                "View Retrieved Chunks"
            ):

                for i, doc in enumerate(docs):

                    st.write(
                        f"Chunk {i+1}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()

        except Exception as e:

            st.error(
                str(e)
            )