import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "models/qa_with_sources_chain"

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Loading data...")

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=520)

    main_placeholder.text("Splitting data...")
    doc_chunks = text_splitter.split_documents(data)

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorindex_gemini = FAISS.from_documents(
        doc_chunks, embedding)

    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    vectorindex_gemini.save_local(file_path)

query = main_placeholder.text_input("Enter your question here")


if query:
    if os.path.exists(file_path):
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorindex_gemini = FAISS.load_local(
            file_path, embedding, allow_dangerous_deserialization=True
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)

        from langchain.chains import RetrievalQA

        retriever = vectorindex_gemini.as_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
        )

        result = chain({"query": query})

        st.header("Answer:")
        st.subheader(result["result"])
