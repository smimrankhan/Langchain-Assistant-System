import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPEN_API_KEY"]

# Initialize session state
if "vector" not in st.session_state:
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.embedding = OpenAIEmbeddings(api_key=openai_api_key)
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)

# Application title
st.title("AI Knowledge Assistant")

# Initialize LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input section
st.header("Ask Your Question")
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time:", time.process_time() - start)
    
    # Display response
    st.header("Response")
    st.write(response["answer"])

    # Document similarity search expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.write("--------------------------------")

# Additional features
st.header("Additional Features")
col1, col2 = st.columns(2)

# Feature 1: Contextual Help
with col1:
    st.subheader("Contextual Help")
    st.write("This application uses AI to answer questions based on the provided context.")

# Feature 2: FAQ
with col2:
    st.subheader("FAQ")
    faq_expander = st.expander("Frequently Asked Questions")
    with faq_expander:
        st.write("Q: How does this application work?")
        st.write("A: It uses AI to retrieve relevant documents and answer questions.")
        st.write("Q: What is the source of the documents?")
        st.write("A: The documents are sourced from the specified URL.")

