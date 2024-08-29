import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A - Rag ChatBot")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only. Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
""")

def initialize_vector_store():
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./us_census")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Initializing Vector Store DB..."):
            initialize_vector_store()
        st.success("Vector Store DB Is Ready")
    else:
        st.info("Vector Store DB is already initialized")

# UI for initializing document embedding
if st.button("Initialize Document Embedding"):
    vector_embedding()

# Input for user question
prompt1 = st.text_input("Enter Your Question About the Documents")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please initialize the document embedding first.")
    else:
        with st.spinner("Processing your question..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            process_time = time.process_time() - start
            
            st.write("Answer:", response['answer'])
            st.info(f"Response time: {process_time:.2f} seconds")

        # Display relevant document chunks
        with st.expander("Document Similarity Search Results"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")

# Add some additional information or instructions for the user
st.sidebar.markdown("""
## How to use:
1. Click 'Initialize Document Embedding' to prepare the system.
2. Enter your question about the documents in the text box.
3. View the answer and related document chunks.

Note: Initialization is required only once per session.
""")



