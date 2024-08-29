# Gemma Model Document Q&A - RAG ChatBot 🤖
This project implements a Question-Answering system using the Gemma-7b-it model from Groq, combined with RAG (Retrieval-Augmented Generation) techniques to answer questions about US Census documents.
# 🌟 Features
1. Document embedding using Google's Generative AI Embeddings <br>
2. Vector storage with FAISS for efficient similarity search <br>
3. RAG-based question answering using the Gemma-7b-it model <br>
4. Streamlit-based user interface for easy interaction <br>
# 🛠️ Installation
1. Clone this repository: <br>
` git clone https://github.com/datascientistabhishek/Gemma-Document-Q-A---RAG-CHATBOT.git` <br>
`cd Gemma-Document-Q&A-RAG-CHATBOT`<br>
2. Install the required packages:
` pip install -r requirements.txt` <br>
3. Set up your environment variables by Creating a .env file in the project root and add your API keys: <br>
`GROQ_API_KEY=your_groq_api_key` <br>
`GOOGLE_API_KEY=your_google_api_key`<br>
# 🚀 Usage
1. Run the Streamlit app: <br>
`streamlit run app.py`<br>
1. Open your web browser and navigate to the provided local URL. <br>
2. Click on "Initialize Document Embedding" to prepare the system. <br>
3. Enter your question about the US Census documents in the text box. <br>
4. View the answer and related document chunks. <br>
# 📝 Notes

1. The system uses the first 20 documents from the US Census directory for demonstration purposes. Adjust the docs[:20] slice in the initialize_vector_store() function to use more or fewer documents.<br>
2. Initialization of the document embedding is required only once per session. <br>
# 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page.

# Demo Vedio
https://github.com/user-attachments/assets/ead3e3ae-3b18-4d08-8a92-5b65473ba12e
