RAG Evaluation System with RAGAS & LangChain
This repository contains a Streamlit application for evaluating Retrieval-Augmented Generation (RAG) systems using LangChain and RAGAS, powered by open-source models.

Features
Document processing and chunking
Vector embeddings with SentenceTransformers
RAG implementation with Mistral 7B
Comprehensive evaluation using RAGAS metrics
Interactive visualizations and result analysis
Secure API token management
Installation
Clone this repository:
bash
git clone https://github.com/yourusername/rag-evaluation-system.git
cd rag-evaluation-system
Install the required packages:
bash
pip install -r requirements.txt
Set up your Hugging Face API token:
Create a .streamlit/secrets.toml file based on the example
Add your Hugging Face API token
Files in this Repository
langchain_app.py: Main application for RAG evaluation
token_setup.py: Helper app for configuring API tokens
secrets_handler.py: Utility for secure token management
.streamlit/secrets.toml.example: Example secrets configuration
Usage
First, set up your Hugging Face API token:
bash
streamlit run token_setup.py
Then run the main application:
bash
streamlit run langchain_app.py
Upload a document, configure evaluation parameters, and run the evaluation.
Deployment
Local Deployment
Run the application locally with:

bash
streamlit run langchain_app.py
Streamlit Cloud Deployment
Push this repository to GitHub
Connect your repository to Streamlit Cloud
Configure secrets in the Streamlit Cloud dashboard:
Go to App Settings > Secrets
Add your Hugging Face API token in TOML format (see example)
Requirements
streamlit>=1.24.0
langchain>=0.0.267
langchain_community>=0.0.10
langchain_core>=0.1.0
ragas>=0.0.20
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
torch>=2.0.1
transformers>=4.30.2
plotly>=5.15.0
pandas>=2.0.3
matplotlib>=3.7.2
huggingface_hub>=0.16.4
License
MIT License
