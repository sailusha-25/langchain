import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import secrets handler
from secrets_handler import setup_api_access

# Langchain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

# RAGAS evaluation imports - updated to current version
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation with RAGAS & LangChain",
    page_icon="🧪",
    layout="wide"
)

# Initialize session states
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "eval_data" not in st.session_state:
    st.session_state.eval_data = None
if "history" not in st.session_state:
    st.session_state.history = []

def initialize_llm():
    """Initialize the open-source LLM (Mistral)."""
    try:
        # Ensure HF token is available
        token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_API_TOKEN")
        if not token:
            st.error("Hugging Face API token not found. Please set the HUGGINGFACE_API_TOKEN environment variable.")
            st.stop()
        
        # Verify the token format
        if not token.startswith("hf_"):
            st.warning(f"Your token '{token[:5]}...' may not be in the correct format. Hugging Face tokens usually start with 'hf_'.")
        else:
            st.success(f"Hugging Face API token loaded: {token[:5]}...")
            
        # Define model endpoints - starting with more general, then newer models
        model_endpoints = [
            "https://api-inference.huggingface.co/models/gpt2-medium", 
            "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
            "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "https://api-inference.huggingface.co/models/microsoft/phi-2", 
            "https://api-inference.huggingface.co/models/google/gemma-2b-it",
            "https://api-inference.huggingface.co/models/facebook/opt-350m",
            "https://api-inference.huggingface.co/models/gpt2" 
        ]
        
        # If HF_INFERENCE_ENDPOINT is set, prioritize it
        user_defined_endpoint = os.environ.get("HF_INFERENCE_ENDPOINT")
        final_model_list = [user_defined_endpoint] + model_endpoints if user_defined_endpoint else model_endpoints
        
        llm = None
        success = False
        attempted_models_feedback = []

        for i, endpoint_url_to_try in enumerate(final_model_list):
            if not endpoint_url_to_try: 
                continue
            
            status_key = f"model_status_{i}"
            # Use an empty placeholder that can be updated with status
            status_placeholder = st.empty()
            status_placeholder.info(f"Attempting model: {endpoint_url_to_try.split('/')[-1]} ({i+1}/{len(final_model_list)})")
            
            try:
                st.info(f"Attempting to connect to {endpoint_url_to_try.split('/')[-1]}")
                
                # Initialize with a more flexible approach and explicit wait_for_model flag
                temp_llm = HuggingFaceEndpoint(
                    endpoint_url=endpoint_url_to_try,
                    huggingfacehub_api_token=token,
                    task="text-generation",
                    max_new_tokens=512,
                    top_k=50,
                    temperature=0.1,
                    repetition_penalty=1.03,
                    model_kwargs={
                        "wait_for_model": True,
                        "use_cache": False
                    }
                )
                
                # Test with a simple prompt
                st.info("Testing model with a simple prompt...")
                test_response = temp_llm.invoke("Hello, can you respond?")
                st.success(f"✅ Model responded successfully")
                
                llm = temp_llm
                status_placeholder.success(f"✅ Successfully connected to: {endpoint_url_to_try.split('/')[-1]}")
                attempted_models_feedback.append((endpoint_url_to_try, "Success"))
                success = True
                break 
            except Exception as e:
                error_detail = str(e)
                status_placeholder.warning(f"❌ Failed for {endpoint_url_to_try.split('/')[-1]}: {error_detail[:100]}...")
                attempted_models_feedback.append((endpoint_url_to_try, f"Failed: {error_detail[:100]}..."))
                
                # More detailed error handling
                if "401" in error_detail:
                    st.error(f"401 Unauthorized for {endpoint_url_to_try.split('/')[-1]}. Your token may be invalid, expired, or lack permissions for this model.")
                    st.info("Verify your token at https://huggingface.co/settings/tokens and ensure it has 'read' access.")
                elif "404" in error_detail:
                    st.error(f"404 Not Found for {endpoint_url_to_try.split('/')[-1]}. The model endpoint is likely incorrect or the model is not available via the free Inference API.")
                elif "Could not authenticate" in error_detail:
                    st.error(f"Authentication failed for {endpoint_url_to_try.split('/')[-1]}. This may be a token permission issue.")
                
                # Try a different authentication approach as fallback
                if ("401" in error_detail or "Could not authenticate" in error_detail) and token.startswith("hf_"):
                    st.info("Attempting alternative authentication approach...")
                    try:
                        # Set token in environment variable
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                        
                        fallback_llm = HuggingFaceEndpoint(
                            endpoint_url=endpoint_url_to_try,
                            task="text-generation",
                            max_new_tokens=512,
                            top_k=50,
                            temperature=0.1,
                            repetition_penalty=1.03,
                            model_kwargs={
                                "wait_for_model": True,
                                "use_cache": False
                            }
                        )
                        
                        _ = fallback_llm.invoke("Hello, can you respond?")
                        llm = fallback_llm
                        status_placeholder.success(f"✅ Successfully connected to: {endpoint_url_to_try.split('/')[-1]} with environment variable token")
                        attempted_models_feedback.append((endpoint_url_to_try, "Success with environment variable token"))
                        success = True
                        break
                    except Exception as e2:
                        st.error(f"Alternative authentication also failed: {str(e2)[:100]}...")
        
        if not success:
            st.error("🔴 All attempted Hugging Face Inference API model endpoints failed.")
            st.markdown("**Troubleshooting Suggestions:**")
            st.markdown(
                """1. **Verify your Hugging Face API Token (`HUGGINGFACE_API_TOKEN`)**: 
                   - Ensure it's correctly set as an environment variable.
                   - Check its validity and permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (it needs at least 'read' access).
                2. **Model Access & Availability**: 
                   - Some models (e.g., Llama, Gemma, Phi) require you to accept their terms of use on their respective Hugging Face model card before you can use them via the API.
                   - Not all models are available on the free Inference API tier, or they might be temporarily unavailable.
                   - You can check model availability and try them directly on the [Hugging Face Hub](https://huggingface.co/models).
                3. **Use a Specific Endpoint via `HF_INFERENCE_ENDPOINT`**: 
                   - If you have a preferred model or a paid/dedicated inference endpoint, set the `HF_INFERENCE_ENDPOINT` environment variable to its full URL.
                   - Example: `HF_INFERENCE_ENDPOINT=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2`
                4. **Check Network Connection**: Ensure you have a stable internet connection.
                5. **Review Logs**: The messages above show which models were attempted and why they might have failed."""
            )
            with st.expander("See all attempted models and statuses:"):
                for model_url, status in attempted_models_feedback:
                    if "Success" in status:
                        st.success(f"{model_url}: {status}")
                    else:
                        st.warning(f"{model_url}: {status}")
            st.stop()
        
        st.session_state.llm = llm
        st.session_state.ragas_llm = LangchainLLMWrapper(llm)
        return llm
    except Exception as e:
        st.error(f"🚨 Critical error during LLM initialization: {e}")
        st.stop()

def initialize_embeddings():
    """Initialize HuggingFace embeddings."""
    try:
        # Using a common, effective sentence transformer model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"} # Explicitly use CPU for wider compatibility
        )
        st.session_state.embeddings = embeddings
        st.session_state.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()

# Load and process document
def load_and_process_document(uploaded_file):
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Determine loader based on file type
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(tmp_file_path)
            else:
                st.error("Unsupported file type. Please upload PDF, TXT, or CSV.")
                return None, None
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            os.remove(tmp_file_path) # Clean up temp file
            return documents, texts
        except Exception as e:
            st.error(f"Error loading or processing document: {e}")
            return None, None
    return None, None

# Create vector store
def create_vector_store(texts, embeddings):
    if texts and embeddings:
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    return None

# Build RAG chain
def build_rag_chain(vectorstore, llm):
    if vectorstore and llm:
        retriever = vectorstore.as_retriever()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # Create a function to handle inputs and create the right prompt format
        def get_context_and_question(input_data):
            if isinstance(input_data, str):
                # If input is just a string (question), retrieve docs using the newer invoke() method
                question = input_data
                try:
                    # Use the newer invoke method instead of get_relevant_documents
                    docs = retriever.invoke(question)
                    context = format_docs(docs)
                except Exception as e:
                    st.warning(f"Error retrieving documents: {e}")
                    # Fallback to get_relevant_documents if invoke fails
                    try:
                        docs = retriever.get_relevant_documents(question)
                        context = format_docs(docs)
                    except Exception as inner_e:
                        st.error(f"Both retrieval methods failed: {inner_e}")
                        context = "No relevant context could be retrieved."
                return {"context": context, "question": question}
            return input_data
        
        def generate_prompt(input_dict):
            context = input_dict["context"]
            question = input_dict["question"]
            return f"""You are a helpful assistant that provides accurate information based on the given context.
Use ONLY the following context to answer the question. If you don't know the answer or cannot find it in the context, just say you don't know.

Context:
{context}

Question: {question}

Answer:"""

        # Construct RAG chain using RunnablePassthrough and a modified structure
        def safe_llm_call(prompt):
            try:
                return llm.invoke(prompt)
            except Exception as e:
                st.error(f"LLM Error: {str(e)}")
                # Provide a fallback response
                return "I'm sorry, I encountered an error while trying to answer. Please try again or check your API connection."
        
        rag_chain = (
            RunnablePassthrough()
            | get_context_and_question
            | generate_prompt
            | safe_llm_call
            | StrOutputParser()
        )
        return rag_chain
    return None

# Prepare data for RAGAS evaluation
def prepare_eval_data(questions: List[str], rag_chain, ground_truths: List[List[str]]):
    answers = []
    contexts = []
    if not rag_chain:
        st.error("RAG chain not initialized. Cannot prepare evaluation data.")
        return None

    for question in questions:
        try:
            # Get the answer from the RAG chain
            response = rag_chain.invoke(question)
            answers.append(response)
            
            # Retrieve contexts for evaluation using the newer invoke method
            retriever = st.session_state.vectorstore.as_retriever()
            try:
                # Use the newer invoke method
                retrieved_docs = retriever.invoke(question)
            except Exception as e:
                st.warning(f"Error using retriever.invoke: {e}. Falling back to get_relevant_documents.")
                # Fallback to the older method
                retrieved_docs = retriever.get_relevant_documents(question)
                
            contexts.append([doc.page_content for doc in retrieved_docs])
            
        except Exception as e:
            st.warning(f"Error processing question '{question}': {str(e)}")
            answers.append("Error generating answer.")
            contexts.append([]) # Empty context if error

    if len(questions) != len(ground_truths):
        st.error("Number of questions and ground truths must match for RAGAS evaluation.")
        return None

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths # RAGAS expects this column name
    }
    return data

# Run RAGAS evaluation
def run_ragas_evaluation(eval_data, ragas_llm, ragas_embeddings):
    if eval_data and ragas_llm and ragas_embeddings:
        from datasets import Dataset
        dataset = Dataset.from_dict(eval_data)
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
            return result
        except Exception as e:
            st.error(f"Error during RAGAS evaluation: {e}")
            return None
    return None

# --- Streamlit App UI ---
st.title("🧪 RAG Evaluation System with RAGAS & LangChain")
st.markdown("""
Welcome! This application allows you to upload a document, ask questions against it using a RAG (Retrieval Augmented Generation) 
pipeline powered by open-source models (Mistral & Sentence Transformers), and then evaluate the RAG system using RAGAS.
""")

# Add a checkbox to bypass API token check for debugging
debug_mode = st.sidebar.checkbox("Debug Mode (Skip API Check)", value=False)

if not debug_mode:
    # Initialize API access (checks for secrets)
    setup_api_access()
else:
    st.sidebar.warning("⚠️ Running in debug mode. API features will not work without a valid token.")

# Initialize LLM and Embeddings once
if "llm" not in st.session_state and not debug_mode:
    st.session_state.llm = initialize_llm()
if "embeddings" not in st.session_state:
    st.session_state.embeddings = initialize_embeddings()

llm = st.session_state.get("llm") if not debug_mode else None
embeddings = st.session_state.embeddings
ragas_llm = st.session_state.get("ragas_llm") # Should be set during initialize_llm
ragas_embeddings = st.session_state.get("ragas_embeddings") # Should be set during initialize_embeddings

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")

# Document Upload
st.sidebar.subheader("1. Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload your document (PDF, TXT, CSV)", type=["pdf", "txt", "csv"])

if uploaded_file:
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        with st.spinner("Processing document..."):
            docs, texts = load_and_process_document(uploaded_file)
            if docs and texts:
                st.session_state.documents = docs
                st.session_state.vectorstore = create_vector_store(texts, embeddings)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.sidebar.success(f"Document `{uploaded_file.name}` processed and vector store created!")
            else:
                st.sidebar.error("Failed to process document.")
                st.session_state.documents = None
                st.session_state.vectorstore = None
                st.session_state.uploaded_file_name = None
elif st.session_state.get("uploaded_file_name"):
    st.sidebar.info(f"Using previously uploaded document: `{st.session_state.uploaded_file_name}`")

# --- Main Area for Interaction and Evaluation ---
if st.session_state.vectorstore and llm:
    st.header("💬 Chat with your Document (RAG)")
    rag_chain = build_rag_chain(st.session_state.vectorstore, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if rag_chain:
                    try:
                        response = rag_chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        # Add to history for RAGAS
                        st.session_state.history.append({"question": prompt, "answer": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                else:
                    st.error("RAG chain not available.")
    
    st.markdown("---")
    st.header("📊 RAGAS Evaluation Setup")

    num_eval_questions = st.number_input("Number of questions for RAGAS evaluation", min_value=1, max_value=10, value=3, step=1)
    
    eval_questions_list = []
    eval_ground_truths_list = []

    st.markdown("Enter your evaluation questions and their corresponding ground truth answers:")
    for i in range(num_eval_questions):
        cols = st.columns([1, 2])
        with cols[0]:
            q = st.text_input(f"Question {i+1}", key=f"q_{i}", placeholder="E.g., What is the main topic?")
        with cols[1]:
            gt = st.text_area(f"Ground Truth {i+1}", key=f"gt_{i}", placeholder="E.g., The main topic is about renewable energy.")
        
        if q and gt:
            eval_questions_list.append(q)
            eval_ground_truths_list.append([gt]) # RAGAS expects list of strings for ground_truth

    if st.button("🚀 Run RAGAS Evaluation", disabled=not (eval_questions_list and len(eval_questions_list) == num_eval_questions)):
        if not ragas_llm:
            st.error("RAGAS LLM not initialized. Cannot run evaluation.")
        elif not ragas_embeddings:
            st.error("RAGAS Embeddings not initialized. Cannot run evaluation.")
        else:
            with st.spinner("Preparing evaluation data and running RAGAS..."):
                # Use the RAG chain to get answers and contexts for the eval questions
                st.session_state.eval_data = prepare_eval_data(eval_questions_list, rag_chain, eval_ground_truths_list)
                
                if st.session_state.eval_data:
                    st.session_state.evaluation_results = run_ragas_evaluation(
                        st.session_state.eval_data, 
                        ragas_llm, 
                        ragas_embeddings
                    )
                    if st.session_state.evaluation_results:
                        st.success("RAGAS Evaluation Completed!")
                    else:
                        st.error("RAGAS Evaluation failed or returned no results.")
                else:
                    st.error("Failed to prepare data for RAGAS evaluation.")

    # Display RAGAS Results
    if st.session_state.evaluation_results:
        st.subheader("📈 Evaluation Results")
        results_df = st.session_state.evaluation_results.to_pandas()
        st.dataframe(results_df)

        # Visualizations (Example: Bar chart for scores)
        st.markdown("#### Metric Scores Overview")
        # Filter out non-numeric columns or select specific metrics for plotting
        metrics_for_plot = [col for col in results_df.columns if isinstance(results_df[col].iloc[0], (int, float))]
        if metrics_for_plot:
            avg_scores = results_df[metrics_for_plot].mean().reset_index()
            avg_scores.columns = ["metric", "average_score"]
            fig = px.bar(avg_scores, x="metric", y="average_score", title="Average RAGAS Metric Scores", color="metric")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics found to plot.")

        # Display detailed results per question if needed
        st.markdown("#### Detailed Results per Question")
        for i, row in results_df.iterrows():
            with st.expander(f"Question: {row['question']}"):
                st.write(f"**Answer:** {row['answer']}")
                st.write(f"**Contexts Provided:**")
                for idx, ctx in enumerate(row["contexts"]):
                    st.text_area(f"Context {idx+1}", ctx, height=100, disabled=True, key=f"ctx_detail_{i}_{idx}")
                st.write("**Metrics:**")
                # Display individual metric scores for this question
                metric_scores_question = {m: row[m] for m in metrics_for_plot if m in row}
                st.json(metric_scores_question)

else:
    if not uploaded_file:
        st.info("Please upload a document in the sidebar to begin.")
    elif not llm:
        st.error("LLM not initialized. Check API token and configuration.")
    elif not embeddings:
        st.error("Embeddings not initialized.")

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This app demonstrates RAG evaluation using LangChain and RAGAS with open-source models.")
