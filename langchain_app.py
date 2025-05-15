import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any, Optional

# Import secrets handler
from secrets_handler import setup_api_access

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceEndpoint

# RAGAS evaluation imports
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas.llm import HuggingFaceEvaluationLLM
from ragas import evaluate

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation with RAGAS & LangChain",
    page_icon="🔍",
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
        token = os.environ.get('HUGGINGFACE_API_TOKEN') or os.environ.get('HF_API_TOKEN')
        if not token:
            st.error("Hugging Face API token not found. Please set up your token first.")
            st.stop()
            
        # Use HF Endpoint for Mistral
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_length=2048,
            temperature=0.1,
            model_kwargs={"max_new_tokens": 512},
            huggingfacehub_api_token=token
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None


def initialize_embeddings():
    """Initialize SentenceTransformers embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None


def process_document(uploaded_file):
    """Process uploaded documents and create vector store."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Load document based on file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path)
        elif file_extension in ['csv', 'xlsx']:
            loader = CSVLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            os.unlink(file_path)
            return None
        
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Clean up temporary file
        os.unlink(file_path)
        
        return split_docs
    except Exception as e:
        st.error(f"Error processing document: {e}")
        # Clean up temp file if it exists
        if 'file_path' in locals():
            os.unlink(file_path)
        return None


def create_vectorstore(documents, embeddings):
    """Create vector store from documents."""
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def create_rag_chain(vectorstore, llm):
    """Create RAG retrieval chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def prompt_template(input_dict):
    """Format the prompt template for the LLM."""
    question = input_dict["question"]
    context = input_dict["context"]
    
    return f"""You are a helpful and truthful assistant that answers questions based on the provided context.
    
Context:
{context}

Question:
{question}

Answer the question based only on the provided context. If the context doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""


def generate_evaluation_questions(documents, llm, num_questions=5):
    """Generate evaluation questions from the document content."""
    try:
        # Combine some document content
        combined_text = "\n\n".join([doc.page_content for doc in documents[:10]])
        
        # Create a prompt for question generation
        prompt = f"""Based on the following text, generate {num_questions} diverse questions that can be answered from this content.
        The questions should be factual and specific to the content provided.
        
        Text:
        {combined_text}
        
        Generate each question on a new line, numbered 1-{num_questions}.
        """
        
        response = llm.invoke(prompt)
        
        # Parse the generated questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("- ")):
                # Remove numbers or bullet points from beginning
                clean_question = line.split(".", 1)[-1] if "." in line else line
                clean_question = clean_question.split(" ", 1)[-1] if clean_question.startswith("- ") else clean_question
                questions.append(clean_question.strip())
        
        # Ensure we have the requested number of questions
        return questions[:num_questions]
    except Exception as e:
        st.error(f"Error generating evaluation questions: {e}")
        # Return some default questions if generation fails
        return [
            "What is the main topic of this document?",
            "What are the key points mentioned?",
            "What conclusions can be drawn from the content?",
            "What evidence is presented in the document?",
            "How does the document organize its information?"
        ][:num_questions]


def run_rag_evaluation(questions, rag_chain, documents, llm):
    """Run RAG evaluation using RAGAS."""
    try:
        # Generate answers using the RAG chain
        answers = []
        retrieved_contexts = []
        
        for question in questions:
            # Get answer from RAG chain
            answer = rag_chain.invoke(question)
            answers.append(answer)
            
            # Get retrieved context (we need to use the retriever again)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(question)
            retrieved_contexts.append([doc.page_content for doc in retrieved_docs])
        
        # Prepare evaluation data
        ground_truths = [""] * len(questions)  # Ground truth is often not available in real scenarios
        
        # Create evaluation data dictionary
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths
        }
        
        # Initialize RAGAS evaluation LLM
        evaluation_llm = HuggingFaceEvaluationLLM(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.1,
            huggingfacehub_api_token=os.environ.get('HUGGINGFACE_API_TOKEN') or os.environ.get('HF_API_TOKEN')
        )
        
        # Run RAGAS evaluation
        metrics = [
            faithfulness, 
            answer_relevancy,
            context_relevancy, 
            context_precision,
            harmfulness
        ]
        
        result = evaluate(
            eval_data,
            metrics=metrics,
            llm=evaluation_llm
        )
        
        return result, eval_data
    except Exception as e:
        st.error(f"Error running RAG evaluation: {e}")
        return None, None


def display_evaluation_results(results, eval_data):
    """Display evaluation results in a readable format."""
    if results is None:
        return
    
    st.subheader("RAG Evaluation Results")
    
    # Display metrics in a table
    metrics_df = results.to_pandas()
    st.dataframe(metrics_df, use_container_width=True)
    
    # Calculate average scores
    avg_scores = {
        "Faithfulness": results.faithfulness.mean(),
        "Answer Relevancy": results.answer_relevancy.mean(),
        "Context Relevancy": results.context_relevancy.mean(),
        "Context Precision": results.context_precision.mean(),
        "Harmfulness": results.harmfulness.mean() if 'harmfulness' in results else 0
    }
    
    # Create radar chart with Plotly
    categories = list(avg_scores.keys())
    values = list(avg_scores.values())
    
    # Add the first value again to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    fig = px.line_polar(
        r=values,
        theta=categories,
        line_close=True,
        range_r=[0, 1],
        title="RAG Evaluation Metrics"
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display individual question evaluations
    st.subheader("Individual Question Evaluations")
    
    for i, question in enumerate(eval_data["question"]):
        with st.expander(f"Question {i+1}: {question}"):
            st.write("**Question:**", question)
            st.write("**Answer:**", eval_data["answer"][i])
            st.write("**Retrieved Context:**")
            for j, ctx in enumerate(eval_data["contexts"][i]):
                st.text_area(f"Context {j+1}", ctx, height=100)
            
            # Individual metrics for this question
            st.write("**Metrics:**")
            metrics = {
                "Faithfulness": results.faithfulness[i],
                "Answer Relevancy": results.answer_relevancy[i],
                "Context Relevancy": results.context_relevancy[i],
                "Context Precision": results.context_precision[i],
                "Harmfulness": results.harmfulness[i] if 'harmfulness' in results else 0
            }
            st.json(metrics)


def save_evaluation_results(results, eval_data):
    """Save evaluation results to session state history."""
    if results is None or eval_data is None:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert results to dictionary for storage
    metrics_dict = results.to_pandas().to_dict('records')[0]
    
    history_entry = {
        "timestamp": timestamp,
        "metrics": metrics_dict,
        "num_questions": len(eval_data["question"]),
        "avg_faithfulness": results.faithfulness.mean(),
        "avg_answer_relevancy": results.answer_relevancy.mean(),
        "avg_context_relevancy": results.context_relevancy.mean(),
        "avg_context_precision": results.context_precision.mean(),
        "avg_harmfulness": results.harmfulness.mean() if 'harmfulness' in results else 0
    }
    
    st.session_state.history.append(history_entry)


def main():
    st.title("🔍 RAG Evaluation System with RAGAS & LangChain")
    
    # Setup API access first
    token_available = setup_api_access()
    if not token_available:
        st.error("Hugging Face API token not configured. Please set up your token first.")
        st.info("Run token_setup.py to configure your Hugging Face API token.")
        st.stop()
    
    st.markdown("""
    This application allows you to evaluate a Retrieval-Augmented Generation (RAG) system using:
    * **Mistral 7B** as the LLM
    * **SentenceTransformers** for embeddings
    * **RAGAS** for comprehensive evaluation metrics
    * **LangChain** for the RAG pipeline
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a document for RAG", type=["pdf", "txt", "csv"])
        
        # Evaluation parameters
        st.subheader("Evaluation Parameters")
        num_questions = st.slider("Number of evaluation questions", min_value=3, max_value=10, value=5)
        
        # Evaluation button
        eval_button = st.button("Run RAG Evaluation", type="primary")
        
        # History section
        st.subheader("Evaluation History")
        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history):
                st.write(f"**{entry['timestamp']}**")
                st.write(f"Questions: {entry['num_questions']}")
                st.write(f"Avg. Faithfulness: {entry['avg_faithfulness']:.3f}")
                st.write(f"Avg. Answer Relevancy: {entry['avg_answer_relevancy']:.3f}")
                st.write("---")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["RAG Evaluation", "Document Analysis", "Comparison"])
    
    # Process uploaded file
    if uploaded_file:
        with st.spinner("Processing document..."):
            documents = process_document(uploaded_file)
            
            if documents:
                st.session_state.documents = documents
                
                # Initialize embeddings and create vector store
                embeddings = initialize_embeddings()
                if embeddings:
                    vectorstore = create_vectorstore(documents, embeddings)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"Successfully processed {uploaded_file.name} - {len(documents)} chunks created")
    
    # Run evaluation
    if eval_button and st.session_state.vectorstore:
        with st.spinner("Running RAG evaluation..."):
            # Initialize LLM
            llm = initialize_llm()
            if not llm:
                st.error("Failed to initialize LLM. Please check your configuration.")
                return
            
            # Create RAG chain
            rag_chain = create_rag_chain(st.session_state.vectorstore, llm)
            
            # Generate evaluation questions
            questions = generate_evaluation_questions(st.session_state.documents, llm, num_questions)
            
            # Run evaluation
            results, eval_data = run_rag_evaluation(questions, rag_chain, st.session_state.documents, llm)
            
            if results and eval_data:
                st.session_state.evaluation_results = results
                st.session_state.eval_data = eval_data
                save_evaluation_results(results, eval_data)
                st.success("Evaluation completed successfully!")
    
    # Display evaluation results in Tab 1
    with tab1:
        if st.session_state.evaluation_results and st.session_state.eval_data:
            display_evaluation_results(st.session_state.evaluation_results, st.session_state.eval_data)
        else:
            st.info("Upload a document and run evaluation to see results here.")
    
    # Document Analysis in Tab 2
    with tab2:
        if st.session_state.documents:
            st.subheader("Document Analysis")
            st.write(f"Total chunks: {len(st.session_state.documents)}")
            
            # Display chunk statistics
            chunk_lengths = [len(doc.page_content) for doc in st.session_state.documents]
            avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Chunk Length", f"{avg_chunk_length:.1f} chars")
            col2.metric("Min Chunk Length", f"{min(chunk_lengths)} chars")
            col3.metric("Max Chunk Length", f"{max(chunk_lengths)} chars")
            
            # Histogram of chunk lengths
            fig, ax = plt.subplots()
            ax.hist(chunk_lengths, bins=20)
            ax.set_xlabel("Chunk Length (characters)")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Chunk Lengths")
            st.pyplot(fig)
            
            # Display sample chunks
            st.subheader("Sample Chunks")
            for i, doc in enumerate(st.session_state.documents[:5]):
                with st.expander(f"Chunk {i+1}"):
                    st.text_area("Content", doc.page_content, height=150)
                    if doc.metadata:
                        st.json(doc.metadata)
        else:
            st.info("Upload a document to see analysis here.")
    
    # Comparison in Tab 3
    with tab3:
        st.subheader("Evaluation Comparison")
        if len(st.session_state.history) >= 2:
            # Create comparison dataframe
            history_df = pd.DataFrame(st.session_state.history)
            
            # Plot comparison metrics
            fig = px.line(
                history_df,
                x="timestamp",
                y=["avg_faithfulness", "avg_answer_relevancy", "avg_context_relevancy", "avg_context_precision"],
                labels={
                    "value": "Score",
                    "variable": "Metric",
                    "timestamp": "Evaluation Time"
                },
                title="RAG Evaluation Metrics Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display comparison table
            st.subheader("Metrics Comparison")
            comparison_df = history_df[["timestamp", "num_questions", "avg_faithfulness", 
                                       "avg_answer_relevancy", "avg_context_relevancy", 
                                       "avg_context_precision"]]
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Run at least two evaluations to see comparison.")


if __name__ == "__main__":
    main()
