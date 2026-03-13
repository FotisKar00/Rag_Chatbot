# RAG Chatbot with FAISS and Ollama

This project implements a simple Retrieval-Augmented Generation (RAG) chatbot using Python.

The system reads a text document, splits it into chunks, converts them into embeddings, stores them in a FAISS vector index, and retrieves the most relevant chunks to answer user questions using a local LLM.

## Technologies Used

- Python
- SentenceTransformers
- FAISS (Vector Database)
- Ollama
- TinyLlama LLM
- NumPy

## How It Works

1. The text file (`Instructions.txt`) is loaded.
2. The text is split into overlapping chunks.
3. Each chunk is converted into an embedding vector using SentenceTransformers.
4. All embeddings are stored in a FAISS vector index.
5. When the user asks a question:
   - The query is converted into an embedding.
   - FAISS searches for the most similar chunks.
   - The top relevant chunks are retrieved.
   - The retrieved context and the user question are sent to a local LLM.
   - The LLM generates an answer based only on the retrieved context.

## RAG Pipeline

User Query  
↓  
Query Embedding  
↓  
Vector Similarity Search (FAISS)  
↓  
Retrieve Top-K Chunks  
↓  
Send Context + Query to LLM  
↓  
Generated Answer

## How to Run

Install the required libraries:

pip install sentence-transformers faiss-cpu numpy ollama

Make sure Ollama is installed and running.

Then run the chatbot:

python Rag_Chatbot.py

Type `exit` to stop the chatbot.

## Project Goal

The goal of this project is to demonstrate the basic architecture of a Retrieval-Augmented Generation (RAG) system and how vector databases can improve LLM responses by retrieving relevant context before generating an answer.
