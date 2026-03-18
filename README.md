# RAG Chatbot

This repository contains two implementations of a **Retrieval-Augmented Generation (RAG) chatbot** in Python. Both allow asking questions based on a text file (`Instructions.txt`) by retrieving relevant chunks and generating answers with a local LLM (`tinyllama` via Ollama).

---

## 1. RAG from Scratch (`rag_from_scratch`)

**Technologies:** `sentence-transformers`, `FAISS`, `Ollama`

### How it works

1. Reads `Instructions.txt`  
2. Splits the text into overlapping chunks  
3. Generates embeddings for each chunk  
4. Builds a FAISS vector index  
5. For a user query:
   - Retrieves top-k relevant chunks  
   - Sends them as context to the LLM  
   - Returns the answer  

### Usage

```bash
cd rag_from_scratch
python rag_chatbot.py
```
Type your question, and the chatbot will answer based on the context. Type exit to quit.

## 2. RAG with LangChain (rag_langchain)
**Technologies:** LangChain, FAISS, HuggingFaceEmbeddings, Ollama

### How it works
1. Load and split Instructions.txt into chunks
2. Create embeddings and store them in FAISS via LangChain 
3. For a user query:
   - Retrieve top-k relevant chunks
   - Build a prompt with the context
   - Generate an answer using ChatOllama

### Usage
```bash
cd rag_langchain
python rag_chatbot.py
```
Type your question, and the chatbot will answer based on the context. Type exit to quit.

### Requirements
Python 3.10+

Ollama CLI installed and configured
