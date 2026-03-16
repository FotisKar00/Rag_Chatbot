# RAG Chatbot

This repository contains two implementations of a **Retrieval-Augmented Generation (RAG) chatbot** using Python. Both versions allow you to ask questions based on a text file (`Instructions.txt`) by retrieving the most relevant chunks and generating answers using a local LLM (`tinyllama` via Ollama).  

---

## Project Structure

rag_chatbot/
│
├── rag_from_scratch/ # RAG implementation from scratch
├── rag_langchain/ # RAG implementation using LangChain
└── Instructions.txt # Example text for the chatbot to answer questions from


---

## 1. RAG from Scratch (`rag_from_scratch`)

This project implements a RAG chatbot **without any framework**. It uses:

- `sentence-transformers` for embedding chunks of text
- `FAISS` for similarity search
- `Ollama` as a local LLM to generate answers

### How it works

1. Reads the text file (`Instructions.txt`)  
2. Splits the text into overlapping chunks  
3. Computes embeddings for each chunk  
4. Builds a FAISS vector index  
5. For a user query:
   - Retrieves the top-k most relevant chunks  
   - Sends them as context to the LLM  
   - Returns the generated answer

### Usage

```bash
cd rag_from_scratch
python rag_chatbot.py
Type your question, and the chatbot will answer based on the context. Type exit to quit.

2. RAG with LangChain (rag_langchain)
This version uses LangChain to simplify RAG implementation. It uses:

langchain_community.document_loaders.TextLoader to load the text

langchain_text_splitters.RecursiveCharacterTextSplitter to split text

langchain_community.vectorstores.FAISS as a vector database

langchain_huggingface.HuggingFaceEmbeddings for embeddings

langchain_ollama.ChatOllama to interact with tinyllama

How it works
Load and split Instructions.txt into chunks

Create embeddings and store them in FAISS via LangChain

For a user query:

Retrieve top-k relevant chunks

Build a prompt with the context

Generate an answer using ChatOllama

Usage
cd rag_langchain
python rag_chatbot.py
Type your question, and the chatbot will answer based on the context. Type exit to quit.

Requirements
Python 3.10+

Packages for rag_from_scratch:

pip install numpy faiss-cpu sentence-transformers ollama
Packages for rag_langchain:

pip install langchain langchain_community langchain_huggingface faiss-cpu ollama
Notes
Both implementations rely on the local tinyllama model through Ollama. Make sure you have it installed and set up.

You can customize chunk_size and chunk_overlap for better retrieval results.

This is a minimal example of RAG; you can extend it with more documents or different LLMs.

License
This repository is open-source under the MIT License.


---

Αν θέλεις, μπορώ να σου φτιάξω και μια **shorter, more “GitHub-ready” version** που θα είναι πιο κομψή για το profile σου, χωρίς να χάνει όμως τις σημαντικές λεπτομέρειες. Θέλεις να το κάνω;

τα χρειαζομαι αυτα για το intership που θα στειλω βιογραφικο επισης στειλτο ετοιμο για copy paste

Τέλεια! Σε καταλαβαίνω, θα σου φτιάξω ένα έτοιμο για copy-paste README.md στα αγγλικά, περιγραφικό αλλά καθαρό για το GitHub σου και κατάλληλο να το αναφέρεις στο internship.

# RAG Chatbot

This repository contains two implementations of a **Retrieval-Augmented Generation (RAG) chatbot** in Python. Both allow asking questions based on a text file (`Instructions.txt`) by retrieving relevant chunks and generating answers with a local LLM (`tinyllama` via Ollama).

---

## Project Structure

rag_chatbot/
│
├── rag_from_scratch/ # RAG implementation from scratch
├── rag_langchain/ # RAG implementation using LangChain
└── Instructions.txt # Example text for the chatbot


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
Type your question, and the chatbot will answer based on the context. Type exit to quit.

2. RAG with LangChain (rag_langchain)
Technologies: LangChain, FAISS, HuggingFaceEmbeddings, Ollama

How it works
Load and split Instructions.txt into chunks

Create embeddings and store them in FAISS via LangChain

For a user query:

Retrieve top-k relevant chunks

Build a prompt with the context

Generate an answer using ChatOllama

Usage
cd rag_langchain
python rag_chatbot.py
Type your question, and the chatbot will answer based on the context. Type exit to quit.

Requirements
Python 3.10+

Install packages for rag_from_scratch:

pip install numpy faiss-cpu sentence-transformers ollama
Install packages for rag_langchain:

pip install langchain langchain_community langchain_huggingface faiss-cpu ollama