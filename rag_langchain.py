from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# ------------- load document -------------
loader = TextLoader("Instructions.txt")
documents = loader.load()

# ------------- split text into chunks (like δικό σου split_text) -------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,     # όπως στο δικό σου chunk_size
    chunk_overlap=30    # όπως στο δικό σου overlap
)
docs = splitter.split_documents(documents)

print(f"Δημιουργήθηκαν {len(docs)} chunks.")

# ------------- embeddings -------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------- vector store (FAISS) -------------
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

print(f"FAISS vectorstore έτοιμο με {len(docs)} vectors.")

# ------------- LLM -------------
llm = ChatOllama(model="tinyllama")  # όπως το ollama.chat(model="tinyllama")

# ------------- RAG function -------------
def rag_answer(query: str, top_k=3) -> str:
    # 1. Ανάκτηση top-k chunks
    docs = retriever._get_relevant_documents(query, run_manager=None)[:top_k]

    context = "\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Chunks ---\n")
    for doc in docs:
        print(doc.page_content)
        print("-----")

    # 2. Δημιουργία prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly based only on the context.
"""

    # 3. Κλήση LLM με generate()
    response = llm.generate([[HumanMessage(content=prompt)]])

    return response.generations[0][0].text

# ------------- Chat loop -------------
print("\nRAG chatbot έτοιμο! Γράψε 'exit' για έξοδο.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("Chat ended.")
        break

    answer = rag_answer(user_query)
    print("\n=== AI ===\n")
    print(answer)
    print("\n")
