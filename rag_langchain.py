from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

loader = TextLoader("Instructions.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,    
    chunk_overlap=30  
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model="tinyllama") 


def rag_answer(query: str, top_k=3) -> str:
    relevant_chokes = retriever._get_relevant_documents(query, run_manager=None)[:top_k]

    context = "\n".join([doc.page_content for doc in relevant_chokes])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly based only on the context.
"""
    response = llm.generate([[HumanMessage(content=prompt)]])

    return response.generations[0][0].text

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
