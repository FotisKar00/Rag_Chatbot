import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama

with open("Instructions.txt", "r") as f:
    text = f.read()


def split_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

chunks = split_text(text)

print(f"Δημιουργήθηκαν {len(chunks)} chunks.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.array(
    [embedding_model.encode(chunk) for chunk in chunks]
).astype("float32")


dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(embeddings)

print(f"FAISS index έτοιμο με {index.ntotal} vectors.")

def rag_answer(query, top_k=3):

    query_vec = np.array(
        [embedding_model.encode(query)]
    ).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    context = "\n".join([chunks[i] for i in indices[0]])

    print("\n--- Retrieved Chunks ---\n")

    for i in indices[0]:
        print(chunks[i])
        print("-----")

    
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly based only on the context.
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


print("\nRAG chatbot έτοιμο!")
print("Γράψε 'exit' για έξοδο.\n")

while True:

    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("Chat ended.")
        break

    answer = rag_answer(user_query)

    print("\n=== AI ===\n")
    print(answer)
    print("\n")