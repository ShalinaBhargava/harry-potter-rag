import chromadb
from google import genai
from dotenv import load_dotenv
import os
from sentence_transformers import CrossEncoder, SentenceTransformer
import time

load_dotenv()

start = time.perf_counter()
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
print(f"Model load: {time.perf_counter() - start:.2f}s")


def query_db(collection, query):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return results["documents"][0]


def rerank_on_query(documents, query):
    score_query = []
    for chunk in documents:
        score_query.append((query, chunk))

    scores = model.predict(score_query)
    reranked_documents = list(zip(documents, scores))
    reranked_documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked_documents[:5]]


def build_context(documents, query):

    context = "\n\n".join(documents)
    prompt = (
        "System Instructions:\n"
        "You are a Harry Potter assistant.\nYour task is to answer questions ONLY from the supplied context.\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- If the context is insufficient, reply exactly:\n"
        "I don't know based on the provided context.\n"
        "- Do not make assumptions.\n"
        "- If multiple context passages are relevant, combine them into one answer.\n"
        "- Keep the answer concise.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Query:\n"
        f"{query}"
    )
    return prompt


def generate_answer(prompt):
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


def main():
    client = chromadb.PersistentClient()
    collection = client.get_collection("harry_potter_chapterwise_chunks")
    print(collection.count())
    print("What's your query muggle?")
    while True:
        query = input()
        if query != "x":
            start = time.perf_counter()
            documents = query_db(collection, query)
            print(f"Chroma: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            reranked_documents = rerank_on_query(documents, query)
            print(f"Rerank: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            prompt = build_context(reranked_documents, query)
            print(f"Prompt: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            response = generate_answer(prompt)
            print(response)
            print(f"Gemini: {time.perf_counter() - start:.3f}s")

            print("Anything else?")
        else:
            break


if __name__ == "__main__":
    main()
