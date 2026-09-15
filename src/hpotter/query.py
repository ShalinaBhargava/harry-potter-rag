import time

import chromadb
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer

from hpotter.config import Settings


class RagPipeline:
    def __init__(self, settings):
        self.settings = settings
        self.reranker = CrossEncoder(settings.reranker_model)
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.gemini_client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value()
        )
        chroma_client = chromadb.PersistentClient()
        self.collection = chroma_client.get_collection(settings.chroma_collection_chunks)

    def query_db(self, query):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=self.settings.n_retrieve
        )
        return results["documents"][0]

    def rerank_on_query(self, documents, query):
        score_query = []
        for chunk in documents:
            score_query.append((query, chunk))

        scores = self.reranker.predict(score_query)
        reranked_documents = list(zip(documents, scores, strict=True))
        reranked_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in reranked_documents[: self.settings.n_rerank]]

    def build_context(self, documents, query):
        context = "\n\n".join(documents)
        prompt = (
            "System Instructions:\n"
            "You are a Harry Potter assistant.\nYour task is to answer "
            "questions ONLY from "
            "the supplied context.\n"
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

    def generate_answer(self, prompt):
        response = self.gemini_client.models.generate_content(
            model=self.settings.gemini_model, contents=prompt
        )
        return response.text


def main():
    settings = Settings()
    rag = RagPipeline(settings)
    print("What's your query muggle?")
    while True:
        user_question = input()
        if user_question != "x" or user_question != "X":
            start = time.perf_counter()
            documents = rag.query_db(user_question)
            print(f"Chroma: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            reranked_documents = rag.rerank_on_query(documents, user_question)
            print(f"Rerank: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            prompt = rag.build_context(reranked_documents, user_question)
            print(f"Prompt: {time.perf_counter() - start:.3f}s")
            start = time.perf_counter()
            response = rag.generate_answer(prompt)
            print(response)
            print(f"Gemini: {time.perf_counter() - start:.3f}s")

            print("Anything else?")
        else:
            break


if __name__ == "__main__":
    main()
