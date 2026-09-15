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

    def retrieve(self, query):
        retrieved = self.query_db(query)
        if self.settings.use_reranker:
            selected = self.rerank_on_query(retrieved, query)
        else:
            selected = retrieved[: self.settings.n_rerank]
        return retrieved, selected

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
        if user_question.lower() != "x":
            _, selected = rag.retrieve(user_question)
            prompt = rag.build_context(selected, user_question)
            response = rag.generate_answer(prompt)
            print(response)

            print("Anything else?")
        else:
            break


if __name__ == "__main__":
    main()
