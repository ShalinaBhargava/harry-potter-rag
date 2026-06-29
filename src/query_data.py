import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()


def query_db(collection, query):
    results = collection.query(query_texts=[query], n_results=5)
    documents = results["documents"][0]

    return documents


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
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(model="gemini-3.5-flash", input=prompt)
    response = interaction.output_text
    return response


def main():
    client = chromadb.PersistentClient()
    collection = client.get_collection("harry_potter")
    print("What's your query muggle?")
    while True:
        query = input()
        if query != "x":
            documents = query_db(collection, query)
            prompt = build_context(documents, query)
            response = generate_answer(prompt)
            print(response)
            print("Anything else?")
        else:
            break


if __name__ == "__main__":
    main()
