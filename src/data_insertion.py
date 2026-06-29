import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunk(filename):
    ids = []
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")

    return chunks, ids


def insert_into_db(collection, docs, ids):
    collection.add(ids=ids, documents=docs)


def main():
    chunks, ids = create_chunk("resources/harry_potter_1.txt")
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("harry_potter")
    insert_into_db(collection, chunks, ids)


if __name__ == "__main__":
    main()
