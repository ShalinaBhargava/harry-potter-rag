import chromadb


def create_chunk(filename, chunk_size, overlap):
    chunks, ids = [], []

    with open(filename, encoding="utf-8") as file:
        text = file.read()

    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])

    for i in range(0, len(chunks)):
        ids.append("chunk_" + str(i))

    return ids, chunks


def insert_into_db(collection, docs, ids):
    collection.add(ids=ids, documents=docs)


def query_db(collection, query):
    results = collection.query(query_texts=[query], n_results=2)
    return results


def main():
    ids, chunks = create_chunk("resources/harry_potter_1.txt", 1000, 200)

    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("harry_potter")
    # insert_into_db(collection, chunks, ids)
    print("What's your query muggle?")
    while True:
        query = input()
        if query != "x":
            results = query_db(collection, query)
            print(results)
            print("Anything else?")
        else:
            break


if __name__ == "__main__":
    main()
