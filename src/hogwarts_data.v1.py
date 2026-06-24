import chromadb


def main():
    client = chromadb.PersistentClient()

    collection = client.get_or_create_collection(name="harry_potter")

    collection.add(
        ids=["id1", "id2", "id3"],
        documents=["Harry met Hagrid", "Hermione loves studying", "Voldemort returned"],
    )
    results = collection.query(
        query_texts=["Who is hagrid"],  # Chroma will embed this for you
        n_results=2
    )
    print(results)


if __name__ == "__main__":
    main()
