def main():
    with open("resources/harry_potter_1.txt", encoding="utf-8") as file:
        text = file.read()

    chunk_size = 1000
    overlap = 200
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])

    print(len(chunks))
    print(chunks[0])
    print("Break")
    print(chunks[1])


if __name__ == "__main__":
    main()
