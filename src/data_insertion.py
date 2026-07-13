import chromadb
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def insert_into_db(collection, docs, ids, mdata):
    collection.add(ids=ids, documents=docs, metadatas=mdata)


def create_chapters(filename):
    mdatas = []
    chapters = []
    chapter_ids = []
    with open(filename, encoding="utf-8") as file:
        text = file.read()

    split_chapters = list(re.finditer(r"CHAPTER \w+", text))
    for chapter, split in enumerate(split_chapters):
        next_chapter = chapter + 1
        chapter_ids.append(str(next_chapter))
        if next_chapter < len(split_chapters):
            next_split = split_chapters[next_chapter]
            extracted_chapter = text[split.start() : next_split.start()].strip()
        else:
            extracted_chapter = text[split.start() :].strip()
        chapters.append(extracted_chapter)
        mdatas.append(
            {"Chapter": next_chapter, "Character Count": len(extracted_chapter)}
        )
    return chapters, chapter_ids, mdatas


def create_chunk_from_chapter(chapter, chapter_id):
    chunk_ids = []
    chunk_metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(chapter)
    for i, chunk in enumerate(chunks):
        chunk_id = f"chapter_{chapter_id}_chunk_{i}"
        chunk_ids.append(chunk_id)
        chunk_metadatas.append(
            {
                "Chunk ID": chunk_id,
                "Character_count": len(chunk),
                "Chapter ID": chapter_id,
            }
        )

    return chunks, chunk_ids, chunk_metadatas


def main():
    chapters, chapter_ids, mdata = create_chapters("resources/harry_potter_1.txt")
    client = chromadb.PersistentClient()
    chapter_collection = client.get_or_create_collection("harry_potter_chapterwise")
    insert_into_db(chapter_collection, chapters, chapter_ids, mdata)
    chapterwise_chunk_collection = client.get_or_create_collection(
        "harry_potter_chapterwise_chunks"
    )
    print(chapter_collection)
    for index, chapter in enumerate(chapters):
        chunks, chunk_ids, chunk_metadata = create_chunk_from_chapter(
            chapter, chapter_id=chapter_ids[index]
        )
        insert_into_db(chapterwise_chunk_collection, chunks, chunk_ids, chunk_metadata)


if __name__ == "__main__":
    main()
