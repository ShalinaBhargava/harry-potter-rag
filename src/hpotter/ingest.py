import re

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hpotter.config import Settings


def reset_collection(client, name):
    try:
        client.delete_collection(name)
    except Exception:
        pass  # first run — nothing to delete
    return client.get_or_create_collection(name)


def insert_into_db(collection, docs, ids, mdata):
    collection.upsert(ids=ids, documents=docs, metadatas=mdata)


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


def create_chunk_from_chapter(chapter, chapter_id, setting):
    chunk_ids = []
    chunk_metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=setting.chunk_size, chunk_overlap=setting.chunk_overlap
    )
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
    settings = Settings()
    chapters, chapter_ids, mdata = create_chapters("resources/harry_potter_1.txt")
    client = chromadb.PersistentClient()
    chapter_collection = reset_collection(client, settings.chroma_collection_chapter)
    chapterwise_chunk_collection = reset_collection(
        client, settings.chroma_collection_chunks
    )
    insert_into_db(chapter_collection, chapters, chapter_ids, mdata)
    for index, chapter in enumerate(chapters):
        chunks, chunk_ids, chunk_metadata = create_chunk_from_chapter(
            chapter, chapter_id=chapter_ids[index], setting=settings
        )
        insert_into_db(chapterwise_chunk_collection, chunks, chunk_ids, chunk_metadata)

    print(
        f"indexed {len(chapters)} chapters, {chapterwise_chunk_collection.count()} chunks"
    )


if __name__ == "__main__":
    main()
