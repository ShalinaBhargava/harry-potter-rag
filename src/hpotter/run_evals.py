import chromadb
from hpotter.query import query_db, rerank_on_query
from hpotter.config import Settings
from hpotter.evals import load_cases
from pathlib import Path


def main():
    settings = Settings()
    client = chromadb.PersistentClient()
    collection = client.get_collection(settings.chroma_collection_chunks)

    cases = load_cases(Path("evals/questions.json"))

    for case in cases:
        if case.category == "unanswerable":
            continue

        docs = query_db(collection, case.question, settings)
        if settings.use_reranker:
            selected = rerank_on_query(docs, case.question, settings)
        else:
            selected = docs[: settings.n_rerank]
        hit10 = any(
            all(kw.lower() in c.lower() for kw in case.expected_keywords) for c in docs
        )
        hit5 = any(
            all(kw.lower() in c.lower() for kw in case.expected_keywords)
            for c in selected
        )

        if not hit5:
            present = {
                kw: any(kw.lower() in c.lower() for c in docs)
                for kw in case.expected_keywords
            }
            print(f"      keywords in top-{settings.n_retrieve}:", present)

        flag = (
            f"In top {settings.n_retrieve} but not top {settings.n_rerank}"
            if (hit10 and not hit5)
            else ""
        )
        print(
            f"{'PASS' if hit5 else 'FAIL'}  {settings.n_retrieve}:{int(hit10)} {settings.n_rerank}:{int(hit5)}  {case.question}  {flag}"
        )


if __name__ == "__main__":
    main()
