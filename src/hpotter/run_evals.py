from pathlib import Path

from hpotter.config import Settings
from hpotter.evals import load_cases
from hpotter.query import RagPipeline


def main():
    settings = Settings()
    rag = RagPipeline(settings)
    count_p = 0
    total = 0

    cases = load_cases(Path("evals/questions.json"))

    for case in cases:
        if case.category == "unanswerable":
            continue
        total += 1
        docs = rag.query_db(case.question)
        if settings.use_reranker:
            selected = rag.rerank_on_query(docs, case.question)
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
        if hit5:
            count_p += 1
        print(
            f"{'PASS' if hit5 else 'FAIL'}  {settings.n_retrieve}:{int(hit10)} "
            f"{settings.n_rerank}:{int(hit5)}  {case.question}  {flag}"
        )

    print(f"Cases passed: {count_p}/{total} ")


if __name__ == "__main__":
    main()
