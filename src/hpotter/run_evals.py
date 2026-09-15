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
        retrieved, selected = rag.retrieve(case.question)

        hit_retrieved = any(
            all(kw.lower() in c.lower() for kw in case.expected_keywords)
            for c in retrieved
        )
        hit_selected = any(
            all(kw.lower() in c.lower() for kw in case.expected_keywords)
            for c in selected
        )

        if not hit_selected:
            present = {
                kw: any(kw.lower() in c.lower() for c in retrieved)
                for kw in case.expected_keywords
            }
            print(f"      keywords in top-{settings.n_retrieve}:", present)

        flag = (
            f"In top {settings.n_retrieve} but not top {settings.n_rerank}"
            if (hit_retrieved and not hit_selected)
            else ""
        )
        if hit_selected:
            count_p += 1
        print(
            f"{'PASS' if hit_selected else 'FAIL'}  {settings.n_retrieve}:"
            f"{int(hit_retrieved)} {settings.n_rerank}:{int(hit_selected)} "
            f"{case.question} {flag}"
        )

    print(f"Cases passed: {count_p}/{total} ")


if __name__ == "__main__":
    main()
