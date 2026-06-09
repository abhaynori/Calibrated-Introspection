from __future__ import annotations

from .schema import BaseQuestion

DEMO_QUESTIONS: list[BaseQuestion] = [
    BaseQuestion("demo-0", "demo", "geography",
                 "What is the capital of Australia?",
                 ["Sydney", "Melbourne", "Canberra", "Perth"], "C"),
    BaseQuestion("demo-1", "demo", "math",
                 "What is 7 multiplied by 8?",
                 ["54", "56", "63", "48"], "B"),
    BaseQuestion("demo-2", "demo", "biology",
                 "Which organelle is the powerhouse of the cell?",
                 ["Nucleus", "Ribosome", "Mitochondrion", "Golgi apparatus"], "C"),
    BaseQuestion("demo-3", "demo", "chemistry",
                 "What is the chemical symbol for sodium?",
                 ["S", "So", "Na", "Sd"], "C"),
    BaseQuestion("demo-4", "demo", "history",
                 "In which year did World War II end?",
                 ["1943", "1945", "1948", "1939"], "B"),
    BaseQuestion("demo-5", "demo", "physics",
                 "What is the SI unit of force?",
                 ["Joule", "Watt", "Newton", "Pascal"], "C"),
    BaseQuestion("demo-6", "demo", "literature",
                 "Who wrote 'Pride and Prejudice'?",
                 ["Charlotte Bronte", "Jane Austen", "Emily Dickinson", "Mary Shelley"], "B"),
    BaseQuestion("demo-7", "demo", "cs",
                 "What data structure uses FIFO ordering?",
                 ["Stack", "Queue", "Tree", "Heap"], "B"),
    BaseQuestion("demo-8", "demo", "math",
                 "What is the derivative of x^2 with respect to x?",
                 ["x", "2x", "x^2", "2"], "B"),
    BaseQuestion("demo-9", "demo", "geography",
                 "Which is the longest river in the world?",
                 ["Amazon", "Nile", "Yangtze", "Mississippi"], "B"),
]


def load_demo() -> list[BaseQuestion]:
    return list(DEMO_QUESTIONS)


def load_mmlu_redux(subjects: list[str] | None = None, limit: int | None = None) -> list[BaseQuestion]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("MMLU-Redux loader needs `datasets`: pip install datasets") from e

    ds = load_dataset("edinburgh-dawg/mmlu-redux", "all", split="test")
    out: list[BaseQuestion] = []
    for i, row in enumerate(ds):
        if subjects and row.get("subject") not in subjects:
            continue
        opts = row["choices"]
        gold = chr(ord("A") + int(row["answer"]))
        out.append(BaseQuestion(f"mmlu-{i}", "mmlu", row.get("subject", "?"),
                                row["question"], list(opts), gold))
        if limit and len(out) >= limit:
            break
    return out


def load_gpqa_diamond(limit: int | None = None) -> list[BaseQuestion]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("GPQA loader needs `datasets`: pip install datasets") from e
    import random as _r

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out: list[BaseQuestion] = []
    for i, row in enumerate(ds):
        rng = _r.Random(i)
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        gold = chr(ord("A") + order.index(0))
        out.append(BaseQuestion(f"gpqa-{i}", "gpqa", "science",
                                row["Question"], shuffled, gold))
        if limit and len(out) >= limit:
            break
    return out
