import os

from datasets import Dataset, load_dataset


def load_boolq(split: str = "validation", num_samples: int | None = None) -> Dataset:
    """Load BoolQ dataset.

    Args:
        split: Split to load (validation, test, train)
        num_samples: Number of samples to load

    Returns:
        Dataset: BoolQ dataset
    """
    dataset = load_dataset("boolq", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_boolq_rl(
    split: str = "validation",
    num_samples: int | None = None,
    num_proc: int | None = None,
) -> Dataset:
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_boolq(split, num_samples)
    return dataset.map(
        lambda x, idx: {
            "question_id": idx,
            "question": x["question"],
            "answers": [str(x["answer"])],
        },
        num_proc=num_proc,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
