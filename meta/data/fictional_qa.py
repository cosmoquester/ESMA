import os

from datasets import Dataset, load_dataset


def load_fictional_qa(num_samples: int | None = None) -> Dataset:
    """Load FictionalQA dataset.

    Args:
        split: Split to load (validation, test, train)
        num_samples: Number of samples to load

    Returns:
        Dataset: FictionalQA dataset
            fields:
                - question_id: Question ID (string)
                - question: Question (string)
                - answer: Dictionary with answers
                    - aliases: List of answer aliases (list of strings)
                    - normalized_aliases: Normalized answer (string)
                    - matched_wiki_entity_name: Matched wiki entity name (string)
                    - normalized_matched_wiki_entity_name: Normalized matched wiki entity name (string)
                    - normalized_value: Normalized value (string)
                    - value: Value (string)
                    - type: Type of the answer
    """
    dataset = load_dataset("tomg-group-umd/fictionalqa", "fict_qa", split="train")
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_fictional_qa_rl(
    split: str = "train", num_samples: int | None = None, num_proc: int | None = None, seed: int = 42
) -> Dataset:
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_fictional_qa(num_samples)
    dataset = dataset.map(
        lambda x: {
            "question_id": x["question_id"],
            "question": x["question"],
            "answers": [x["natural_answer"]],
        },
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    split_dataset = dataset.train_test_split(test_size=0.5, seed=seed, shuffle=True)

    if split == "train":
        return split_dataset["train"]
    elif split == "validation":
        return split_dataset["test"]
    elif split == "all":
        return dataset
    else:
        raise ValueError(f"Invalid split: {split}")
