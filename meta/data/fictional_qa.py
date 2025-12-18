import os

from datasets import Dataset, load_dataset


def load_fictional_qa(split: str = "validation", num_samples: int | None = None) -> Dataset:
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
    dataset = load_dataset("tomg-group-umd/fictionalqa_reformatted_triviaqa", "trivia_qa_cbqa_ds", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_fictional_qa_rl(
    split: str = "validation", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_fictional_qa(split, num_samples)
    return dataset.map(
        lambda x: {
            "question_id": x["question_id"],
            "question": x["input"][10:-10],
            "answers": [x["answer"]],
        },
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
