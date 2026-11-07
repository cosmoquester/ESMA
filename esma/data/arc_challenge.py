import os

from datasets import Dataset, load_dataset


def _format_question(question: str, choices: dict) -> str:
    """Embed the multiple-choice options into the question text.

    Args:
        question: Raw question text
        choices: Dictionary with "text" (list of option strings) and "label" (list of labels)

    Returns:
        str: Question text with the labeled options appended
    """
    options = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    return f"{question}\n{options}"


def load_arc_challenge(split: str = "test", num_samples: int | None = None) -> Dataset:
    """Load ARC-Challenge dataset from HuggingFace.

    https://huggingface.co/datasets/allenai/ai2_arc
    ARC (AI2 Reasoning Challenge) is a multiple-choice science question answering
    dataset. The Challenge set contains questions that are answered incorrectly by
    both a retrieval-based and a word co-occurrence algorithm.

    Args:
        split: Split to load (train, validation, test)
        num_samples: Number of samples to load

    Returns:
        Dataset: ARC-Challenge dataset
            fields:
                - id: Question ID (string)
                - question: Question text (string)
                - choices: Dictionary with "text" and "label" lists
                - answerKey: Label of the correct choice (string)
    """
    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=split,
    )
    if num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    return dataset


def load_arc_challenge_meta(
    split: str = "test", num_samples: int | None = None, num_proc: int | None = None
) -> Dataset:
    """Load ARC-Challenge dataset formatted for RL training.

    The multiple-choice options are embedded into the question text and the answer
    is the text of the correct choice.

    Args:
        split: Split to load (train, validation, test)
        num_samples: Number of samples to load
        num_proc: Number of processes for mapping

    Returns:
        Dataset: ARC-Challenge dataset formatted for RL
            fields:
                - question_id: Question ID (string)
                - question: Question text with options (string)
                - answers: List with the correct choice text (list of strings)
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    dataset = load_arc_challenge(split, num_samples)

    def _map(x):
        choices = x["choices"]
        answer_key = x["answerKey"]
        answers = [text for label, text in zip(choices["label"], choices["text"]) if label == answer_key]
        return {
            "question_id": x["id"],
            "question": _format_question(x["question"], choices),
            "answers": answers,
        }

    return dataset.map(
        _map,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
