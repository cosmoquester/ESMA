import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .prompt import DIRECT_QA_PROMPT, META_QA_PROMPT


class RLDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        use_meta: bool = False,
        prompt: str = DIRECT_QA_PROMPT,
    ):
        """Initialize RL dataset.

        Args:
            dataset: TriviaQA dataset having fields:
                - question: Question (string)
                - answers: List of answer aliases (list of strings)
            tokenizer: Tokenizer to tokenize the questions and answers
            max_length: Maximum length of the input text
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.meta_prompt = META_QA_PROMPT if use_meta else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        examples = [
            {
                "role": "user",
                "content": self.prompt.format(question=item["question"]),
            }
        ]
        if self.tokenizer.chat_template is not None:
            examples = self.tokenizer.apply_chat_template(
                examples, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            examples = [example["content"] for example in examples]

        tokens = self.tokenizer(
            examples,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        example = {
            "question_id": item["question_id"],
            "question": item["question"],
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "answers": item["answers"],
        }
        if self.meta_prompt is not None:
            meta_tokens = self.tokenizer(
                self.meta_prompt.format(question=item["question"]),
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            example["meta_input_ids"] = meta_tokens["input_ids"].squeeze(0)
            example["meta_attention_mask"] = meta_tokens["attention_mask"].squeeze(0)
        return example


class SFTDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt: str = DIRECT_QA_PROMPT,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        examples = [
            {
                "role": "user",
                "content": self.prompt.format(question=item["question"]),
            },
            {
                "role": "assistant",
                "content": self.random.choice(item["answers"]),
            },
        ]
        examples = self.tokenizer.apply_chat_template(examples, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(examples, return_tensors="pt", truncation=True, max_length=self.max_length)
        example = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }
        return example

    def sft_collate_fn(self, batch: list[dict]) -> dict:
        """Collate function for SFT training with proper padding and labels."""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        max_len = max(ids.size(0) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - ids.size(0)
            # Left padding
            padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            padded_mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            # Labels: -100 for padding tokens (ignored in loss)
            label = padded_ids.clone()
            label[:pad_len] = -100

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(labels),
        }


def simple_collate_fn(batch: list[dict]) -> list[dict]:
    batched = {
        "question_id": [item["question_id"] for item in batch],
        "input_ids": [item["input_ids"] for item in batch],
        "question": [item["question"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
        "answers": [item["answers"] for item in batch],
    }
    if "meta_input_ids" in batch[0]:
        batched["meta_input_ids"] = pad_sequence(
            [item["meta_input_ids"] for item in batch],
            batch_first=True,
            padding_side="left",
        )
        batched["meta_attention_mask"] = pad_sequence(
            [item["meta_attention_mask"] for item in batch],
            batch_first=True,
            padding_side="left",
        )
    return batched


def pad_collate_fn(batch: list[dict]) -> dict:
    batched = {
        "question_id": [item["question_id"] for item in batch],
        "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_side="left"),
        "question": [item["question"] for item in batch],
        "attention_mask": pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_side="left",
        ),
        "answers": [item["answers"] for item in batch],
    }
    if "meta_input_ids" in batch[0]:
        batched["meta_input_ids"] = pad_sequence(
            [item["meta_input_ids"] for item in batch],
            batch_first=True,
            padding_side="left",
        )
        batched["meta_attention_mask"] = pad_sequence(
            [item["meta_attention_mask"] for item in batch],
            batch_first=True,
            padding_side="left",
        )
    return batched
