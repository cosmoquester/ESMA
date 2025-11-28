from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .prompt import DIRECT_QA_PROMPT


class RLDataset(Dataset):
    def __init__(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512, prompt: str = DIRECT_QA_PROMPT
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        examples = [
            {
                "role": "user",
                "content": self.prompt.format(question=item["question"]),
            }
        ]
        if self.tokenizer.chat_template is not None:
            examples = self.tokenizer.apply_chat_template(examples, tokenize=False, add_generation_prompt=True)
        else:
            examples = [example["content"] for example in examples]

        input_tokens = self.tokenizer(
            examples,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        answers = self.dataset[idx]["answers"]
        return {
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "answers": answers,
        }


def simple_collate_fn(batch: list[dict]) -> list[dict]:
    return {
        "input_ids": [item["input_ids"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
        "answers": [item["answers"] for item in batch],
    }
