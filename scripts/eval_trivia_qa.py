import argparse
import csv
import random
import re
import string

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_exact_match(prediction, ground_truths):
    norm_pred = normalize_answer(prediction)
    for truth in ground_truths:
        if normalize_answer(truth) in norm_pred:
            return True
    return False


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Using seed: {args.seed}")

    print("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "rc", split="validation")

    if args.num_samples > 0:
        dataset = dataset.select(range(min(len(dataset), args.num_samples)))

    print(f"Total samples to evaluate: {len(dataset)}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    correct_count = 0
    total_count = 0

    batch_data = {"ids": [], "raw_questions": [], "prompts": [], "ground_truths": []}

    if args.output_path is None:
        base_model = args.model.split("/")[-1]
        args.output_path = f"outputs/triviaqa_result_{base_model}_{args.num_samples}.tsv"

    with open(args.output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["example_id", "question", "answer", "ground_truths", "correctness"])

        for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
            qid = example["question_id"]
            question = example["question"]
            ground_truths = example["answer"]["aliases"]

            if tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"Question: {question}\nAnswer:"

            batch_data["ids"].append(qid)
            batch_data["raw_questions"].append(question)
            batch_data["prompts"].append(prompt)
            batch_data["ground_truths"].append(ground_truths)

            if len(batch_data["prompts"]) == args.batch_size or i == len(dataset) - 1:
                inputs = tokenizer(
                    batch_data["prompts"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_output_length,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                for j, pred in enumerate(decoded_preds):
                    current_id = batch_data["ids"][j]
                    current_q = batch_data["raw_questions"][j]
                    current_truths = batch_data["ground_truths"][j]

                    pred_clean = pred.strip().split("\n")[0]

                    is_correct = is_exact_match(pred_clean, current_truths)
                    if is_correct:
                        correct_count += 1
                    total_count += 1

                    writer.writerow([current_id, current_q, pred_clean, current_truths, int(is_correct)])

                batch_data = {"ids": [], "raw_questions": [], "prompts": [], "ground_truths": []}

    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "=" * 40)
    print(f"Model: {args.model}")
    print(f"Total Samples: {total_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {args.output_path}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM on TriviaQA and save to TSV")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--output-path", type=str, help="Output TSV file path")
    parser.add_argument("--max-input-length", type=int, default=2048, help="Maximum length of the input text")
    parser.add_argument("--max-output-length", type=int, default=32, help="Maximum length of the output text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
