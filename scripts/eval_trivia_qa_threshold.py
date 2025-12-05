import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from meta.data import load_trivia_qa_rl
from meta.dataset import RLDataset, pad_collate_fn
from meta.metric import IGNORE_VALUE, meta_metrics
from meta.utils import get_logger, seed_everything

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Evaluate LLM on TriviaQA and save to TSV")
parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    help="HuggingFace Model ID",
)
parser.add_argument("--split", type=str, default="validation", help="Split to evaluate")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate (0 for all)")
parser.add_argument("--output-path", type=str, help="Output TSV file path")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum length of the input text")
parser.add_argument(
    "--max-output-length",
    type=int,
    default=32,
    help="Maximum length of the output text",
)
parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")


def main(args):
    logger = get_logger(__name__)  # noqa: F821

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    logger.info(f"[+] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    yes_token_id = tokenizer.vocab["Yes"]
    no_token_id = tokenizer.vocab["No"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    logger.info("[+] Loading TriviaQA dataset...")
    data = load_trivia_qa_rl(split=args.split, num_samples=args.num_samples)
    dataset = RLDataset(data, tokenizer, max_length=args.max_input_length, use_meta=True)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_fn,
    )
    logger.info(f"[+] Total samples to evaluate: {len(data)}")

    if args.output_path is None:
        base_model = args.model.split("/")[-1]
        os.makedirs("eval_outputs", exist_ok=True)
        args.output_path = f"eval_outputs/triviaqa_{base_model}_{args.split}_{args.num_samples}_threshold.tsv"

    all_question_ids = []
    all_questions = []
    all_ground_truths = []
    all_predictions = []
    all_meta_answers = []
    all_direct_correctness = []
    all_yes_over_no = []
    all_yes = []
    all_yes_failures = []
    all_no_failures = []
    all_meta_alignments = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_output_length,
        )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        meta_input_ids = batch["meta_input_ids"].to(model.device)
        meta_attention_mask = batch["meta_attention_mask"].to(model.device)

        model_outputs = model(input_ids=meta_input_ids, attention_mask=meta_attention_mask, logits_to_keep=1)
        last_token_logits = model_outputs.logits[:, -1, :]
        yes_logits = last_token_logits[:, yes_token_id]
        no_logits = last_token_logits[:, no_token_id]
        yes_over_no = (yes_logits - no_logits).exp().cpu().tolist()
        meta_decoded_outputs = ["Yes" if p > 1 else "No" for p in yes_over_no]

        direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
            decoded_outputs,
            meta_decoded_outputs,
            batch["answers"],
            keep_length=True,
        )

        all_question_ids.extend(batch["question_id"])
        all_questions.extend(batch["question"])
        all_ground_truths.extend(batch["answers"])
        all_predictions.extend(decoded_outputs)
        all_meta_answers.extend(meta_decoded_outputs)
        all_direct_correctness.extend(direct_correctness)
        all_yes_over_no.extend(yes_over_no)
        all_yes.extend(yes)
        all_yes_failures.extend(yes_failures)
        all_no_failures.extend(no_failures)
        all_meta_alignments.extend(meta_alignments)

    sorted_results = sorted(zip(all_yes_over_no, all_direct_correctness))
    best_threshold = sorted_results[0][0]
    gain = 0
    for yes_over_no, direct_correctness in sorted_results:
        if direct_correctness:
            if gain >= 0:
                best_threshold = yes_over_no
                gain = -1
            else:
                gain -= 1
        else:
            gain += 1
    if gain > 0:
        best_threshold = sorted_results[-1][0] + 1e-6
    logger.info(f"[+] Best threshold: {best_threshold}")

    best_meta_answer = ["Yes" if p >= best_threshold else "No" for p in all_yes_over_no]
    _, best_yes, best_yes_failures, best_no_failures, best_meta_alignments = meta_metrics(
        all_predictions, best_meta_answer, all_ground_truths, keep_length=True
    )

    with open(args.output_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "question_id",
                "question",
                "ground_truths",
                "prediction",
                "meta_answer",
                "direct_correctness",
                "yes",
                "yes_failures",
                "no_failures",
                "meta_alignments",
                "yes_over_no",
                "best_meta_answer",
                "best_yes",
                "best_yes_failures",
                "best_no_failures",
                "best_meta_alignments",
            ]
        )
        for (
            question_id,
            question,
            ground_truths,
            prediction,
            meta_answer,
            direct_correctness,
            yes,
            yes_failures,
            no_failures,
            meta_alignments,
            yes_over_no,
            _best_meta_answer,
            _best_yes,
            best_yes_failure,
            best_no_failure,
            best_meta_alignment,
        ) in zip(
            all_question_ids,
            all_questions,
            all_ground_truths,
            all_predictions,
            all_meta_answers,
            all_direct_correctness,
            all_yes,
            all_yes_failures,
            all_no_failures,
            all_meta_alignments,
            all_yes_over_no,
            best_meta_answer,
            best_yes,
            best_yes_failures,
            best_no_failures,
            best_meta_alignments,
        ):
            writer.writerow(
                [
                    question_id,
                    question,
                    str(ground_truths),
                    prediction,
                    meta_answer,
                    direct_correctness,
                    yes,
                    yes_failures,
                    no_failures,
                    meta_alignments,
                    yes_over_no,
                    _best_meta_answer,
                    _best_yes,
                    best_yes_failure,
                    best_no_failure,
                    best_meta_alignment,
                ]
            )
    logger.info(f"[+] Results saved to: {args.output_path}")
    logger.info(f"[+] Exact match accuracy: {sum(all_direct_correctness) / len(all_direct_correctness):.2%}")
    logger.info(f"[+] Yes rate: {sum(all_yes) / len(all_yes):.2%}")
    logger.info(f"[+] Meta alignments: {sum(all_meta_alignments) / len(all_meta_alignments):.2%}")

    all_yes_failures = [v for v in all_yes_failures if v != IGNORE_VALUE]
    all_no_failures = [v for v in all_no_failures if v != IGNORE_VALUE]
    if len(all_yes_failures) > 0:
        logger.info(f"[+] Yes failures rate: {sum(all_yes_failures) / len(all_yes_failures):.2%}")
    else:
        logger.info("[-] All meta answers are No")
    if len(all_no_failures) > 0:
        logger.info(f"[+] No failures rate: {sum(all_no_failures) / len(all_no_failures):.2%}")
    else:
        logger.info("[-] All meta answers are Yes")

    best_yes_failures = [v for v in best_yes_failures if v != IGNORE_VALUE]
    best_no_failures = [v for v in best_no_failures if v != IGNORE_VALUE]
    if len(best_yes_failures) > 0:
        logger.info(f"[+] Best yes failures rate: {sum(best_yes_failures) / len(best_yes_failures):.2%}")
    else:
        logger.info("[-] All best meta answers are No")
    if len(best_no_failures) > 0:
        logger.info(f"[+] Best no failures rate: {sum(best_no_failures) / len(best_no_failures):.2%}")
    else:
        logger.info("[-] All best meta answers are Yes")
    logger.info(f"[+] Best meta alignments: {sum(best_meta_alignments) / len(best_meta_alignments):.2%}")
    logger.info(f"[+] Best yes rate: {sum(best_yes) / len(best_yes):.2%}")
    logger.info(f"[+] Best meta alignments: {sum(best_meta_alignments) / len(best_meta_alignments):.2%}")


if __name__ == "__main__":
    main(parser.parse_args())
