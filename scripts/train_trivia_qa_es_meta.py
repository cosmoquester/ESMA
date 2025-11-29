import argparse
import itertools
import logging
import os

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from meta.data import load_trivia_qa_rl
from meta.dataset import RLDataset, pad_collate_fn, simple_collate_fn
from meta.evolution import apply_evolution
from meta.metric import IGNORE_VALUE, meta_metrics
from meta.utils import get_logger, seed_everything

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Train TriviaQA ES")
parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    help="HuggingFace Model ID",
)
parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum input length")
parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens")
parser.add_argument("--sigma", type=float, default=1e-3, help="Sigma")
parser.add_argument("--alpha", type=float, default=5e-4, help="Alpha")
parser.add_argument("--num-iterations", type=int, default=300, help="Number of iterations")
parser.add_argument("--population-size", type=int, default=32, help="Population size")
parser.add_argument(
    "--num-data-per-iteration",
    "-n",
    type=int,
    default=256,
    help="Number of data per iteration",
)
parser.add_argument("--num-samples", type=int, help="Number of samples to load")
parser.add_argument("--num-val-samples", type=int, help="Number of samples to load for validation")
parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--output-dir", type=str, help="Output directory")
parser.add_argument("--model-save-interval", type=int, default=60, help="Model save interval")
parser.add_argument("--evaluate-interval", type=int, default=60, help="Evaluate interval")
parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")
parser.add_argument("--wandb-project", type=str, default="meta-cognition", help="Wandb project")
parser.add_argument("--wandb-entity", type=str, default="cosmoquester", help="Wandb entity")


def multilevel_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(3)
            else:
                rewards.append(2)
        else:
            if correct:
                rewards.append(1)
            else:
                rewards.append(0)
    return rewards


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    max_new_tokens: int,
) -> dict[str, torch.Tensor]:
    all_direct_correctness = []
    all_yes = []
    all_yes_failures = []
    all_no_failures = []
    all_meta_alignments = []
    all_rewards = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        meta_input_ids = batch["meta_input_ids"].to(model.device)
        meta_attention_mask = batch["meta_attention_mask"].to(model.device)
        meta_outputs = model.generate(
            input_ids=meta_input_ids,
            attention_mask=meta_attention_mask,
            max_new_tokens=max_new_tokens,
        )
        meta_generated_tokens = meta_outputs[:, meta_input_ids.shape[1] :]
        meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

        direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
            decoded_outputs, meta_decoded_outputs, batch["answers"], keep_length=True
        )

        all_direct_correctness.extend(direct_correctness)
        all_yes.extend(yes)
        all_yes_failures.extend(yes_failures)
        all_no_failures.extend(no_failures)
        all_meta_alignments.extend(meta_alignments)
        all_rewards.extend(multilevel_reward(direct_correctness, yes))
    return {
        "rewards": torch.tensor(all_rewards, dtype=torch.float32, device=model.device),
        "direct_correctness": torch.tensor(all_direct_correctness, dtype=torch.float32, device=model.device),
        "yes": torch.tensor(all_yes, dtype=torch.float32, device=model.device),
        "yes_failures": torch.tensor(all_yes_failures, dtype=torch.float32, device=model.device),
        "no_failures": torch.tensor(all_no_failures, dtype=torch.float32, device=model.device),
        "meta_alignments": torch.tensor(all_meta_alignments, dtype=torch.float32, device=model.device),
    }


def single_iteration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    iteration_batch: dict,
    accelerator: Accelerator,
    local_seeds: np.ndarray,
    sigma: float,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    all_direct_correctness = []
    all_yes = []
    all_yes_failures = []
    all_no_failures = []
    all_meta_alignments = []
    rewards = []
    for seed in local_seeds:
        apply_evolution(model, seed, absolute_scale=sigma)

        seed_rewards = []
        for i in range(0, len(iteration_batch["answers"]), batch_size):
            batch = {k: v[i : i + batch_size] for k, v in iteration_batch.items()}
            batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_side="left").to(
                accelerator.device
            )
            batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_side="left").to(
                accelerator.device
            )

            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            generated_tokens = outputs[:, batch["input_ids"].shape[1] :]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            batch["meta_input_ids"] = pad_sequence(batch["meta_input_ids"], batch_first=True, padding_side="left").to(
                accelerator.device
            )
            batch["meta_attention_mask"] = pad_sequence(
                batch["meta_attention_mask"], batch_first=True, padding_side="left"
            ).to(accelerator.device)
            meta_outputs = model.generate(
                input_ids=batch["meta_input_ids"],
                attention_mask=batch["meta_attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            meta_generated_tokens = meta_outputs[:, batch["meta_input_ids"].shape[1] :]
            meta_decoded_outputs = tokenizer.batch_decode(meta_generated_tokens, skip_special_tokens=True)

            direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
                decoded_outputs, meta_decoded_outputs, batch["answers"], keep_length=True
            )
            all_direct_correctness.extend(direct_correctness)
            all_yes.extend(yes)
            all_yes_failures.extend(yes_failures)
            all_no_failures.extend(no_failures)
            all_meta_alignments.extend(meta_alignments)
            seed_rewards.extend(multilevel_reward(direct_correctness, yes))
        rewards.append(np.mean(seed_rewards))
        apply_evolution(model, seed, absolute_scale=sigma, reverse=True)
    return rewards, {
        "direct_correctness": torch.tensor(all_direct_correctness, dtype=torch.float32, device=accelerator.device),
        "yes": torch.tensor(all_yes, dtype=torch.float32, device=accelerator.device),
        "yes_failures": torch.tensor(all_yes_failures, dtype=torch.float32, device=accelerator.device),
        "no_failures": torch.tensor(all_no_failures, dtype=torch.float32, device=accelerator.device),
        "meta_alignments": torch.tensor(all_meta_alignments, dtype=torch.float32, device=accelerator.device),
    }


def main(args):
    accelerator = Accelerator()
    logger = get_logger(__name__)

    logger.info(f"[+] Accelerator device: {accelerator.device}")
    if not accelerator.is_main_process:
        logger.setLevel(logging.CRITICAL)
    logger.info(f"[+] Accelerator num_processes: {accelerator.num_processes}")

    if args.output_dir is not None and args.wandb_run_name is None:
        args.wandb_run_name = os.path.basename(args.output_dir)
    if args.output_dir is None and args.wandb_run_name is not None:
        args.output_dir = os.path.join("outputs", args.wandb_run_name)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"[+] Output directory: {args.output_dir}")
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    else:
        checkpoint_dir = None
    if args.wandb_run_name is not None and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.output_dir,
        )
        run.config.update(vars(args))
    else:
        run = None

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    logger.info("[+] Loading TriviaQA dataset...")
    train_data = load_trivia_qa_rl(split="train", num_samples=args.num_samples)
    val_data = load_trivia_qa_rl(split="validation", num_samples=args.num_val_samples)
    logger.info(f"[+] Total samples: {len(train_data)}")

    logger.info(f"[+] Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"[+] Tokenized dataset: {len(train_data)}")

    train_dataset = RLDataset(
        train_data,
        tokenizer,
        max_length=args.max_input_length,
        use_meta=True,
    )
    val_dataset = RLDataset(
        val_data,
        tokenizer,
        max_length=args.max_input_length,
        use_meta=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.num_data_per_iteration,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    infinite_loader = itertools.chain.from_iterable(itertools.repeat(train_loader))
    val_loader = accelerator.prepare(val_loader)

    logger.info(f"[+] Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    model.eval()
    model.to(accelerator.device)
    logger.info("[+] Loaded model successfully")

    local_population_size = args.population_size // accelerator.num_processes
    population_seed_gen = np.random.RandomState(args.seed + accelerator.process_index)
    logger.info("Starting training...")
    for iteration, iteration_batch in enumerate(infinite_loader, start=1):
        if iteration >= args.num_iterations:
            break

        local_seeds = population_seed_gen.randint(0, 1000000, local_population_size)
        rewards, metrics = single_iteration(
            model,
            tokenizer,
            iteration_batch,
            accelerator,
            local_seeds,
            args.sigma,
            args.batch_size,
            args.max_new_tokens,
        )

        local_seeds_tensor = torch.tensor(local_seeds, dtype=torch.long, device=accelerator.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
        all_seeds = accelerator.gather(local_seeds_tensor)
        all_rewards = accelerator.gather(rewards_tensor)
        all_metrics = {"train/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
        all_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_metrics.items()}
        if accelerator.is_main_process:
            avg_reward = all_rewards.mean().item()
            all_metrics["train/rewards"] = avg_reward
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()])
            logger.info(f"[+] Iteration {iteration:03d} {metric_str}")
            if run is not None:
                run.log(all_metrics, step=iteration)

            if iteration % args.model_save_interval == 0 and checkpoint_dir is not None:
                model.save_pretrained(os.path.join(checkpoint_dir, f"iteration_{iteration:03d}"))
                tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"iteration_{iteration:03d}"))
                logger.info(f"[+] Model saved to {os.path.join(checkpoint_dir, f'iteration_{iteration:03d}')}")
        normalized_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        apply_evolution(
            model,
            all_seeds,
            absolute_scale=args.alpha,
            relative_scales=normalized_rewards,
        )

        if iteration % args.evaluate_interval == 0:
            metrics = evaluate_model(model, tokenizer, val_loader, args.max_new_tokens)
            all_val_metrics = {"val/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
            all_val_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_val_metrics.items()}
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_val_metrics.items()])
            logger.info(f"[+] Validation Iteration {iteration:03d} {metric_str}")
            if run is not None and accelerator.is_main_process:
                run.log(all_val_metrics, step=iteration)

    test_data = load_trivia_qa_rl(split="test", num_samples=100)
    test_dataset = RLDataset(
        test_data,
        tokenizer,
        max_length=args.max_input_length,
        use_meta=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    test_loader = accelerator.prepare(test_loader)
    test_metrics = evaluate_model(model, tokenizer, test_loader, args.max_new_tokens)
    all_test_metrics = {"test/" + k: accelerator.gather(v) for k, v in sorted(test_metrics.items())}
    all_test_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_test_metrics.items()}
    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in all_test_metrics.items()])
    logger.info(f"[+] Test {metric_str}")
    if run is not None and accelerator.is_main_process:
        run.log(all_test_metrics, step=args.num_iterations)


if __name__ == "__main__":
    main(parser.parse_args())
