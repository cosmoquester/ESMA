import argparse
import itertools
import logging
import os

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from meta.data import load_trivia_qa_rl
from meta.dataset import RLDataset, pad_collate_fn, simple_collate_fn
from meta.evolution import apply_evolution
from meta.utils import get_logger, seed_everything

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Train TriviaQA ES")
parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-0.5B-Instruct",
    help="HuggingFace Model ID",
)
parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
parser.add_argument("--max-input-length", type=int, default=128, help="Maximum input length")
parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens")
parser.add_argument("--sigma", type=float, default=1e-3, help="Sigma")
parser.add_argument("--alpha", type=float, default=5e-4, help="Alpha")
parser.add_argument("--num-iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--population-size", type=int, default=32, help="Population size")
parser.add_argument(
    "--num-data-per-iteration",
    "-n",
    type=int,
    default=1000,
    help="Number of data per iteration",
)
parser.add_argument("--num-samples", type=int, help="Number of samples to load")
parser.add_argument("--num-val-samples", type=int, help="Number of samples to load for validation")
parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--output-dir", type=str, help="Output directory")
parser.add_argument("--model-save-interval", type=int, default=50, help="Model save interval")
parser.add_argument("--evaluate-interval", type=int, default=50, help="Evaluate interval")
parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")
parser.add_argument("--wandb-project", type=str, default="meta-cognition", help="Wandb project")
parser.add_argument("--wandb-entity", type=str, default="cosmoquester", help="Wandb entity")


def simple_reward(output: str, answers: list[str]) -> float:
    for answer in answers:
        if answer in output:
            return 1
    return 0


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    max_new_tokens: int,
) -> float:
    rewards = []
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
        batch_answers = batch["answers"]
        for decoded_output, answers in zip(decoded_outputs, batch_answers):
            rewards.append(simple_reward(decoded_output, answers))
    return torch.tensor(rewards, dtype=torch.float32, device=model.device)


def single_iteration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    iteration_batch: dict,
    accelerator: Accelerator,
    local_seeds: np.ndarray,
    sigma: float,
    batch_size: int,
    max_new_tokens: int,
) -> list[float]:
    rewards = []
    for seed in local_seeds:
        apply_evolution(model, seed, absolute_scale=sigma)

        seed_rewards = []
        for i in range(0, len(iteration_batch), batch_size):
            input_ids = pad_sequence(
                iteration_batch["input_ids"][i : i + batch_size],
                batch_first=True,
                padding_side="left",
            ).to(accelerator.device)
            attention_mask = pad_sequence(
                iteration_batch["attention_mask"][i : i + batch_size],
                batch_first=True,
                padding_side="left",
            ).to(accelerator.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
            generated_tokens = outputs[:, input_ids.shape[1] :]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            batch_answers = iteration_batch["answers"][i : i + batch_size]
            for decoded_output, answers in zip(decoded_outputs, batch_answers):
                seed_rewards.append(simple_reward(decoded_output, answers))
        rewards.append(np.mean(seed_rewards))
        apply_evolution(model, seed, absolute_scale=sigma, reverse=True)
    return rewards


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
    if args.wandb_run_name is not None:
        import wandb

        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )
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

    train_dataset = RLDataset(train_data, tokenizer, max_length=args.max_input_length)
    val_dataset = RLDataset(val_data, tokenizer, max_length=args.max_input_length)
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
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to(accelerator.device)
    logger.info("[+] Loaded model successfully")

    local_population_size = args.population_size // accelerator.num_processes
    population_seed_gen = np.random.RandomState(args.seed + accelerator.process_index)
    logger.info("Starting training...")
    for iteration, iteration_batch in enumerate(infinite_loader):
        if iteration >= args.num_iterations:
            break

        local_seeds = population_seed_gen.randint(0, 1000000, local_population_size)
        rewards = single_iteration(
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
        if accelerator.is_main_process:
            avg_reward = all_rewards.mean().item()
            logger.info(f"[+] Iteration {iteration + 1:03d} Rewards: {avg_reward:.4f}")
            if run is not None:
                run.log({"rewards": avg_reward})

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
            val_rewards = evaluate_model(model, tokenizer, val_loader, args.max_new_tokens)
            all_val_rewards = accelerator.gather(val_rewards)
            avg_val_reward = all_val_rewards.mean().item()
            logger.info(f"[+] Iteration {iteration + 1:03d} Val Rewards: {avg_val_reward:.4f}")
            if run is not None and accelerator.is_main_process:
                run.log({"val_rewards": avg_val_reward})


if __name__ == "__main__":
    main(parser.parse_args())
