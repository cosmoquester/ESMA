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
from meta.dataset import RLDataset, simple_collate_fn
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
parser.add_argument("--split", type=str, default="train", help="Split to load")
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
parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2, help="Number of workers")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()


def simple_reward(output: str, answers: list[str]) -> float:
    for answer in answers:
        if answer in output:
            return 1
    return 0


def single_iteration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    iteration_batch: dict,
    accelerator: Accelerator,
    local_seeds: np.ndarray,
    sigma: float,
    batch_size: int,
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
                max_new_tokens=args.max_new_tokens,
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

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    logger.info("[+] Loading TriviaQA dataset...")
    dataset = load_trivia_qa_rl(split=args.split, num_samples=args.num_samples)
    logger.info(f"[+] Total samples: {len(dataset)}")

    logger.info(f"[+] Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"[+] Tokenized dataset: {len(dataset)}")

    dataset = RLDataset(dataset, tokenizer, max_length=args.max_input_length)
    data_loader = DataLoader(
        dataset,
        batch_size=args.num_data_per_iteration,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
    )
    infinite_loader = itertools.chain.from_iterable(itertools.repeat(data_loader))

    logger.info(f"[+] Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to(accelerator.device)
    logger.info("[+] Loaded model successfully")

    local_population_size = args.population_size // accelerator.num_processes
    population_seed_gen = np.random.RandomState(args.seed + accelerator.process_index)
    for iteration, iteration_batch in enumerate(infinite_loader):
        if iteration >= args.num_iterations:
            break

        logger.info(f"[+] Iteration {iteration + 1} of {args.num_iterations}...")
        local_seeds = population_seed_gen.randint(0, 1000000, local_population_size)
        rewards = single_iteration(
            model,
            tokenizer,
            iteration_batch,
            accelerator,
            local_seeds,
            args.sigma,
            args.batch_size,
        )

        local_seeds_tensor = torch.tensor(local_seeds, dtype=torch.long, device=accelerator.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=accelerator.device)
        all_seeds = accelerator.gather(local_seeds_tensor)
        all_rewards = accelerator.gather(rewards_tensor)
        normalized_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        apply_evolution(
            model,
            all_seeds,
            absolute_scale=args.alpha,
            relative_scales=normalized_rewards,
        )


if __name__ == "__main__":
    main(args)
