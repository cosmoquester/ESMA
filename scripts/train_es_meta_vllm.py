import argparse
import itertools
import os
import time

import numpy as np
import ray
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from esma.data import load_trivia_qa_meta
from esma.metric import IGNORE_VALUE, meta_metrics, type2_d_prime
from esma.prompt import DIRECT_QA_PROMPT, META_QA_PROMPT
from esma.reward import REWARD_TYPE_TO_FUNCTION
from esma.utils import format_metrics, get_logger, seed_everything

torch.set_grad_enabled(False)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# fmt: off
parser = argparse.ArgumentParser(description="Train TriviaQA ES-Meta with vLLM")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")

g = parser.add_argument_group("Data")
g.add_argument("--max-input-length", type=int, default=128, help="Maximum input token length (prompt)")
g.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens to generate")
g.add_argument("--num-samples", type=int, help="Number of training samples to load (None = all)")
g.add_argument("--num-val-samples", type=int, help="Number of validation samples to load (None = all)")

g = parser.add_argument_group("Evolution")
g.add_argument("--sigma", type=float, default=1e-3, help="Perturbation scale (sigma)")
g.add_argument("--alpha", type=float, default=5e-4, help="ES learning rate (alpha)")
g.add_argument("--num-iterations", type=int, default=300, help="Number of ES iterations")
g.add_argument("--population-size", type=int, default=32, help="Number of perturbation candidates per iteration")
g.add_argument("--num-data-per-iteration", "-n", type=int, default=256, help="Questions evaluated per iteration")
g.add_argument("--reward-type", type=str, default="esma", choices=list(REWARD_TYPE_TO_FUNCTION.keys()), help="Reward function")

g = parser.add_argument_group("vLLM")
g.add_argument("--batch-size", type=int, default=512, help="Max prompts per llm.generate() call")
g.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "auto"], help="Model weight dtype")
g.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory fraction reserved for vLLM KV cache")
g.add_argument("--max-num-seqs", type=int, default=512, help="Max sequences processed concurrently by vLLM (higher = more throughput, more KV cache)")
g.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
g.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for population parallelism (each GPU evaluates a subset of seeds)")

g = parser.add_argument_group("Experiment")
g.add_argument("--num-workers", type=int, default=(os.cpu_count() // 2), help="DataLoader worker threads")
g.add_argument("--seed", type=int, default=42, help="Global random seed")
g.add_argument("--output-dir", type=str, help="Root output directory for checkpoints and logs")
g.add_argument("--model-save-interval", type=int, default=60, help="Save checkpoint every N iterations")
g.add_argument("--evaluate-interval", type=int, default=60, help="Run validation every N iterations")
g.add_argument("--wandb-run-name", type=str, help="W&B run name (also used to derive --output-dir when omitted)")
g.add_argument("--wandb-project", type=str, default="meta-cognition", help="W&B project")
g.add_argument("--wandb-entity", type=str, default="cosmoquester", help="W&B entity")
# fmt: on


def _apply_evolution(
    worker,
    seeds: list[int] | int,
    absolute_scale: float,
    relative_scales: list[float] = [1.0],
    reverse: bool = False,
) -> None:
    if isinstance(seeds, int):
        seeds = [seeds]
    model = worker.model_runner.model
    param_gem = model.parameters()
    device = next(param_gem).device
    gens = [torch.Generator(device=device) for _ in seeds]
    sign = -1.0 if reverse else 1.0
    for gen, seed in zip(gens, seeds):
        gen.manual_seed(int(seed))

    for p in model.parameters():
        for gen, scale in zip(gens, relative_scales):
            noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
            noise.mul_(sign * scale * absolute_scale / len(seeds))
            p.data.add_(noise)
            del noise
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _save_weights(worker, filepath: str) -> None:
    import gc

    state_dict = {name: p.detach().cpu() for name, p in worker.model_runner.model.named_parameters()}
    torch.save(state_dict, filepath)
    gc.collect()


METRIC_KEYS = (
    "direct_correctness",
    "yes",
    "yes_failures",
    "no_failures",
    "meta_alignments",
)


@ray.remote(num_gpus=1)
class ESWorker:
    def __init__(self, args):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_input_length + args.max_new_tokens,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
        )
        self.sampling_params = SamplingParams(max_tokens=args.max_new_tokens)

    def evaluate_seeds(self, seeds, questions, answers):
        rewards, agg = [], {k: [] for k in METRIC_KEYS}
        for seed in seeds:
            self.llm.collective_rpc(_apply_evolution, args=(int(seed), self.args.sigma))
            r, m = run_inference(
                self.llm,
                self.tokenizer,
                questions,
                answers,
                self.sampling_params,
                self.args.reward_type,
                self.args.batch_size,
                self.args.max_input_length,
            )
            rewards.append(float(np.mean(r)))
            for k in agg:
                agg[k].extend(m[k])
            self.llm.collective_rpc(_apply_evolution, args=(int(seed), self.args.sigma, [1.0], True))
        return rewards, agg

    def apply_update(self, seeds, alpha, normalized_rewards):
        self.llm.collective_rpc(_apply_evolution, args=(seeds, alpha, normalized_rewards))

    def evaluate_full(self, data):
        return evaluate(
            self.llm,
            self.tokenizer,
            data,
            self.sampling_params,
            self.args.reward_type,
            self.args.batch_size,
            self.args.max_input_length,
        )

    def save_weights(self, filepath):
        self.llm.collective_rpc(_save_weights, args=(filepath,))

    def ready(self):
        return True


class RawQADataset(Dataset):
    """Lightweight dataset that returns raw question/answer pairs without tokenisation."""

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        return {"question": item["question"], "answers": item["answers"]}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "questions": [item["question"] for item in batch],
            "answers": [item["answers"] for item in batch],
        }


def build_prompts(
    questions: list[str],
    tokenizer: AutoTokenizer,
    prompt_template: str,
    max_length: int,
) -> list[dict]:
    """Tokenize questions with truncation and return prompt_token_ids dicts for vLLM.

    Tokenizing here (with truncation=True, max_length=max_length) mirrors the
    original MetaDataset behaviour and avoids silently overflowing max_model_len
    when raw text is passed directly to vLLM.
    """
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": prompt_template.format(question=q)}]
        if tokenizer.chat_template is not None and prompt_template == DIRECT_QA_PROMPT:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = prompt_template.format(question=q)
        token_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        prompts.append({"prompt_token_ids": token_ids})
    return prompts


def _safe_mean(values: list, ignore: int = IGNORE_VALUE) -> float:
    filtered = [v for v in values if v != ignore]
    return float(np.mean(filtered)) if filtered else 0.0


def _generate_chunked(llm: LLM, prompts: list[str], sampling_params: SamplingParams, batch_size: int) -> list[str]:
    """Submit prompts to vLLM in chunks of batch_size, mirroring the original script's
    inner sub-batch loop over iteration_batch."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        outputs.extend(o.outputs[0].text for o in llm.generate(chunk, sampling_params, use_tqdm=False))
    return outputs


def run_inference(
    llm: LLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    answers: list[list[str]],
    sampling_params: SamplingParams,
    reward_type: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[float], dict]:
    direct_prompts = build_prompts(questions, tokenizer, DIRECT_QA_PROMPT, max_length)
    meta_prompts = build_prompts(questions, tokenizer, META_QA_PROMPT, max_length)

    # Submit direct + meta as a single batch so vLLM can schedule them together.
    all_outputs = _generate_chunked(llm, direct_prompts + meta_prompts, sampling_params, batch_size)
    direct_outputs = all_outputs[: len(questions)]
    meta_outputs = all_outputs[len(questions) :]

    direct_correctness, yes, yes_failures, no_failures, meta_alignments = meta_metrics(
        direct_outputs, meta_outputs, answers, keep_length=True
    )
    rewards = REWARD_TYPE_TO_FUNCTION[reward_type](direct_correctness, yes)
    return rewards, {
        "direct_correctness": direct_correctness,
        "yes": yes,
        "yes_failures": yes_failures,
        "no_failures": no_failures,
        "meta_alignments": meta_alignments,
    }


def evaluate(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data,
    sampling_params: SamplingParams,
    reward_type: str,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    """Evaluate the current vLLM model weights on the given dataset."""
    questions = [item["question"] for item in data]
    answers = [item["answers"] for item in data]
    rewards, metrics = run_inference(
        llm,
        tokenizer,
        questions,
        answers,
        sampling_params,
        reward_type,
        batch_size,
        max_length,
    )
    return {
        "rewards": float(np.mean(rewards)),
        "direct_correctness": float(np.mean(metrics["direct_correctness"])),
        "yes": float(np.mean(metrics["yes"])),
        "yes_failures": _safe_mean(metrics["yes_failures"]),
        "no_failures": _safe_mean(metrics["no_failures"]),
        "meta_alignments": float(np.mean(metrics["meta_alignments"])),
        "d_prime_type2": type2_d_prime(metrics["direct_correctness"], metrics["yes"]),
    }


def main(args):
    logger = get_logger(__name__)

    if args.output_dir is not None and args.wandb_run_name is None:
        args.wandb_run_name = os.path.basename(args.output_dir)
    if args.output_dir is None and args.wandb_run_name is not None:
        args.output_dir = os.path.join("outputs", args.wandb_run_name)
    checkpoint_dir, basename, run = None, None, None
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"[+] Output directory: {args.output_dir}")
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        basename = os.path.basename(args.output_dir)
    if args.wandb_run_name is not None:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.output_dir,
        )
        run.config.update(vars(args))

    seed_everything(args.seed)
    num_gpus = args.num_gpus

    logger.info("[+] Loading TriviaQA...")
    train_data = load_trivia_qa_meta(split="train", num_samples=args.num_samples)
    val_data = load_trivia_qa_meta(split="validation", num_samples=args.num_val_samples)
    logger.info(f"[+] Training samples: {len(train_data)}  |  Validation samples: {len(val_data)}")

    train_loader = DataLoader(
        RawQADataset(train_data),
        batch_size=args.num_data_per_iteration,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=RawQADataset.collate_fn,
    )
    infinite_loader = itertools.chain.from_iterable(itertools.repeat(train_loader))

    ray.init(ignore_reinit_error=True)
    logger.info(f"[+] Spawning {num_gpus} ESWorker actor(s)...")
    workers = [ESWorker.remote(args) for _ in range(num_gpus)]
    ray.get([w.ready.remote() for w in workers])
    logger.info("[+] All engines ready")

    local_population_size = args.population_size // num_gpus
    population_seed_gens = [np.random.RandomState(args.seed + i) for i in range(num_gpus)]
    logger.info("[+] Starting ES-Meta training")

    for iteration, batch in enumerate(infinite_loader, start=1):
        if iteration > args.num_iterations:
            break

        iter_start = time.time()
        questions, answers = batch["questions"], batch["answers"]
        seed_chunks = [gen.randint(0, 1_000_000, local_population_size) for gen in population_seed_gens]
        seeds = np.concatenate(seed_chunks)

        futures = [
            w.evaluate_seeds.remote(chunk.tolist(), questions, answers) for w, chunk in zip(workers, seed_chunks)
        ]
        results = ray.get(futures)

        rewards, agg = [], {k: [] for k in METRIC_KEYS}
        for chunk_rewards, chunk_metrics in results:
            rewards.extend(chunk_rewards)
            for k in agg:
                agg[k].extend(chunk_metrics[k])

        rewards_arr = np.array(rewards, dtype=np.float32)
        normalized = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
        ray.get([w.apply_update.remote(seeds.tolist(), args.alpha, normalized.tolist()) for w in workers])

        train_metrics = {
            "train/rewards": float(rewards_arr.mean()),
            "train/direct_correctness": float(np.mean(agg["direct_correctness"])),
            "train/yes": float(np.mean(agg["yes"])),
            "train/yes_failures": _safe_mean(agg["yes_failures"]),
            "train/no_failures": _safe_mean(agg["no_failures"]),
            "train/meta_alignments": float(np.mean(agg["meta_alignments"])),
            "train/d_prime_type2": type2_d_prime(agg["direct_correctness"], agg["yes"]),
        }
        iter_elapsed = time.time() - iter_start
        logger.info(f"[+] Iteration {iteration:03d} ({iter_elapsed:.1f}s) {format_metrics(train_metrics)}")
        if run is not None:
            run.log(train_metrics, step=iteration)

        if iteration % args.model_save_interval == 0 and checkpoint_dir is not None:
            save_dir = os.path.join(checkpoint_dir, f"{basename}_iter{iteration:03d}")
            os.makedirs(save_dir, exist_ok=True)
            ray.get(workers[0].save_weights.remote(os.path.join(save_dir, "model_weights.pth")))
            AutoTokenizer.from_pretrained(args.model).save_pretrained(save_dir)
            logger.info(f"[+] Checkpoint saved → {save_dir}")

        if iteration % args.evaluate_interval == 0:
            val_metrics = ray.get(workers[0].evaluate_full.remote(val_data))
            val_metrics = {"val/" + k: v for k, v in val_metrics.items()}
            logger.info(f"[+] Validation {iteration:03d} {format_metrics(val_metrics)}")
            if run is not None:
                run.log(val_metrics, step=iteration)

    logger.info("[+] Running final test evaluation (100 samples)...")
    test_data = load_trivia_qa_meta(split="test", num_samples=100)
    test_metrics = ray.get(workers[0].evaluate_full.remote(test_data))
    test_metrics = {"test/" + k: v for k, v in test_metrics.items()}
    logger.info(f"[+] Test {format_metrics(test_metrics)}")
    if run is not None:
        run.log(test_metrics, step=args.num_iterations)

    if args.output_dir is not None:
        final_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        ray.get(workers[0].save_weights.remote(os.path.join(final_dir, "model_weights.pth")))
        AutoTokenizer.from_pretrained(args.model).save_pretrained(final_dir)
        logger.info(f"[+] Final model saved → {final_dir}")

    ray.shutdown()


if __name__ == "__main__":
    main(parser.parse_args())
