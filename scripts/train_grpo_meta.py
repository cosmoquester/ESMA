"""Reinforcement learning for metacognition.

Train a model to align meta-judgments (Yes/No "Do you know the answer?") with
actual correctness: get direct answer via direct prompt, compute correctness,
prompt meta questions, sample a *group* of G meta responses per prompt, compute
group-relative advantages, and optimize with GRPO (Group Relative Policy Optimization).

GRPO replaces the per-sample value baseline in PPO with within-group normalization:
for each prompt, G responses are sampled and rewards are normalized across the group,
yielding advantages without a learned critic.  A clipped surrogate objective and
KL penalty against a frozen reference model are used for stable optimization.
"""

import argparse
import contextlib
import copy
import logging
import os

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from esma.data import load_fictional_qa_meta, load_trivia_qa_meta
from esma.dataset import MetaDataset
from esma.metric import IGNORE_VALUE, correctness_by_inclusion, meta_metrics, meta_yes, type2_d_prime
from esma.prompt import DIRECT_QA_PROMPT, META_QA_PROMPT
from esma.reward import REWARD_TYPE_TO_FUNCTION
from esma.utils import format_metrics, get_logger, seed_everything

# fmt: off
parser = argparse.ArgumentParser(description="RL meta training: GRPO for meta-judgment alignment")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model ID")

g = parser.add_argument_group("Data")
g.add_argument("--dataset", type=str, default="fictional_qa", choices=["trivia_qa", "fictional_qa"])
g.add_argument("--max-input-length", type=int, default=128, help="Maximum sequence length")
g.add_argument("--num-samples", type=int, help="Number of training samples to load")
g.add_argument("--num-val-samples", type=int, help="Number of validation samples to load")
g.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens for direct answer generation")
g.add_argument("--meta-max-new-tokens", type=int, default=8, help="Maximum tokens for meta (Yes/No) generation")
g.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for meta response")

g = parser.add_argument_group("Training")
g.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
g.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
g.add_argument("--learning-rate", "-lr", type=float, default=1e-6, help="Learning rate")
g.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
g.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
g.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
g.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")
g.add_argument("--reward-type", type=str, default="alignment", choices=list(REWARD_TYPE_TO_FUNCTION),
                help="Reward: alignment (0/1), esma (0/1/2), or correct (direct only)")
g.add_argument("--clip-epsilon", type=float, default=0.2, help="GRPO clipping epsilon")
g.add_argument("--group-size", type=int, default=4, help="Number of responses sampled per prompt (G)")
g.add_argument("--grpo-epochs", type=int, default=4, help="GRPO optimization epochs per rollout batch")
g.add_argument("--kl-coeff", type=float, default=0.01, help="KL penalty coefficient against reference model")

g = parser.add_argument_group("Experiment Settings")
g.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
g.add_argument("--seed", type=int, default=42, help="Random seed")
g.add_argument("--output-dir", type=str, help="Output directory")
g.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
g.add_argument("--eval-steps", type=int, default=1000, help="Evaluate every N steps")
g.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
g.add_argument("--wandb-run-name", type=str, help="Wandb run name")
g.add_argument("--wandb-project", type=str, default="meta-cognition", help="Wandb project")
g.add_argument("--wandb-entity", type=str, default="cosmoquester", help="Wandb entity")
# fmt: on


def run_batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: dict,
    max_new_tokens: int,
) -> list[int]:
    """Run inference on direct questions and return correctness (0/1) per item."""
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        correctness = correctness_by_inclusion(decoded, batch["answers"])
    model.train()
    return correctness


def compute_log_prob_for_generated(
    model: AutoModelForCausalLM,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_ids: list[list[int]],
    tokenizer: AutoTokenizer,
    temperature: float,
) -> torch.Tensor:
    """Compute sum of log probabilities for pre-generated tokens in a single forward pass.

    Given prompts and their sampled continuations, concatenates [prompt | generated]
    and runs one forward pass to score all generated tokens under the current policy.

    Returns sum_log_prob [B] with gradients for REINFORCE.
    """
    device = prompt_input_ids.device
    B = prompt_input_ids.size(0)
    prompt_len = prompt_input_ids.size(1)

    # Pad generated_ids to the same length
    max_gen_len = max(len(g) for g in generated_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_gen = torch.full((B, max_gen_len), pad_token_id, dtype=prompt_input_ids.dtype, device=device)
    gen_mask = torch.zeros(B, max_gen_len, dtype=prompt_attention_mask.dtype, device=device)
    for i, g in enumerate(generated_ids):
        gen_tokens = torch.tensor(g, dtype=prompt_input_ids.dtype, device=device)
        padded_gen[i, : len(g)] = gen_tokens
        gen_mask[i, : len(g)] = 1

    # Concatenate prompt + generated tokens
    full_input_ids = torch.cat([prompt_input_ids, padded_gen], dim=1)  # [B, prompt_len + max_gen_len]
    full_attention_mask = torch.cat([prompt_attention_mask, gen_mask], dim=1)

    # Single forward pass with gradients
    out = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
    logits = out.logits  # [B, seq_len, V]

    # Logits at positions [prompt_len-1 .. prompt_len+max_gen_len-2] predict
    # generated tokens at positions [prompt_len .. prompt_len+max_gen_len-1]
    gen_logits = logits[:, prompt_len - 1 : prompt_len + max_gen_len - 1, :]  # [B, max_gen_len, V]
    log_probs = F.log_softmax(gen_logits / temperature, dim=-1)  # [B, max_gen_len, V]

    # Gather log probs for the actual generated tokens
    token_log_probs = log_probs.gather(2, padded_gen.unsqueeze(2)).squeeze(2)  # [B, max_gen_len]

    # Mask out padding positions and sum per sample
    token_log_probs = token_log_probs * gen_mask
    sum_log_prob = token_log_probs.sum(dim=1)  # [B]

    return sum_log_prob


def generate_meta_responses(
    model: AutoModelForCausalLM,
    meta_prompt_input_ids: torch.Tensor,
    meta_prompt_attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool = True,
) -> tuple[list[list[int]], list[str], list[int]]:
    """No-grad meta generation for evaluation only. Returns generated_ids, decoded, meta_yes."""
    model.eval()
    with torch.no_grad():
        prompt_ids = meta_prompt_input_ids.to(model.device)
        prompt_mask = meta_prompt_attention_mask.to(model.device)
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = prompt_ids.shape[1]
        generated_ids = [out[i, prompt_len:].tolist() for i in range(out.size(0))]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        meta_yes_list = meta_yes(decoded)
    model.train()
    return generated_ids, decoded, meta_yes_list


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    max_new_tokens: int,
    reward_type: str,
) -> dict[str, torch.Tensor]:
    """Same evaluation as train_es.py: direct + meta generation, meta_metrics, reward."""
    model.eval()
    all_direct_correctness = []
    all_yes = []
    all_yes_failures = []
    all_no_failures = []
    all_meta_alignments = []
    all_rewards = []
    with torch.no_grad():
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
            all_rewards.extend(REWARD_TYPE_TO_FUNCTION[reward_type](direct_correctness, yes))
    result = {
        "rewards": torch.tensor(all_rewards, dtype=torch.float32, device=model.device),
        "direct_correctness": torch.tensor(all_direct_correctness, dtype=torch.float32, device=model.device),
        "yes": torch.tensor(all_yes, dtype=torch.float32, device=model.device),
        "yes_failures": torch.tensor(all_yes_failures, dtype=torch.float32, device=model.device),
        "no_failures": torch.tensor(all_no_failures, dtype=torch.float32, device=model.device),
        "meta_alignments": torch.tensor(all_meta_alignments, dtype=torch.float32, device=model.device),
        "d_prime_type2": torch.tensor(
            type2_d_prime(all_direct_correctness, all_yes), dtype=torch.float32, device=model.device
        ),
    }
    model.train()
    return result


def split_batch(batch: dict, num_chunks: int) -> list[dict]:
    """Split a batch dict into num_chunks mini-batches along the batch dimension."""
    if num_chunks <= 1:
        return [batch]
    first_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
    total = first_tensor.size(0)
    chunk_size = total // num_chunks
    if chunk_size == 0:
        return [batch]
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                chunk[k] = v[start:end]
            elif isinstance(v, list):
                chunk[k] = v[start:end]
            else:
                chunk[k] = v
        chunks.append(chunk)
    return chunks


def main(args):
    logger = get_logger(__name__)

    accelerator = Accelerator(
        log_with="wandb" if args.wandb_run_name else None,
    )

    if not accelerator.is_main_process:
        logger.setLevel(logging.CRITICAL)

    logger.info(f"[+] Accelerator device: {accelerator.device}")
    logger.info(f"[+] Num processes: {accelerator.num_processes}")

    if args.output_dir is not None and args.wandb_run_name is None:
        args.wandb_run_name = os.path.basename(args.output_dir)
    if args.output_dir is None and args.wandb_run_name is not None:
        args.output_dir = os.path.join("outputs", args.wandb_run_name)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"[+] Output directory: {args.output_dir}")
    else:
        checkpoint_dir = None

    if args.wandb_run_name is not None and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.output_dir,
            config=vars(args),
        )
    else:
        run = None

    seed_everything(args.seed)
    logger.info(f"[+] Using seed: {args.seed}")

    logger.info(f"[+] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info(f"[+] Loading {args.dataset} dataset...")
    if args.dataset == "trivia_qa":
        train_data = load_trivia_qa_meta(split="train", num_samples=args.num_samples)
        val_data = load_trivia_qa_meta(split="validation", num_samples=args.num_val_samples)
        prompt = DIRECT_QA_PROMPT
    elif args.dataset == "fictional_qa":
        train_data = load_fictional_qa_meta(split="train", num_samples=args.num_samples)
        val_data = load_fictional_qa_meta(split="validation", num_samples=args.num_val_samples)
        prompt = DIRECT_QA_PROMPT
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"[+] Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    logger.info(f"[+] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    logger.info(f"[+] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("[+] Creating frozen reference model for KL penalty...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_dataset = MetaDataset(
        train_data, tokenizer, max_length=args.max_input_length, prompt=prompt, meta_prompt=META_QA_PROMPT
    )
    val_dataset = MetaDataset(
        val_data, tokenizer, max_length=args.max_input_length, prompt=prompt, meta_prompt=META_QA_PROMPT
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * args.gradient_accumulation_steps,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MetaDataset.pad_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MetaDataset.pad_collate_fn,
    )

    reward_fn = REWARD_TYPE_TO_FUNCTION[args.reward_type]
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = len(train_loader)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    unwrapped_model = accelerator.unwrap_model(model)
    ref_model = ref_model.to(accelerator.device)

    global_step = 0
    best_meta_alignment = 0.0

    logger.info(f"[+] Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"[+] Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"[+] Starting GRPO meta training (group_size={args.group_size})...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"[+] Epoch {epoch}/{args.epochs}")
        epoch_rewards = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)

        for batch in progress_bar:
            mini_batches = split_batch(batch, args.gradient_accumulation_steps)
            num_accum = len(mini_batches)

            # ===================== Rollout phase =====================
            rollout_data = []
            all_correctness = []
            all_first_meta_yes = []
            all_rewards_stacks = []
            all_group_stds = []

            for mb in mini_batches:
                # 1) Direct answer and correctness (shared across the group)
                correctness = run_batch_inference(unwrapped_model, tokenizer, mb, args.max_new_tokens)

                # B = Batch Size
                meta_prompt_ids = mb["meta_input_ids"]
                meta_prompt_mask = mb["meta_attention_mask"]

                # 2) Sample G meta responses per prompt
                group_generated_ids = []
                group_meta_yes_lists = []
                group_old_log_probs = []
                group_ref_log_probs = []
                group_rewards = []

                for _ in range(args.group_size):
                    generated_ids, _, meta_yes_list = generate_meta_responses(
                        unwrapped_model,
                        meta_prompt_ids,
                        meta_prompt_mask,
                        tokenizer,
                        max_new_tokens=args.meta_max_new_tokens,
                        temperature=args.temperature,
                        do_sample=True,
                    )

                    with torch.no_grad():
                        old_lp = compute_log_prob_for_generated(
                            model,
                            meta_prompt_ids,
                            meta_prompt_mask,
                            generated_ids,
                            tokenizer,
                            args.temperature,
                        ).detach()

                        ref_lp = compute_log_prob_for_generated(
                            ref_model,
                            meta_prompt_ids,
                            meta_prompt_mask,
                            generated_ids,
                            tokenizer,
                            args.temperature,
                        ).detach()

                    rewards_list = reward_fn(correctness, meta_yes_list)
                    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=accelerator.device)

                    group_generated_ids.append(generated_ids)
                    group_meta_yes_lists.append(meta_yes_list)
                    group_old_log_probs.append(old_lp)
                    group_ref_log_probs.append(ref_lp)
                    group_rewards.append(rewards)

                # 3) Compute group-relative advantages
                rewards_stack = torch.stack(group_rewards, dim=0)  # [G, B]
                group_mean = rewards_stack.mean(dim=0, keepdim=True)  # [1, B]
                group_std = rewards_stack.std(dim=0, keepdim=True)  # [1, B]
                advantages_stack = (rewards_stack - group_mean) / (group_std + 1e-8)  # [G, B]

                # Flatten all groups
                flat_generated_ids = [gid for g_gids in group_generated_ids for gid in g_gids]
                flat_old_log_probs = torch.cat(group_old_log_probs, dim=0)  # [G*B]
                flat_ref_log_probs = torch.cat(group_ref_log_probs, dim=0)  # [G*B]
                flat_advantages = advantages_stack.reshape(-1).detach()  # [G*B]

                flat_meta_prompt_ids = meta_prompt_ids.repeat(args.group_size, 1)  # [G*B, L]
                flat_meta_prompt_mask = meta_prompt_mask.repeat(args.group_size, 1)  # [G*B, L]

                rollout_data.append(
                    {
                        "flat_meta_prompt_ids": flat_meta_prompt_ids,
                        "flat_meta_prompt_mask": flat_meta_prompt_mask,
                        "flat_generated_ids": flat_generated_ids,
                        "flat_old_log_probs": flat_old_log_probs,
                        "flat_ref_log_probs": flat_ref_log_probs,
                        "flat_advantages": flat_advantages,
                    }
                )
                all_correctness.extend(correctness)
                all_first_meta_yes.extend(group_meta_yes_lists[0])
                all_rewards_stacks.append(rewards_stack)
                all_group_stds.append(group_std)

            # ===================== GRPO update phase =====================
            for _ in range(args.grpo_epochs):
                all_ratios = []
                all_kl = []
                total_grpo_loss = 0.0
                total_policy_loss = 0.0
                for accum_idx, rd in enumerate(rollout_data):
                    is_last = accum_idx == num_accum - 1
                    ctx = contextlib.nullcontext() if is_last else accelerator.no_sync(model)
                    with ctx:
                        new_log_probs = compute_log_prob_for_generated(
                            model,
                            rd["flat_meta_prompt_ids"],
                            rd["flat_meta_prompt_mask"],
                            rd["flat_generated_ids"],
                            tokenizer,
                            args.temperature,
                        )

                        ratio = torch.exp(new_log_probs - rd["flat_old_log_probs"])  # [G*B]
                        surr1 = ratio * rd["flat_advantages"]
                        surr2 = (
                            torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * rd["flat_advantages"]
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # KL divergence against reference
                        log_ratio_ref = rd["flat_ref_log_probs"] - new_log_probs
                        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0  # [G*B], always >= 0

                        grpo_loss = (policy_loss + args.kl_coeff * kl.mean()) / num_accum

                        accelerator.backward(grpo_loss)
                    all_ratios.append(ratio.detach())
                    all_kl.append(kl.detach())
                    total_grpo_loss += grpo_loss.item()
                    total_policy_loss += policy_loss.item() / num_accum

                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            # ===================== Logging =====================
            all_rewards_flat = torch.cat([rs.reshape(-1) for rs in all_rewards_stacks])
            all_kl_cat = torch.cat(all_kl)
            all_ratios_cat = torch.cat(all_ratios)
            all_group_stds_mean = torch.cat([gs.reshape(-1) for gs in all_group_stds]).mean().item()

            r_mean = all_rewards_flat.mean().item()
            epoch_rewards.append(r_mean)
            global_step += 1

            if global_step % args.logging_steps == 0:
                # Aggregate meta alignment across the group (use first group sample for display)
                meta_aligned = sum(int(c == y) for c, y in zip(all_correctness, all_first_meta_yes))
                accuracy = sum(all_correctness) / len(all_correctness) if all_correctness else 0
                clip_frac = ((all_ratios_cat - 1.0).abs() > args.clip_epsilon).float().mean().item()
                mean_kl = all_kl_cat.mean().item()
                progress_bar.set_postfix(
                    {
                        "loss": f"{total_grpo_loss:.4f}",
                        "reward": f"{r_mean:.4f}",
                        "acc": f"{accuracy:.2f}",
                        "kl": f"{mean_kl:.4f}",
                        "clip": f"{clip_frac:.2f}",
                        "g_std": f"{all_group_stds_mean:.3f}",
                        "align": f"{meta_aligned}/{len(all_correctness)}",
                    }
                )
                if run is not None and accelerator.is_main_process:
                    run.log(
                        {
                            "train/grpo_loss": total_grpo_loss,
                            "train/policy_loss": total_policy_loss,
                            "train/mean_reward": r_mean,
                            "train/accuracy": accuracy,
                            "train/mean_kl": mean_kl,
                            "train/clip_fraction": clip_frac,
                            "train/mean_ratio": all_ratios_cat.mean().item(),
                            "train/group_reward_std": all_group_stds_mean,
                            "train/meta_alignment": meta_aligned / len(all_correctness) if all_correctness else 0,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if global_step % args.eval_steps == 0:
                metrics = evaluate_model(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    val_loader,
                    args.max_new_tokens,
                    args.reward_type,
                )
                all_val_metrics = {"val/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
                val_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_val_metrics.items()}
                if accelerator.is_main_process:
                    logger.info(f"[+] Step {global_step} validation {format_metrics(val_metrics)}")
                    if run is not None:
                        run.log(val_metrics, step=global_step)
                    if val_metrics["val/meta_alignments"] > best_meta_alignment and checkpoint_dir is not None:
                        best_meta_alignment = val_metrics["val/meta_alignments"]
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best"))
                        logger.info("[+] Best model (meta alignment) saved")

            if global_step % args.save_steps == 0 and checkpoint_dir is not None and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
                tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
                logger.info(f"[+] Checkpoint saved to step_{global_step}")

        if checkpoint_dir is not None and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, f"epoch_{epoch}"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"epoch_{epoch}"))
            logger.info(f"[+] Epoch {epoch} model saved")

        metrics = evaluate_model(
            accelerator.unwrap_model(model),
            tokenizer,
            val_loader,
            args.max_new_tokens,
            args.reward_type,
        )
        all_val_metrics = {"val/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
        val_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_val_metrics.items()}
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        if accelerator.is_main_process:
            logger.info(f"[+] Epoch {epoch} complete: train_reward={avg_reward:.4f}, val {format_metrics(val_metrics)}")
            if run is not None:
                run.log({"epoch/train_mean_reward": avg_reward, **val_metrics}, step=global_step)

    if checkpoint_dir is not None and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, "final"))
        tokenizer.save_pretrained(os.path.join(checkpoint_dir, "final"))
        logger.info("[+] Final model saved")

    metrics = evaluate_model(
        accelerator.unwrap_model(model),
        tokenizer,
        val_loader,
        args.max_new_tokens,
        args.reward_type,
    )
    all_val_metrics = {"val/" + k: accelerator.gather(v) for k, v in sorted(metrics.items())}
    val_metrics = {k: v[v != IGNORE_VALUE].mean().item() for k, v in all_val_metrics.items()}
    if accelerator.is_main_process:
        logger.info(f"[+] Training complete. Final val {format_metrics(val_metrics)}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(parser.parse_args())
