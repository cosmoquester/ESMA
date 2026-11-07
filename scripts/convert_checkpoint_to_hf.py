"""Convert an ES-Meta pickle checkpoint (model_weights.pth) to HuggingFace
saved-pretrained format.

Checkpoint layout produced by train_es_meta_vllm.py:
    <vllm_ckpt>/
        model_weights.pth   – torch.save'd {param_name: tensor} state dict
        tokenizer_config.json, tokenizer.json, …  – saved by tokenizer.save_pretrained

Usage
-----
    python scripts/convert_checkpoint_to_hf.py \
        --vllm-ckpt outputs/run1/checkpoints/run1_iter060 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir outputs/run1/hf_iter060

    # Infer model from tokenizer files already inside vllm-ckpt:
    python scripts/convert_checkpoint_to_hf.py \
        --vllm-ckpt outputs/run1/final_model \
        --output-dir outputs/run1/hf_final
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Convert ES-Meta pickle checkpoint to HuggingFace saved-pretrained format")
parser.add_argument("--vllm-ckpt", required=True, help="Directory containing model_weights.pth")
parser.add_argument("--model", required=True, help="Original HuggingFace model ID.")
parser.add_argument("--output-dir", required=True, help="Destination directory for the converted HuggingFace model.")
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])


def main(args: argparse.Namespace) -> None:
    weights_path = os.path.join(args.vllm_ckpt, "model_weights.pth")
    if not os.path.isfile(weights_path):
        sys.exit(f"[!] model_weights.pth not found in {args.vllm_ckpt}")

    print(f"[+] Loading state dict from {weights_path} …")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[args.dtype]

    print(f"[+] Loading model architecture: {args.model} (dtype={args.dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True)

    hf_state_dict = {}
    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."

        # Split qkv_proj -> q_proj, k_proj, v_proj
        qkv_w_key = f"{prefix}self_attn.qkv_proj.weight"
        if qkv_w_key in state_dict:
            qkv_w = state_dict[qkv_w_key]
            q_dim = model.model.layers[layer_idx].self_attn.q_proj.weight.shape[0]
            k_dim = model.model.layers[layer_idx].self_attn.k_proj.weight.shape[0]
            hf_state_dict[f"{prefix}self_attn.q_proj.weight"] = qkv_w[:q_dim, :]
            hf_state_dict[f"{prefix}self_attn.k_proj.weight"] = qkv_w[q_dim : q_dim + k_dim, :]
            hf_state_dict[f"{prefix}self_attn.v_proj.weight"] = qkv_w[q_dim + k_dim :, :]
            del state_dict[qkv_w_key]

        qkv_b_key = f"{prefix}self_attn.qkv_proj.bias"
        if qkv_b_key in state_dict:
            qkv_b = state_dict[qkv_b_key]
            q_dim = model.model.layers[layer_idx].self_attn.q_proj.bias.shape[0]
            k_dim = model.model.layers[layer_idx].self_attn.k_proj.bias.shape[0]
            hf_state_dict[f"{prefix}self_attn.q_proj.bias"] = qkv_b[:q_dim]
            hf_state_dict[f"{prefix}self_attn.k_proj.bias"] = qkv_b[q_dim : q_dim + k_dim]
            hf_state_dict[f"{prefix}self_attn.v_proj.bias"] = qkv_b[q_dim + k_dim :]
            del state_dict[qkv_b_key]

        # Split gate_up_proj -> gate_proj, up_proj
        gate_up_w_key = f"{prefix}mlp.gate_up_proj.weight"
        if gate_up_w_key in state_dict:
            gate_up_w = state_dict[gate_up_w_key]
            gate_dim = model.model.layers[layer_idx].mlp.gate_proj.weight.shape[0]
            hf_state_dict[f"{prefix}mlp.gate_proj.weight"] = gate_up_w[:gate_dim, :]
            hf_state_dict[f"{prefix}mlp.up_proj.weight"] = gate_up_w[gate_dim:, :]
            del state_dict[gate_up_w_key]

    for k, v in state_dict.items():
        hf_state_dict[k] = v

    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if missing:
        print(f"[!] Missing keys ({len(missing)}): {missing[:10]}{'…' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[!] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'…' if len(unexpected) > 10 else ''}")
    if not missing and not unexpected:
        print("[+] State dict loaded with no missing or unexpected keys")

    print(f"[+] Saving tokenizer to {args.output_dir} …")
    tokenizer.save_pretrained(args.output_dir)

    print(f"[+] Saving model to {args.output_dir} …")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    print(f"[+] Done. HuggingFace model saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
