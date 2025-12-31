import argparse
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("weight_change", type=str)
parser.add_argument("--filter-highest", type=float, default=1.0)
parser.add_argument("--output-path", "-o", type=str, required=True)
args = parser.parse_args()


def main(args):
    print(f"[+] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("[+] Loading weight change...")
    with open(args.weight_change, "rb") as f:
        weight_change = pickle.load(f)

    if args.filter_highest < 1.0:
        print(f"[+] Filtering highest {args.filter_highest * 100}% of weight change...")
        # Collect all absolute values efficiently
        abs_values = []
        for v in tqdm(weight_change.values(), desc="Filtering weight change"):
            abs_values.append(v.abs().flatten())

        print(f"[+] Concatenating {len(abs_values)} values...")
        all_values = torch.cat(abs_values)

        k = int(len(all_values) * (1 - args.filter_highest))
        if k == 0:
            k = 1
        print(f"[+] Finding {k}-th smallest value (out of {len(all_values)})...")
        threshold = torch.kthvalue(all_values, k).values.item()
        print(f"[+] Threshold: {threshold}")
    else:
        threshold = None

    print("[+] Applying weight change...")
    for name, param in tqdm(model.named_parameters(), desc="Applying weight change"):
        if name in weight_change:
            v = weight_change[name]
            if args.filter_highest < 1.0:
                v[v.abs() < threshold] = 0
            param.data.add_(v)

    print("[+] Saving model...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"[+] Saved model to {args.output_path}")


if __name__ == "__main__":
    main(args)
