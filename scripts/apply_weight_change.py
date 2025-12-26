import argparse
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("weight_change", type=str)
parser.add_argument("--output-path", "-o", type=str)
args = parser.parse_args()


def main(args):
    print(f"[+] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("[+] Loading weight change...")
    with open(args.weight_change, "rb") as f:
        weight_change = pickle.load(f)

    print("[+] Applying weight change...")
    for name, param in model.named_parameters():
        if name in weight_change:
            param.data.add_(weight_change[name])

    print("[+] Saving model...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"[+] Saved model to {args.output_path}")


if __name__ == "__main__":
    main(args)
