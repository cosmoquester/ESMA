import argparse
import csv

import numpy as np

from esma.metric import type2_d_prime
from esma.utils import get_logger

parser = argparse.ArgumentParser(description="Bootstrap test for type 2 d' difference between two evaluation results")
parser.add_argument("--eval-a", type=str, required=True, help="Path to the first evaluation TSV file (candidate A)")
parser.add_argument("--eval-b", type=str, required=True, help="Path to the second evaluation TSV file (candidate B)")
parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


def load_eval_tsv(path: str) -> tuple[list[str], list[int], list[int]]:
    """Load an evaluation TSV file and return question_ids, direct_correctness, and yes columns."""
    question_ids = []
    direct_correctness = []
    yes = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            question_ids.append(row["question_id"])
            direct_correctness.append(int(row["direct_correctness"]))
            yes.append(int(row["yes"]))
    return question_ids, direct_correctness, yes


def bootstrap_type2_d_prime_difference(
    correctness_a: np.ndarray,
    yes_a: np.ndarray,
    correctness_b: np.ndarray,
    yes_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute bootstrapped distribution of (d'_b - d'_a) using paired resampling."""
    n = len(correctness_a)
    differences = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        d_a = type2_d_prime(correctness_a[indices].tolist(), yes_a[indices].tolist())
        d_b = type2_d_prime(correctness_b[indices].tolist(), yes_b[indices].tolist())
        differences[i] = d_b - d_a
    return differences


def main(args):
    logger = get_logger(__name__)
    rng = np.random.default_rng(args.seed)

    logger.info(f"[+] Loading candidate A: {args.eval_a}")
    ids_a, correctness_a, yes_a = load_eval_tsv(args.eval_a)
    logger.info(f"    Loaded {len(ids_a)} samples")

    logger.info(f"[+] Loading candidate B: {args.eval_b}")
    ids_b, correctness_b, yes_b = load_eval_tsv(args.eval_b)
    logger.info(f"    Loaded {len(ids_b)} samples")

    assert ids_a == ids_b, "Question IDs must be identical and in the same order between the two files."
    logger.info(f"[+] Verified {len(ids_a)} matching question_ids")

    correctness_a = np.array(correctness_a)
    yes_a = np.array(yes_a)
    correctness_b = np.array(correctness_b)
    yes_b = np.array(yes_b)

    d_prime_a = type2_d_prime(correctness_a.tolist(), yes_a.tolist())
    d_prime_b = type2_d_prime(correctness_b.tolist(), yes_b.tolist())
    observed_diff = d_prime_b - d_prime_a

    logger.info(f"[+] Type 2 d' (A): {d_prime_a:.4f}")
    logger.info(f"[+] Type 2 d' (B): {d_prime_b:.4f}")
    logger.info(f"[+] Observed difference (B - A): {observed_diff:.4f}")

    logger.info(f"[+] Running paired bootstrap with {args.n_bootstrap} iterations...")
    boot_diffs = bootstrap_type2_d_prime_difference(
        correctness_a,
        yes_a,
        correctness_b,
        yes_b,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )

    p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    p_value = min(p_value, 1.0)
    se = np.std(boot_diffs, ddof=1)

    logger.info(f"[+] Bootstrap results ({args.n_bootstrap} iterations):")
    logger.info(f"    Mean difference:  {np.mean(boot_diffs):.4f}")
    logger.info(f"    Std error:        {se:.4f}")
    logger.info(f"    p-value (two-sided): {p_value:.4f}")

    for confidence_level in [0.95, 0.99]:
        alpha = 1 - confidence_level
        ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
        significant = "significant" if p_value < alpha else "NOT significant"
        logger.info(f"    {confidence_level:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}] ({significant})")


if __name__ == "__main__":
    main(parser.parse_args())
