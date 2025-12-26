import numpy as np
from scipy.stats import norm

IGNORE_VALUE = -100


def correctness_by_inclusion(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        if any(answer.lower() in output.lower() for answer in answers):
            correctness.append(1)
        else:
            correctness.append(0)
    return correctness


def meta_yes(meta_outputs: list[str]) -> list[int]:
    return [1 if "yes" in output.lower() else 0 for output in meta_outputs]


def meta_wrong_yes(correctness: list[int], yes: list[int], keep_length: bool = False) -> list[int]:
    if not keep_length:
        return [1 - correct for correct, yes in zip(correctness, yes) if yes == 1]
    else:
        return [(IGNORE_VALUE if yes != 1 else 1 - correct) for correct, yes in zip(correctness, yes)]


def meta_wrong_no(correctness: list[int], yes: list[int], keep_length: bool = False) -> list[int]:
    if not keep_length:
        return [correct for correct, yes in zip(correctness, yes) if yes == 0]
    else:
        return [(IGNORE_VALUE if yes != 0 else correct) for correct, yes in zip(correctness, yes)]


def meta_alignment(correctness: list[int], yes: list[int]) -> list[int]:
    return [int(correct == yes) for correct, yes in zip(correctness, yes)]


def type2_d_prime(direct_correctness: list[int], meta_yes: list[int]) -> float:
    hit = [meta_yes[i] for i in range(len(direct_correctness)) if direct_correctness[i] == 1]
    false_alarm = [meta_yes[i] for i in range(len(direct_correctness)) if direct_correctness[i] == 0]

    hit_rate = np.mean(hit)
    false_alarm_rate = np.mean(false_alarm)

    hit_rate = np.clip(hit_rate, 1e-4, 1 - 1e-4)
    false_alarm_rate = np.clip(false_alarm_rate, 1e-4, 1 - 1e-4)

    d_prime_type2 = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    return float(d_prime_type2)


def meta_metrics(
    direct_outputs: list[str], meta_outputs: list[str], answer_lists: list[list[str]], keep_length: bool = False
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    direct_correctness = correctness_by_inclusion(direct_outputs, answer_lists)
    yes = meta_yes(meta_outputs)
    yes_failures = meta_wrong_yes(direct_correctness, yes, keep_length)
    no_failures = meta_wrong_no(direct_correctness, yes, keep_length)
    meta_alignments = meta_alignment(direct_correctness, yes)
    return direct_correctness, yes, yes_failures, no_failures, meta_alignments
