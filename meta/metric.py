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


def meta_yes_ko(meta_outputs: list[str]) -> list[int]:
    return [1 if "예" in output.lower() else 0 for output in meta_outputs]


def meta_yes_cn(meta_outputs: list[str]) -> list[int]:
    return [1 if "是" in output.lower() else 0 for output in meta_outputs]


def meta_yes_es(meta_outputs: list[str]) -> list[int]:
    return [1 if "sí" in output.lower() else 0 for output in meta_outputs]


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


def relative_meta_information(correctness: list[int], yes: list[int]) -> float:
    correctness = np.array(correctness)
    yes = np.array(yes)
    N = correctness.shape[0]

    p_acc = np.mean(correctness)
    if p_acc == 0 or p_acc == 1:
        return 0.0

    h_acc = -(p_acc * np.log2(p_acc) + (1 - p_acc) * np.log2(1 - p_acc))

    counts = np.zeros((2, 2))
    for a, c in zip(correctness, yes):
        counts[a, c] += 1

    p_joint = counts / N
    p_acc_marginal = np.sum(p_joint, axis=1)
    p_conf_marginal = np.sum(p_joint, axis=0)

    mi = 0
    for i in range(2):
        for j in range(2):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_acc_marginal[i] * p_conf_marginal[j]))

    return float(mi / h_acc)


def meta_metrics(
    direct_outputs: list[str],
    meta_outputs: list[str],
    answer_lists: list[list[str]],
    keep_length: bool = False,
    lang: str = "en",
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    direct_correctness = correctness_by_inclusion(direct_outputs, answer_lists)
    if lang == "ko":
        yes = meta_yes_ko(meta_outputs)
    elif lang.startswith("zh"):
        yes = meta_yes_cn(meta_outputs)
    elif lang == "es":
        yes = meta_yes_es(meta_outputs)
    else:
        yes = meta_yes(meta_outputs)
    yes_failures = meta_wrong_yes(direct_correctness, yes, keep_length)
    no_failures = meta_wrong_no(direct_correctness, yes, keep_length)
    meta_alignments = meta_alignment(direct_correctness, yes)
    return direct_correctness, yes, yes_failures, no_failures, meta_alignments
