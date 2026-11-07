import re

import numpy as np
from scipy.stats import norm

from .utils import normalize_answer

IGNORE_VALUE = -100


def correctness_by_inclusion(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        if any(answer.lower() in output.lower() for answer in answers):
            correctness.append(1)
        else:
            correctness.append(0)
    return correctness


def correctness_by_exact_match(outputs: list[str], answer_lists: list[list[str]]) -> list[int]:
    """Grade open-ended QA (e.g. TriviaQA) by normalized exact match.

    The model's output is counted correct only if, after normalization, it equals one
    of the gold answers exactly. Unlike ``correctness_by_inclusion`` this rejects a
    correct answer buried inside extra words, so it requires the model to emit just the
    answer (see ``DIRECT_QA_EXACT_PROMPT``).
    """
    correctness = []
    for output, answers in zip(outputs, answer_lists):
        pred = normalize_answer(output)
        golds = {normalize_answer(answer) for answer in answers}
        correctness.append(1 if pred in golds else 0)
    return correctness


def correctness_arc(outputs: list[str], questions: list[str], answer_lists: list[list[str]]) -> list[int]:
    """Grade ARC-Challenge by either the chosen option letter or the choice text.

    Instruct models often answer a multiple-choice question with just the option
    letter ("B"), which substring inclusion of the choice text would miss. The
    labeled options are parsed back out of the question, the correct label is
    resolved from the gold choice text, and the output is counted correct if it
    either mentions the correct choice text or selects the correct letter.
    """
    correctness = []
    for output, question, answers in zip(outputs, questions, answer_lists):
        options = {}  # label -> text
        for line in question.splitlines():
            match = re.match(r"\s*([A-Za-z0-9])[.)]\s+(.*)", line)
            if match:
                options[match.group(1).upper()] = match.group(2).strip()

        gold_texts = {a.strip() for a in answers}
        correct_labels = {label for label, text in options.items() if text in gold_texts}

        text_hit = any(a.lower() in output.lower() for a in answers)

        letter_hit = False
        if options:
            label_pattern = "".join(re.escape(label) for label in options)
            letter_match = re.search(rf"(?<![A-Za-z0-9])([{label_pattern}])(?![A-Za-z0-9])", output.upper())
            if letter_match is not None:
                letter_hit = letter_match.group(1) in correct_labels

        correctness.append(1 if (text_hit or letter_hit) else 0)
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
    direct_correctness: list[int] | None = None,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    if direct_correctness is None:
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
