def correct_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    return direct_correctness.copy()


def meta_alignment_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    return [int(correct == yes) for correct, yes in zip(direct_correctness, meta_yes)]


def esma_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(2)
            else:
                rewards.append(1)
        else:
            if correct:
                rewards.append(1)
            else:
                rewards.append(0)
    return rewards


REWARD_TYPE_TO_FUNCTION = {
    "correct": correct_reward,
    "alignment": meta_alignment_reward,
    "esma": esma_reward,
}
