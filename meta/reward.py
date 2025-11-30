def correct_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    return direct_correctness.copy()


def multilevel_reward(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(3)
            else:
                rewards.append(2)
        else:
            if correct:
                rewards.append(1)
            else:
                rewards.append(0)
    return rewards


def multilevel_reward2(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(4)
            else:
                rewards.append(2)
        else:
            if correct:
                rewards.append(1)
            else:
                rewards.append(0)
    return rewards


def multilevel_reward3(direct_correctness: list[int], meta_yes: list[int]) -> list[int]:
    rewards = []
    for correct, yes in zip(direct_correctness, meta_yes):
        if correct == yes:
            if correct:
                rewards.append(4)
            else:
                rewards.append(2)
        else:
            if correct:
                rewards.append(2)
            else:
                rewards.append(0)
    return rewards


REWARD_TYPE_TO_FUNCTION = {
    "correct": correct_reward,
    "multilevel": multilevel_reward,
    "multilevel2": multilevel_reward2,
    "multilevel3": multilevel_reward3,
}
