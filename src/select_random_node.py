from random import random, randint


def flat_tree(selected, depth, MAX_DEPTH):
    ret = []
    if "children" not in selected or depth > MAX_DEPTH:
        return ret

    ret.append((selected, depth))

    for i in selected["children"]:
        ret += flat_tree(i, depth + 1, MAX_DEPTH)

    return ret


def select_random_node(selected, depth, MAX_DEPTH):
    flated = flat_tree(selected, depth, MAX_DEPTH)

    flat_count = len(flated)
    return flated[randint(0, flat_count - 1)]
