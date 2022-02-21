from random import random, randint


def select_random_node(selected, parent, depth, MAX_DEPTH):
    if "children" not in selected:
        return (parent, depth - 1)
    # favor nodes near the root
    # if randint(0, MAX_DEPTH) >= depth:

    child_count = len(selected["children"])
    # completly random
    if random() <= 0.5 or child_count == 0:
        return (selected, depth)
    return select_random_node(
        selected["children"][randint(0, child_count - 1)],
        selected,
        depth + 1,
        MAX_DEPTH,
    )
