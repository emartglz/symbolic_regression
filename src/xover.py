# TODO ahora mismo el xover puede darme un arbol con mayor profundidad máxima que MAX_DEPTH
from copy import deepcopy
from random import randint

from src.select_random_node import select_random_node


def xover(selected1, selected2, MAX_DEPTH):
    r = randint(0, len(selected1["children"]) - 1)

    offspring = deepcopy(selected1)
    xover_point1, depth1 = select_random_node(
        offspring["children"][r], offspring, 1, MAX_DEPTH
    )

    if depth1 == 0:
        xover_point1["children"][r] = selected2["children"][r]
    elif depth1 == 1:
        xover_point2 = selected2["children"][r]

        child_count1 = len(xover_point1["children"])
        child_count2 = len(xover_point2["children"])

        if child_count1 != 0:
            if child_count2 != 0:
                xover_point1["children"][randint(0, child_count1 - 1)] = xover_point2[
                    "children"
                ][randint(0, child_count2 - 1)]
            else:
                xover_point1["children"].pop(randint(0, child_count1 - 1))
        else:
            if child_count2 != 0:
                xover_point1["children"].append(
                    xover_point2["children"][randint(0, child_count2 - 1)]
                )

    elif depth1 >= 2:
        xover_point2 = selected2["children"][r]

        child_count1 = len(xover_point1["children"])
        child_count2 = len(xover_point2["children"])

        if child_count2 != 0:
            r1 = randint(0, child_count1 - 1)
            r2 = randint(0, child_count2 - 1)

            xover_point2, depth2 = select_random_node(
                xover_point2["children"][r2]["children"][1],
                xover_point2["children"][r2],
                3,
                MAX_DEPTH,
            )

            child_count2 = len(xover_point2["children"])
            r2 = randint(0, child_count2 - 1)

            if depth1 == 2:
                r1 = 1
            if depth2 == 2:
                r2 = 1

            xover_point1["children"][r1] = xover_point2["children"][r2]

    return offspring
