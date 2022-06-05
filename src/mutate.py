from copy import deepcopy
from random import randint, random
from src.select_random_node import select_random_node
from src.random_prog import random_edo_term


def mutate_leaf(mutate_point, features_names, operations, VARIABLE_PROBABILITY):
    r = random()

    # add variable
    if r < VARIABLE_PROBABILITY:
        return {"feature_name": features_names[randint(0, len(features_names) - 1)]}

    # add program
    op = operations[randint(0, len(operations) - 1)]

    return {
        "func": op["func"],
        "format_str": op["format_str"],
        "children": [mutate_point]
        + [
            {"feature_name": features_names[randint(0, len(features_names) - 1)]}
            for _ in range(op["arg_count"] - 1)
        ],
    }


def mutate_operation(
    mutate_point,
    features_names,
    operations,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
):
    r = random()

    offspring = deepcopy(mutate_point)

    # Change operation same arity
    if r < CHANGE_OPERATION_PROBABILITY:
        possibles_func = [
            i
            for i in filter(
                lambda x: x["arg_count"] == len(offspring["children"]),
                operations,
            )
        ]
        op = possibles_func[randint(0, len(possibles_func) - 1)]

        offspring["func"] = op["func"]
        offspring["format_str"] = op["format_str"]

    # Delete node
    if r < CHANGE_OPERATION_PROBABILITY + DELETE_NODE_PROBABILITY:
        offspring = offspring["children"][0]

    # Add operation using same structure
    else:
        r_operation = operations[randint(0, len(operations) - 1)]
        node = deepcopy(offspring)

        offspring["children"] = [
            {"feature_name": features_names[randint(0, len(features_names) - 1)]}
            for _ in range(r_operation["arg_count"])
        ]
        offspring["children"][-1] = node
        offspring["func"] = r_operation["func"]
        offspring["format_str"] = r_operation["format_str"]

    return offspring


def mutate_operation_tree(
    mutate_point,
    depth,
    features_names,
    operations,
    VARIABLE_PROBABILITY,
    MAX_DEPTH,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
):
    offspring = deepcopy(mutate_point)

    # offspring is leaf
    if "children" not in offspring:
        offspring = mutate_leaf(
            offspring,
            features_names=features_names,
            operations=operations,
            VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
        )
    # offspring is operation
    else:
        mutate_node, depth = select_random_node(
            offspring, offspring, depth=depth + 1, MAX_DEPTH=MAX_DEPTH
        )

        mutate_node_children_count = len(mutate_node["children"])
        children = randint(0, mutate_node_children_count - 1)

        # children is leaf
        if "children" not in mutate_node["children"][children]:
            mutate_node["children"][children] = mutate_leaf(
                mutate_node["children"][children],
                features_names=features_names,
                operations=operations,
                VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
            )
        # children is operation
        else:
            mutate_node["children"][children] = mutate_operation(
                mutate_node["children"][children],
                features_names=features_names,
                operations=operations,
                CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
                DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
            )

    return offspring


def mutate_edo_term(
    mutate_point,
    features_names,
    operations,
    MAX_DEPTH,
    VARIABLE_PROBABILITY,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
):
    offspring = deepcopy(mutate_point)

    offspring["children"][1] = mutate_operation_tree(
        mutate_point=offspring["children"][1],
        depth=3,
        features_names=features_names,
        operations=operations,
        VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
        MAX_DEPTH=MAX_DEPTH,
        CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
        DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
    )

    return offspring


def mutate_edo_equation(
    mutate_point,
    features_names,
    operations,
    MAX_DEPTH,
    VARIABLE_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
    CHANGE_OPERATION_PROBABILITY,
):
    offspring = deepcopy(mutate_point)

    edo_term_count = len(offspring["children"])

    if edo_term_count > 0:
        r = random()
        edo_term = randint(0, edo_term_count - 1)

        # delete edo term
        if r < DELETE_NODE_PROBABILITY:
            offspring["children"].pop(edo_term)

        # add edo term
        elif r < DELETE_NODE_PROBABILITY + ADD_OPERATION_PROBABILITY:
            offspring["children"].append(
                random_edo_term(
                    features_names=features_names,
                    operations=operations,
                    MAX_DEPTH=4,
                )
            )

        # mutate edo term
        else:
            offspring["children"][edo_term] = mutate_edo_term(
                offspring["children"][edo_term],
                features_names=features_names,
                operations=operations,
                MAX_DEPTH=MAX_DEPTH,
                VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
                CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
                DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
            )

    # no children in population ecuation, add one
    else:
        offspring["children"].append(
            random_edo_term(
                features_names=features_names,
                operations=operations,
                MAX_DEPTH=4,
            )
        )

    return offspring


def mutate_system(
    selected,
    operations,
    features_names,
    MAX_DEPTH,
    VARIABLE_PROBABILITY,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
):
    offspring = deepcopy(selected)

    edo_equation_count = len(offspring["children"])

    edo_equation = randint(0, edo_equation_count - 1)

    offspring["children"][edo_equation] = mutate_edo_equation(
        mutate_point=offspring["children"][edo_equation],
        features_names=features_names[edo_equation],
        operations=operations,
        MAX_DEPTH=MAX_DEPTH,
        VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
        DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
        ADD_OPERATION_PROBABILITY=ADD_OPERATION_PROBABILITY,
        CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
    )

    return offspring
