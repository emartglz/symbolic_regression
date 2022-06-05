from random import randint, random
from src.nodes import (
    population_edo_ecuation,
    population_edo_ecuation_str,
    population_edo_term,
    population_edo_term_str,
    system,
    system_str,
)


def random_operation_tree(depth, features_names, operations, MAX_DEPTH):
    # favor adding function nodes near the tree root and
    # leaf nodes as depth increases
    if depth < MAX_DEPTH and randint(0, MAX_DEPTH) >= depth:
        op = operations[randint(0, len(operations) - 1)]
        return {
            "func": op["func"],
            "children": [
                random_operation_tree(
                    depth=depth + 1,
                    features_names=features_names,
                    operations=operations,
                    MAX_DEPTH=MAX_DEPTH,
                )
                for _ in range(op["arg_count"])
            ],
            "format_str": op["format_str"],
        }
    else:
        return {"feature_name": features_names[randint(0, len(features_names) - 1)]}


def random_edo_term(features_names, operations, MAX_DEPTH):
    return {
        "func": population_edo_term,
        "children": [
            {"value": 1},
            random_operation_tree(
                depth=3,
                features_names=features_names,
                operations=operations,
                MAX_DEPTH=MAX_DEPTH,
            ),
        ],
        "format_str": population_edo_term_str,
    }


def random_edo_equation(features_names, operations, MAX_DEPTH):
    return {
        "func": population_edo_ecuation,
        "children": [
            random_edo_term(
                features_names=features_names,
                operations=operations,
                MAX_DEPTH=MAX_DEPTH,
            )
            for _ in range(randint(1, MAX_DEPTH))
        ],
        "format_str": population_edo_ecuation_str,
    }


def random_system(
    system_lenght,
    operations,
    features_names,
    MAX_DEPTH,
):
    return {
        "func": system,
        "children": [
            random_edo_equation(
                features_names=features_names,
                operations=operations,
                MAX_DEPTH=MAX_DEPTH,
            )
            for _ in range(system_lenght)
        ],
        "format_str": system_str,
    }
