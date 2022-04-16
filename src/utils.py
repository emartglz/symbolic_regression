from copy import deepcopy

import numpy as np

from src.constants import ZERO


def constant_name_assign(selected, number=0, constant=[]):
    offspring = deepcopy(selected)
    if "children" not in offspring:
        if "value" in offspring:
            offspring["constant"] = number
            constant.append(offspring.pop("value"))
            number += 1
        return offspring, number, constant

    child_number = len(offspring["children"])
    for c in range(child_number):
        offspring["children"][c], number, constant = constant_name_assign(
            offspring["children"][c], number, constant
        )

    return offspring, number, constant


def constant_value_assign(selected, constants):
    offspring = deepcopy(selected)
    if "children" not in offspring:
        if "constant" in offspring:
            offspring["value"] = constants[offspring.pop("constant")]
        return offspring

    child_number = len(offspring["children"])
    for c in range(child_number):
        offspring["children"][c] = constant_value_assign(
            offspring["children"][c], constants
        )

    return offspring


def constant_count(x):
    if "children" not in x:
        if "constant" in x:
            return 1
        return 0
    return sum([constant_count(c) for c in x["children"]])


def node_count(x):
    if "children" not in x:
        return 1
    return sum([node_count(c) for c in x["children"]])


def evaluate(node, row):
    if "children" not in node:
        if "feature_name" in node:
            return row[node["feature_name"]]
        return node["value"]
    return node["func"](*[evaluate(c, row) for c in node["children"]])


def render_prog(node):
    if "children" not in node:
        if "feature_name" in node:
            return node["feature_name"]
        if "value" in node:
            return node["value"]
        if "constant" in node:
            return f"C{node['constant']}"
    return node["format_str"](*[render_prog(c) for c in node["children"]])


def take_n_samples_regular(n, l):
    step = int(len(l) / n)
    list_r = []
    for i in range(n):
        list_r.append(l[i * step])

    return list_r


def filter_zero_terms_edo_equation(equation):
    offspring = deepcopy(equation)
    for i, edo_term in enumerate(offspring["children"]):
        if abs(edo_term["children"][0]["value"]) < ZERO:
            offspring["children"].pop(i)

    return offspring


def filter_zero_terms_edo_system(system):
    offspring = deepcopy(system)
    for i, edo_equation in enumerate(offspring["children"]):
        offspring["children"][i] = filter_zero_terms_edo_equation(equation=edo_equation)

    return offspring


def round_terms_edo_equation(equation, ROUND_SIZE=5):
    offspring = deepcopy(equation)
    for i, edo_term in enumerate(offspring["children"]):
        edo_term["children"][0]["value"] = np.round(
            edo_term["children"][0]["value"], ROUND_SIZE
        )

    return offspring


def round_terms_edo_system(system, ROUND_SIZE=5):
    offspring = deepcopy(system)
    for i, edo_equation in enumerate(offspring["children"]):
        offspring["children"][i] = round_terms_edo_equation(
            equation=edo_equation, ROUND_SIZE=ROUND_SIZE
        )

    return offspring
