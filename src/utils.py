from copy import deepcopy
import json
import marshal
import base64
import types
import numpy as np
from src.constants import ZERO
import csv
import os


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
    for edo_term in equation["children"]:
        if abs(edo_term["children"][0]["value"]) < ZERO:
            offspring["children"].remove(edo_term)

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


def serialize_system(node):
    offspring = deepcopy(node)

    if "children" not in offspring:
        return offspring

    func_code_string = marshal.dumps(offspring["func"].__code__)
    func_code_base64 = base64.b64encode(func_code_string)
    offspring["func"] = str(func_code_base64.decode("ascii"))

    format_code_string = marshal.dumps(offspring["format_str"].__code__)
    format_code_base64 = base64.b64encode(format_code_string)
    offspring["format_str"] = str(format_code_base64.decode("ascii"))

    child_number = len(offspring["children"])
    for c in range(child_number):
        offspring["children"][c] = serialize_system(offspring["children"][c])

    return offspring


def deserialize_system(node):
    offspring = deepcopy(node)

    if "children" not in offspring:
        return offspring

    func_code_string = base64.b64decode(offspring["func"])
    func_code_marshal = marshal.loads(func_code_string)
    func_code_base64 = types.FunctionType(func_code_marshal, globals())
    offspring["func"] = func_code_base64

    format_code_string = base64.b64decode(offspring["format_str"])
    format_code_marshal = marshal.loads(format_code_string)
    format_code_base64 = types.FunctionType(format_code_marshal, globals())
    offspring["format_str"] = format_code_base64

    child_number = len(offspring["children"])
    for c in range(child_number):
        offspring["children"][c] = deserialize_system(offspring["children"][c])

    return offspring


def save_results(results, file_name):
    offsprint = deepcopy(results["system"])
    results["system"] = serialize_system(offsprint)

    with open(f"{file_name}.json", "w") as fp:
        json.dump(results, fp)

    return results


def get_results(file_name):
    with open(f"{file_name}.json") as json_file:
        data = json.load(json_file)

    offspring = deepcopy(data["system"])
    data["system"] = deserialize_system(offspring)

    return data


def save_samples(X, file_name):
    if not len(X):
        return
    
    directory = os.path.dirname(f"{file_name}.csv")
    os.makedirs(directory, exist_ok=True)

    with open(f"{file_name}.csv", "w", newline="") as csvfile:
        fieldnames = list(X[0].keys())

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(X)):
            writer.writerow(X[i])


def load_samples(file_name):
    X = []

    with open(f"{file_name}.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            X.append({k: float(v) for k, v in row.items()})

    return X


def group_without_names(X):
    result = []
    for i in range(len(X[0])):
        result_i = []
        for j in range(len(X)):
            result_i.append(X[j][i])

        result.append(result_i)

    return result


def group_with_names(X, variable_names):
    ret = [
        {variable_names[j]: X[j][i] for j in range(len(variable_names))}
        for i in range(len(X[0]))
    ]

    return ret


def separate_samples(variable_names, X_samples):
    ret = []

    for i in variable_names:
        actual = []
        for j in X_samples:
            actual.append(j[i])
        ret.append(actual)

    return ret
