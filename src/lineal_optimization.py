from copy import deepcopy
from pprint import pprint
from src.utils import (
    constant_name_assign,
    constant_value_assign,
    evaluate,
    node_count,
    render_prog,
)
import numpy as np
import math


def compute_fitness(program, prediction, target, REG_STRENGTH):
    mse = 0
    for i in range(len(prediction)):
        mse_2 = 0
        for j in range(len(prediction[i])):
            mse_2 += abs(prediction[i][j] - target[i][j])
        mse += mse_2 / len(prediction[i])
    mse /= len(prediction)

    nodes_c = node_count(program)
    if nodes_c < REG_STRENGTH:
        return mse
    return mse + 9999 * nodes_c


def lineal_optimization_system(system, X, target):
    offspring = deepcopy(system)

    for system_i, edo_equation in enumerate(offspring["children"]):
        offspring_edo_equation = deepcopy(edo_equation)

        constants_count = len(offspring_edo_equation["children"])

        if constants_count > 0:
            A = np.array(
                [
                    np.array(
                        [
                            evaluate(ode_equation_term["children"][1], X_i)
                            for ode_equation_term in offspring_edo_equation["children"]
                        ]
                    )
                    for X_i in X
                ]
            )
            b = np.array([y[system_i] for y in target])

            x = np.linalg.lstsq(A, b, rcond=None)[0]

            offspring_edo_equation, _, _ = constant_name_assign(offspring_edo_equation)
            offspring_edo_equation = constant_value_assign(offspring_edo_equation, x)

            offspring["children"][system_i] = offspring_edo_equation

    return offspring
