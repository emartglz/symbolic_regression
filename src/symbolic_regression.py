import operator
from random import randint, random, seed
from copy import deepcopy
from functools import reduce
from scipy.optimize import least_squares
import numpy as np
from pprint import pprint
from math import *
from src.mutate import mutate_system

from src.operation import ADD, DIV, MUL, NEG, SUB
from src.random_prog import random_system
from src.xover import xover


def render_prog(node):
    if "children" not in node:
        if "feature_name" in node:
            return node["feature_name"]
        if "value" in node:
            return node["value"]
        if "constant" in node:
            return f"C{node['constant']}"
    return node["format_str"](*[render_prog(c) for c in node["children"]])


def evaluate(node, row):
    if "children" not in node:
        if "feature_name" in node:
            return row[node["feature_name"]]
        return node["value"]
    return node["func"](*[evaluate(c, row) for c in node["children"]])


def evaluate_least_squares(theta, node, row):
    if "children" not in node:
        if "feature_name" in node:
            return row[node["feature_name"]]
        if "constant" in node:
            return theta[node["constant"]]
        if "value" in node:
            return node["value"]
    return node["func"](
        *[evaluate_least_squares(theta, c, row) for c in node["children"]]
    )


def get_random_parent(population, fitness, TOURNAMENT_SIZE):
    # randomly select population members for the tournament
    tournament_members = [
        randint(0, len(population) - 1) for _ in range(TOURNAMENT_SIZE)
    ]
    # select tournament member with best fitness
    member_fitness = [(fitness[i], population[i]) for i in tournament_members]
    return min(member_fitness, key=lambda x: x[0])[1]


def get_offspring(
    population,
    fitness,
    system_lenght,
    operations,
    features_names,
    MAX_DEPTH,
    CONSTANT_PROBABILITY,
    VARIABLE_PROBABILITY,
    MAX_CONSTANT,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
    XOVER_PCT,
    TOURNAMENT_SIZE,
):
    parent1 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
    if random() < XOVER_PCT:
        parent2 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
        return xover(parent1, parent2, MAX_DEPTH)
    else:
        return mutate_system(
            selected=parent1,
            operations=operations,
            features_names=features_names,
            MAX_DEPTH=MAX_DEPTH,
            VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
            CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
            DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
            ADD_OPERATION_PROBABILITY=ADD_OPERATION_PROBABILITY,
            MAX_CONSTANT=MAX_CONSTANT,
        )


def node_count(x):
    if "children" not in x:
        return 1
    return sum([node_count(c) for c in x["children"]])


def constant_count(x):
    if "children" not in x:
        if "constant" in x:
            return 1
        return 0
    return sum([constant_count(c) for c in x["children"]])


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


def compute_fitness(program, prediction, target, REG_STRENGTH):
    mse = 0
    for i in range(len(prediction)):
        mse_2 = 0
        for j in range(len(prediction[i])):
            mse_2 += (prediction[i][j] - target[i][j]) ** 2
        mse += mse_2 / len(prediction[i])
    mse /= len(prediction)

    penalty = max(1, log(node_count(program), REG_STRENGTH))
    # penalty = 1
    return mse * penalty


def fun(theta, prog, ts, ys, REG_STRENGTH):
    prediction = [evaluate_least_squares(theta, prog, t) for t in ts]
    return compute_fitness(prog, prediction, ys, REG_STRENGTH)


def symbolic_regression(
    X,
    target,
    MAX_GENERATIONS=10,
    N_GENERATION_OPTIMIZE=3,
    seed_g=random(),
    MAX_DEPTH=10,
    POP_SIZE=300,
    CONSTANT_PROBABILITY=0.3,
    VARIABLE_PROBABILITY=0.3,
    MAX_CONSTANT=100,
    CHANGE_OPERATION_PROBABILITY=0.3,
    DELETE_NODE_PROBABILITY=0.3,
    ADD_OPERATION_PROBABILITY=0.4,
    TOURNAMENT_SIZE=3,
    XOVER_PCT=0.7,
    REG_STRENGTH=5,
    EPSILON=0.001,
    PROPORTION_OF_BESTS=1 / 3,
):
    seed(seed_g)
    feature_lenght = len(X[0])
    system_lenght = len(target[0])

    features_names = []
    for i in range(feature_lenght):
        features_names.append(f"X{i + 1}")

    X2 = []
    for x in X:
        temp = {}
        for i, k in enumerate(features_names):
            temp[k] = x[i]
        X2.append(temp)
    X = X2

    operations = (ADD, SUB, MUL, DIV, NEG)

    population = [
        random_system(
            system_lenght=system_lenght,
            operations=operations,
            features_names=features_names,
            MAX_DEPTH=MAX_DEPTH,
            MAX_CONSTANT=MAX_CONSTANT,
        )
        for _ in range(POP_SIZE)
    ]

    global_best = float("inf")
    for gen in range(MAX_GENERATIONS):
        fitness = []
        for i_prog, prog in enumerate(population):
            print(f"{i_prog + 1}/{POP_SIZE}", end="\r")

            # print(render_prog(prog))

            optimize = False
            if gen % N_GENERATION_OPTIMIZE == 0:
                prog_const, constant, constant_ini = constant_name_assign(prog, 0, [])

                if constant != 0:
                    optimize = True
                    prediction = least_squares(
                        fun,
                        [random() * MAX_CONSTANT for _ in range(constant)],
                        bounds=(-1, MAX_CONSTANT + 1),
                        kwargs={
                            "prog": prog_const,
                            "ts": X,
                            "ys": target,
                            "REG_STRENGTH": REG_STRENGTH,
                        },
                    )

                    score = prediction.cost

                    prog = constant_value_assign(prog_const, prediction.x)

            if not optimize:
                prediction = [evaluate(prog, Xi) for Xi in X]
                score = compute_fitness(prog, prediction, target, REG_STRENGTH)

            # print(score, optimize)
            # print(render_prog(prog))

            if score < global_best:
                global_best = score
                best_prog = prog

            fitness.append(score)

        mean = reduce(lambda a, b: a + b, fitness) / len(fitness)

        print(
            f"Generation: {gen + 1}\nBest Score: {global_best}\nMean score: {mean}\nBest program:\n{render_prog(best_prog)}\n"
        )

        if global_best < EPSILON:
            break

        member_fitness = [(fitness[i], i, population[i]) for i in range(POP_SIZE)]
        member_fitness.sort()

        best_amount = int(POP_SIZE * PROPORTION_OF_BESTS)
        population_next_gen = [i[2] for i in member_fitness[:best_amount]]
        for i in range(best_amount, POP_SIZE):
            population_next_gen.append(
                get_offspring(
                    population,
                    fitness,
                    system_lenght,
                    operations,
                    features_names,
                    MAX_DEPTH,
                    CONSTANT_PROBABILITY,
                    VARIABLE_PROBABILITY,
                    MAX_CONSTANT,
                    CHANGE_OPERATION_PROBABILITY,
                    DELETE_NODE_PROBABILITY,
                    ADD_OPERATION_PROBABILITY,
                    XOVER_PCT,
                    TOURNAMENT_SIZE,
                )
            )

        population = population_next_gen

    print(f"Best score: {global_best}")
    print(f"Best program:\n{render_prog(best_prog)}")

    return render_prog(best_prog)
