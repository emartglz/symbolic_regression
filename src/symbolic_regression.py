import operator
from random import randint, random, seed
from functools import reduce
from math import *
from src.lineal_optimization import compute_fitness, lineal_optimization_system
from src.mutate import mutate_system
import timeit
from src.operation import ADD, DIV, MUL, NEG, SUB
from src.random_prog import random_system
from src.xover import xover
from src.utils import (
    evaluate,
    filter_zero_terms_edo_system,
    render_prog,
    round_terms_edo_system,
)


def get_random_parent(population):
    return population[randint(0, len(population) - 1)]


def get_mutate_population(
    population,
    MUTATION_SIZE,
    operations,
    features_names,
    MAX_DEPTH,
    VARIABLE_PROBABILITY,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
):
    mutations_populations = []
    for _ in range(MUTATION_SIZE):
        selected = get_random_parent(population)

        mutations_populations.append(
            mutate_system(
                selected=selected,
                operations=operations,
                features_names=features_names,
                MAX_DEPTH=MAX_DEPTH,
                VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
                CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
                DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
                ADD_OPERATION_PROBABILITY=ADD_OPERATION_PROBABILITY,
            )
        )

    return mutations_populations


def get_xover_population(population, XOVER_SIZE, MAX_DEPTH):
    xover_population = []
    for _ in range(XOVER_SIZE):
        parent1 = get_random_parent(population)
        parent2 = get_random_parent(population)

        xover_population.append(xover(parent1, parent2, MAX_DEPTH))

    return xover_population


def symbolic_regression(
    X,
    target,
    MAX_GENERATIONS=100,
    seed_g=random(),
    MAX_DEPTH=10,
    POP_SIZE=300,
    FEATURES_NAMES=None,
    VARIABLE_PROBABILITY=0.3,
    CHANGE_OPERATION_PROBABILITY=0.3,
    DELETE_NODE_PROBABILITY=0.3,
    ADD_OPERATION_PROBABILITY=0.4,
    XOVER_SIZE=100,
    MUTATION_SIZE=100,
    RANDOM_SELECTION_SIZE=0,
    REG_STRENGTH=5,
    EPSILON=1e-7,
    ROUND_SIZE=5,
    verbose=False,
):
    start = timeit.default_timer()

    seed(seed_g)
    system_lenght = len(target[0])

    features_names = FEATURES_NAMES or [list(X[0].keys()) for _ in range(system_lenght)]

    operations = (ADD, SUB, MUL, DIV, NEG)

    population = [
        random_system(
            system_lenght=system_lenght,
            operations=operations,
            features_names=features_names,
            MAX_DEPTH=MAX_DEPTH,
        )
        for _ in range(POP_SIZE)
    ]

    global_best = float("inf")
    gen = 0
    for gen in range(MAX_GENERATIONS):
        mutations_population = get_mutate_population(
            population=population,
            MUTATION_SIZE=MUTATION_SIZE,
            operations=operations,
            features_names=features_names,
            MAX_DEPTH=MAX_DEPTH,
            VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
            CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
            DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
            ADD_OPERATION_PROBABILITY=ADD_OPERATION_PROBABILITY,
        )

        xover_population = get_xover_population(
            population=population, XOVER_SIZE=XOVER_SIZE, MAX_DEPTH=MAX_DEPTH
        )

        total_population = population + mutations_population + xover_population

        fitness = []
        for i_prog, prog in enumerate(total_population):
            if verbose:
                print(f"{i_prog + 1}/{len(total_population)}", end="\r")

            optimized_program = prog
            optimized_program = lineal_optimization_system(
                system=prog, X=X, target=target
            )

            prediction = [evaluate(optimized_program, Xi) for Xi in X]
            score = compute_fitness(optimized_program, prediction, target, REG_STRENGTH)

            if score < global_best:
                global_best = score
                best_prog = optimized_program

            fitness.append(score)

        mean = sum(fitness) / len(fitness)

        if verbose:
            print(
                f"Generation: {gen + 1}\nBest Score: {global_best}\nMean score: {mean}\nBest program:\n{render_prog(best_prog)}\n"
            )

        if global_best < EPSILON:
            break

        member_fitness = [
            (fitness[i], i, total_population[i]) for i in range(len(total_population))
        ]
        member_fitness.sort()

        population = [
            i[2] for i in member_fitness[: (POP_SIZE - RANDOM_SELECTION_SIZE)]
        ] + [get_random_parent(total_population) for i in range(RANDOM_SELECTION_SIZE)]

    best_prog = round_terms_edo_system(system=best_prog, ROUND_SIZE=ROUND_SIZE)
    best_prog = filter_zero_terms_edo_system(system=best_prog)

    prediction = [evaluate(best_prog, Xi) for Xi in X]
    score = compute_fitness(best_prog, prediction, target, REG_STRENGTH)
    stop = timeit.default_timer()

    if verbose:
        print(f"Generations : {gen + 1}")
        print(f"Best score: {score}")
        print(f"Best program:\n{render_prog(best_prog)}")

        print("Time: ", stop - start)

    return {
        "system": best_prog,
        "system_representation": render_prog(best_prog),
        "generations": gen + 1,
        "score": score,
        "time": stop - start,
        "X": X,
        "target": target,
        "MAX_GENERATIONS": MAX_GENERATIONS,
        "seed_g": seed_g,
        "MAX_DEPTH": MAX_DEPTH,
        "POP_SIZE": POP_SIZE,
        "FEATURES_NAMES": features_names,
        "VARIABLE_PROBABILITY": VARIABLE_PROBABILITY,
        "CHANGE_OPERATION_PROBABILITY": CHANGE_OPERATION_PROBABILITY,
        "DELETE_NODE_PROBABILITY": DELETE_NODE_PROBABILITY,
        "ADD_OPERATION_PROBABILITY": ADD_OPERATION_PROBABILITY,
        "XOVER_SIZE": XOVER_SIZE,
        "MUTATION_SIZE": MUTATION_SIZE,
        "REG_STRENGTH": REG_STRENGTH,
        "EPSILON": EPSILON,
        "ROUND_SIZE": ROUND_SIZE,
    }
