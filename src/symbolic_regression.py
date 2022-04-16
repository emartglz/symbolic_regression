import operator
from random import randint, random, seed
from functools import reduce
from math import *
from src.lineal_optimization import compute_fitness, lineal_optimization_system
from src.mutate import mutate_system

from src.operation import ADD, DIV, MUL, NEG, SUB
from src.random_prog import random_system
from src.xover import xover
from src.utils import evaluate, render_prog


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
    operations,
    features_names,
    MAX_DEPTH,
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


def symbolic_regression(
    X,
    target,
    MAX_GENERATIONS=100,
    N_GENERATION_OPTIMIZE=1,
    seed_g=random(),
    MAX_DEPTH=10,
    POP_SIZE=300,
    FEATURES_NAMES=None,
    VARIABLE_PROBABILITY=0.3,
    MAX_CONSTANT=100,
    CHANGE_OPERATION_PROBABILITY=0.3,
    DELETE_NODE_PROBABILITY=0.3,
    ADD_OPERATION_PROBABILITY=0.4,
    TOURNAMENT_SIZE=3,
    XOVER_PCT=0.5,
    REG_STRENGTH=5,
    EPSILON=1e-7,
    PROPORTION_OF_BESTS=1 / 3,
):
    seed(seed_g)
    feature_lenght = len(X[0])
    system_lenght = len(target[0])

    features_names = FEATURES_NAMES or [f"X{i + 1}" for i in range(feature_lenght)]

    X = [dict(zip(features_names, x)) for x in X]

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

            optimized_program = prog

            if gen % N_GENERATION_OPTIMIZE == 0:
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
                    population=population,
                    fitness=fitness,
                    operations=operations,
                    features_names=features_names,
                    MAX_DEPTH=MAX_DEPTH,
                    VARIABLE_PROBABILITY=VARIABLE_PROBABILITY,
                    MAX_CONSTANT=MAX_CONSTANT,
                    CHANGE_OPERATION_PROBABILITY=CHANGE_OPERATION_PROBABILITY,
                    DELETE_NODE_PROBABILITY=DELETE_NODE_PROBABILITY,
                    ADD_OPERATION_PROBABILITY=ADD_OPERATION_PROBABILITY,
                    XOVER_PCT=XOVER_PCT,
                    TOURNAMENT_SIZE=TOURNAMENT_SIZE,
                )
            )

        population = population_next_gen

    print(f"Best score: {global_best}")
    print(f"Best program:\n{render_prog(best_prog)}")

    return best_prog
