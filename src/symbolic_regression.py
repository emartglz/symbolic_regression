from random import randint, random, seed
from src.aproximation import generate_dx
from src.genetic_algorithm import genetic_algorithm
from src.utils import group_with_names, group_without_names


def symbolic_regression(
    X,
    variable_names,
    smoothing_factor,
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

    X_less_last_element = [x[:-1] for x in X]
    X_samples = group_with_names(X_less_last_element, variable_names)
    target = group_without_names(generate_dx(X[0], X[1:], smoothing_factor))

    ret = genetic_algorithm(
        X_samples,
        target,
        MAX_GENERATIONS,
        seed_g,
        MAX_DEPTH,
        POP_SIZE,
        FEATURES_NAMES,
        VARIABLE_PROBABILITY,
        CHANGE_OPERATION_PROBABILITY,
        DELETE_NODE_PROBABILITY,
        ADD_OPERATION_PROBABILITY,
        XOVER_SIZE,
        MUTATION_SIZE,
        RANDOM_SELECTION_SIZE,
        REG_STRENGTH,
        EPSILON,
        ROUND_SIZE,
        verbose,
    )

    return ret
