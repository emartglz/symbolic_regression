from random import randint, random, seed
from src.aproximation import generate_dx
from src.genetic_algorithm import genetic_algorithm
from src.utils import group_with_names, group_without_names
from matplotlib import pyplot as plt


def symbolic_regression(
    X,
    variable_names,
    smoothing_factor,
    add_N=False,
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
    show_spline=False,
):

    X_less_last_element = [x[:-1] for x in X]
    X_samples = group_with_names(X_less_last_element, variable_names)

    X_dx = generate_dx(X[0], X[1:], smoothing_factor)

    if show_spline:
        for i, variable_name in enumerate(variable_names[1:]):
            plt.plot(
                X[0][:-1],
                X_dx[i],
                "--",
                label=f"{variable_name} samples spline",
            )
        plt.legend()
        plt.show()

    target = group_without_names(X_dx)
    if add_N:
        for i in X_samples:
            sum = 0
            for j in variable_names[1:]:
                sum += i[j]
            i["N"] = sum

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
