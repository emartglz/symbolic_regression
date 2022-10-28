import os
from models.utils import make_experiment
import sys


def lotka_volterra_dx(X, t, a, b, c, d):
    return [X[0] * (a - b * X[1]), -X[1] * (c - d * X[0])]


def try_lotka_volterra(noise, seed, name, save_to):
    a = 0.04
    b = 0.0005
    c = 0.2
    d = 0.004

    x1_0 = x2_0 = 20
    X0 = [x1_0, x2_0]

    variable_names = ["t", "X", "Y"]
    if noise == 0:
        smoothing_factor = [1] * len(variable_names[1:])
    else:
        smoothing_factor = [0.005, 0.1]

    make_experiment(
        lotka_volterra_dx,
        X0,
        variable_names,
        smoothing_factor,
        noise,
        seed,
        name,
        save_to,
        [a, b, c, d],
        {
            "MAX_GENERATIONS": 100,
            "POP_SIZE": 100,
            "XOVER_SIZE": 50,
            "MUTATION_SIZE": 50,
            "RANDOM_SELECTION_SIZE": 10,
            "MAX_DEPTH": 10,
            "REG_STRENGTH": 15,
            # "verbose": True,
        },
        # show_spline=True,
    )


if __name__ == "__main__":
    r = 30
    noise = float(sys.argv[1])

    save_to = f"RESULTS/LV/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_lotka_volterra(noise, i, f"LV_{i}", save_to)
