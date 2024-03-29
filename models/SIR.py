import os
import sys
from models.utils import make_experiment


def sir_dx(X, t, a, b):
    S, I, R = X

    return [-a * I * S, a * I * S - b * I, b * I]


def try_sir(noise, seed, name, save_to):
    a = 0.3
    b = 0.1

    X0 = [0.7, 0.3, 0]

    time = 20
    n = 10000

    samples = 300

    variable_names = ["t", "S", "I", "R"]
    if noise == "original_model":
        smoothing_factor = None
    elif noise == 0:
        smoothing_factor = [1] * 3
    else:
        smoothing_factor = [0.1, 0.1, 0.1]

    make_experiment(
        sir_dx,
        X0,
        variable_names,
        smoothing_factor,
        noise,
        seed,
        name,
        save_to,
        [a, b],
        {
            "MAX_GENERATIONS": 100,
            "POP_SIZE": 100,
            "FEATURES_NAMES": [["S", "I"], ["S", "I"], ["I"]],
            "MUTATION_SIZE": 50,
            "XOVER_SIZE": 50,
            "MAX_DEPTH": 5,
            "REG_STRENGTH": 20,
            "RANDOM_SELECTION_SIZE": 10,
            # "verbose": True,
        },
        time=time,
        n=n,
        samples=samples,
        # show_spline=True,
    )


if __name__ == "__main__":
    r = 30

    noise = "original_model"
    if len(sys.argv) == 2:
        noise = float(sys.argv[1])

    save_to = f"RESULTS/SIR/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_sir(noise, i, f"SIR_{i}", save_to)
