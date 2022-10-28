import os
import sys

from models.utils import make_experiment


def siqrd_system(X, t, alpha, beta, delta, gamma, mu):
    # alpha - rate at which susceptible people are sent to social isolation
    # beta  - transmission rate
    # delta - rate at which individuals in social isolation return to the susceptible class
    # gamma - recovery rate
    # mu    - mortality rate
    S, I, Q, R, D = X

    N = S + I + Q + R + D

    S_d = -beta * S * I / N - alpha * S + delta * Q
    I_d = beta * S * I / N - gamma * I - mu * I
    Q_d = alpha * S - delta * Q
    R_d = gamma * I
    D_d = mu * I

    return [S_d, I_d, Q_d, R_d, D_d]


def try_siqrd(noise, seed, name, save_to):
    alpha = 0.2
    beta = 0.9
    delta = 0.1
    gamma = 0.1
    mu = 0.05

    X0 = [5000, 3000, 1000, 0, 0]

    time = 20
    n = 10000

    samples = 300

    # smoothing_factor = [1] * 5
    smoothing_factor = [0.1] * 5

    variable_names = ["t", "S", "I", "Q", "R", "D"]

    make_experiment(
        siqrd_system,
        X0,
        variable_names,
        smoothing_factor,
        noise,
        seed,
        name,
        save_to,
        [alpha, beta, delta, gamma, mu],
        {
            "MAX_GENERATIONS": 100,
            "POP_SIZE": 100,
            "FEATURES_NAMES": [
                ["S", "I", "Q", "N"],
                ["S", "I", "N"],
                ["S", "Q"],
                ["I"],
                ["I"],
            ],
            "MUTATION_SIZE": 50,
            "XOVER_SIZE": 50,
            "MAX_DEPTH": 10,
            "REG_STRENGTH": 30,
            "RANDOM_SELECTION_SIZE": 10,
            # "verbose": True,
        },
        add_N=True,
        time=time,
        n=n,
        samples=samples,
        # show_spline=True,
    )


if __name__ == "__main__":
    r = 30
    noise = float(sys.argv[1])

    save_to = f"RESULTS/SIQRD/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_siqrd(noise, i, f"SIRQD_{i}", save_to)
