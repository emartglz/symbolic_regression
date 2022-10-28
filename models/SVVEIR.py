import os
import sys

from models.utils import make_experiment


def svveir_system(X, t, alpha, beta, delta, gamma, mu, n, roh, omega, sigma):
    # alpha - proportion of the exposed progress to the infected
    # beta  - transmission rate
    # delta - measles-related death rate
    # gamma - natural recovery rate
    # mu    - natural death rate
    # n     - rate of susceptibles who receive the first dose of vaccine move to the vaccinated
    # roh   - rate of first dose of vaccinated (V1) moves to the susceptibles
    # omega - recovery rate of second dose of the vaccinated
    # sigma - rate of first dose of vaccinated (V1) moves to the second dose of vaccinated (V2)

    S, V1, V2, E, I, R = X
    N = S + V1 + V2 + E + I + R

    S_d = mu * N - beta * I / N * S - n * S - mu * S + roh * V1
    V1_d = n * S - roh * V1 - sigma * V1 - mu * V1
    V2_d = sigma * V1 - omega * V2 - mu * V2
    E_d = beta * I / N * S - alpha * E - mu * E
    I_d = alpha * E - gamma * I - delta * I - mu * I
    R_d = gamma * I + omega * V2 - mu * R

    return [S_d, V1_d, V2_d, E_d, I_d, R_d]


def try_svveir(noise, seed, name, save_to):
    alpha = 0.1
    beta = 0.7
    delta = 0.0005
    gamma = 0.05
    mu = 0.01
    n = 0.2
    roh = 0.01
    omega = 0.05
    sigma = 0.2

    X0 = [5000, 1000, 0, 2000, 1000, 500]

    time = 20
    n_integration = 10000

    samples = 300

    if noise == 0:
        smoothing_factor = [1] * 6
    else:
        smoothing_factor = [0.1] * 6

    variable_names = ["t", "S", "V1", "V2", "E", "I", "R"]

    make_experiment(
        svveir_system,
        X0,
        variable_names,
        smoothing_factor,
        noise,
        seed,
        name,
        save_to,
        [alpha, beta, delta, gamma, mu, n, roh, omega, sigma],
        {
            "MAX_GENERATIONS": 100,
            "POP_SIZE": 100,
            "FEATURES_NAMES": [
                ["N", "I", "S", "V1"],
                ["S", "V1"],
                ["V1", "V2"],
                ["I", "N", "S", "E"],
                ["E", "I"],
                ["I", "V2", "R"],
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
        n=n_integration,
        samples=samples,
        # show_spline=True,
    )


if __name__ == "__main__":
    r = 30
    noise = float(sys.argv[1])

    save_to = f"RESULTS/SVVEIR/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_svveir(noise, i, f"SVVEIR_{i}", save_to)
