import os
import sys
from scipy import integrate
from models.utils import (
    add_noise_and_get_data,
    integrate_model,
    join_samples,
    plot_data,
    separate_samples,
)
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results, save_samples


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
    symbolic_regression_samples = samples

    # noise = 0.1
    smoothing_factor = [1] * 6

    variable_names = ["t", "S", "V1", "V2", "E", "I", "R"]

    t, *X = integrate_model(
        svveir_system,
        time,
        n_integration,
        X0,
        alpha,
        beta,
        delta,
        gamma,
        mu,
        n,
        roh,
        omega,
        sigma,
    )

    data = add_noise_and_get_data(
        t,
        X,
        samples,
        symbolic_regression_samples,
        noise,
        smoothing_factor,
        variable_names,
        seed,
    )
    X_samples = data["X_samples"]
    ode = data["ode"]

    t_spline, *X_spline = separate_samples(variable_names, X_samples)

    save_samples(
        join_samples(variable_names, [data["t_noise"]] + data["X_noise"]),
        f"{save_to}/data_{name}",
    )

    plot_data(
        variables_names=variable_names[1:],
        t_samples=data["t"],
        samples=data["X"],
        t_noise=data["t_noise"],
        samples_noise=data["X_noise"],
        t_spline=t_spline,
        samples_spline=X_spline,
        name=f"{save_to}/initial_plot_{name}.svg",
    )

    for i in X_samples:
        i["N"] = i["S"] + i["V1"] + i["V2"] + i["E"] + i["I"] + i["R"]

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=seed,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[
            ["N", "I", "S", "V1"],
            ["S", "V1"],
            ["V1", "V2"],
            ["I", "N", "S", "E"],
            ["E", "I"],
            ["I", "V2", "R"],
        ],
        MUTATION_SIZE=50,
        XOVER_SIZE=50,
        MAX_DEPTH=10,
        REG_STRENGTH=30,
        # verbose=True,
    )

    # results = get_results("models_jsons/SVVEIR")
    best_system = results["system"]
    save_results(results, f"{save_to}/SVVEIR_{name}")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {
            "t": t,
            "S": X[0],
            "V1": X[1],
            "V2": X[2],
            "E": X[3],
            "I": X[4],
            "R": X[5],
            "N": X[0] + X[1] + X[2] + X[3] + X[4] + X[5],
        },
    )

    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)
    X_gp = X_gp.T.tolist()

    plot_data(
        variables_names=variable_names[1:],
        t_samples=data["t"],
        samples=data["X"],
        # t_noise=data["t_noise"],
        # samples_noise=data["X_noise"],
        # t_spline=t_spline,
        # samples_spline=X_spline,
        t_symbolic_regression=t,
        samples_symbolic_regression=X_gp,
        name=f"{save_to}/final_plot_{name}.svg",
    )


if __name__ == "__main__":
    r = 30
    noise = float(sys.argv[1])

    save_to = f"RESULTS/SVVEIR/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_svveir(noise, i, f"{i}", save_to)
