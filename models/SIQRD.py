import os
import sys
from scipy import integrate
from models.utils import (
    add_noise_and_get_data,
    integrate_model,
    plot_data,
    separate_samples,
)
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results


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
    symbolic_regression_samples = samples

    # noise = 0.1
    smoothing_factor = [1] * 5

    variable_names = ["t", "S", "I", "Q", "R", "D"]

    t, *X = integrate_model(siqrd_system, time, n, X0, alpha, beta, delta, gamma, mu)

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
        i["N"] = i["S"] + i["I"] + i["Q"] + i["R"] + i["D"]

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=seed,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[
            ["S", "I", "Q", "N"],
            ["S", "I", "N"],
            ["S", "Q"],
            ["I"],
            ["I"],
        ],
        MUTATION_SIZE=50,
        XOVER_SIZE=50,
        MAX_DEPTH=10,
        REG_STRENGTH=30,
        # verbose=True,
    )

    # results = get_results("models_jsons/SIQRD")
    best_system = results["system"]
    save_results(results, f"{save_to}/SIQRD_{name}")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {
            "t": t,
            "S": X[0],
            "I": X[1],
            "Q": X[2],
            "R": X[3],
            "D": X[4],
            "N": X[0] + X[1] + X[2] + X[3] + X[4],
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

    save_to = f"RESULTS/SIQRD/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_siqrd(noise, i, f"{i}", save_to)
