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


def sir_dx(X, t, a, b):
    S, I, R = X

    return [-a * I * S, a * I * S - b * I, b * I]


def try_sir(noise, seed, name, save_to):
    a = 0.0003
    b = 0.1

    X0 = [700, 300, 0]

    time = 20
    n = 10000

    samples = 300
    symbolic_regression_samples = samples

    # noise = 0.1
    smoothing_factor = [1] * 3

    variable_names = ["t", "S", "I", "R"]

    t, *X = integrate_model(sir_dx, time, n, X0, a, b)

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

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=seed,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[["S", "I"], ["S", "I"], ["I"]],
        MUTATION_SIZE=50,
        XOVER_SIZE=50,
        MAX_DEPTH=5,
        REG_STRENGTH=20,
        # verbose=True,
    )

    # results = get_results("models_jsons/SIR")
    best_system = results["system"]
    save_results(results, f"{save_to}/SIR_{name}")

    integrate_gp = lambda X, t: evaluate(
        best_system, {"t": t, "S": X[0], "I": X[1], "R": X[2]}
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

    save_to = f"RESULTS/SIR/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_sir(noise, i, f"{i}", save_to)
