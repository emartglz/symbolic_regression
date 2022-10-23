import os
from matplotlib import pyplot as plt
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

    time = 300
    n = 100000

    samples = 300
    symbolic_regression_samples = 300

    # noise = 0
    smoothing_factor = [1, 1]

    variable_names = ["t", "X", "Y"]

    t, *X = integrate_model(lotka_volterra_dx, time, n, X0, a, b, c, d)

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

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=seed,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        XOVER_SIZE=50,
        MUTATION_SIZE=50,
        RANDOM_SELECTION_SIZE=20,
        MAX_DEPTH=10,
        REG_STRENGTH=15,
        # verbose=True,
    )

    # results = get_results("models_jsons/LV")
    best_system = results["system"]
    save_results(results, f"{save_to}/LV_{name}")

    integrate_gp = lambda X, t: evaluate(best_system, {"t": t, "X": X[0], "Y": X[1]})
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

    save_to = f"RESULTS/LV/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i in range(r):
        print(i)
        try_lotka_volterra(noise, i, f"{i}", save_to)
