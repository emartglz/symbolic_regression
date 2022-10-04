from matplotlib import pyplot as plt
from scipy import integrate
from models.utils import (
    add_noise_and_get_data,
    integrate_model,
    plot_data,
    separate_samples,
)
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results


def lotka_volterra_dx(X, t, a, b, c, d):
    return [X[0] * (a - b * X[1]), -X[1] * (c - d * X[0])]


def try_lotka_volterra():
    a = 0.04
    b = 0.0005
    c = 0.2
    d = 0.004

    x1_0 = x2_0 = 20
    X0 = [x1_0, x2_0]

    time = 300
    n = 100000

    samples = 300
    symbolic_regression_samples = samples

    noise = 0.1
    smoothing_factor = [
        symbolic_regression_samples * 100,
        symbolic_regression_samples * 1000,
    ]

    variable_names = ["t", "X1", "X2"]

    t, *X = integrate_model(lotka_volterra_dx, time, n, X0, a, b, c, d)

    data = add_noise_and_get_data(
        t,
        X,
        samples,
        symbolic_regression_samples,
        noise,
        smoothing_factor,
        variable_names,
        0,
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
    )

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        MAX_GENERATIONS=200,
        POP_SIZE=100,
        XOVER_SIZE=50,
        MUTATION_SIZE=50,
        MAX_DEPTH=10,
        REG_STRENGTH=15,
        verbose=True,
    )

    # results = get_results("LV")
    best_system = results["system"]
    save_results(results, "LV")

    integrate_gp = lambda X, t: evaluate(best_system, {"t": t, "X1": X[0], "X2": X[1]})
    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)
    X_gp = X_gp.T.tolist()

    plot_data(
        variables_names=variable_names[1:],
        t_samples=data["t"],
        samples=data["X"],
        t_noise=data["t_noise"],
        samples_noise=data["X_noise"],
        t_spline=t_spline,
        samples_spline=X_spline,
        t_symbolic_regression=t,
        samples_symbolic_regression=X_gp,
    )


if __name__ == "__main__":
    try_lotka_volterra()
