from scipy import integrate
from models.utils import (
    add_noise_and_get_data,
    integrate_model,
    plot_data,
    separate_samples,
)
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, save_results


def sir_dx(X, t, a, b):
    S = X[0]
    I = X[1]
    R = X[2]

    return [-a * I * S, a * I * S - b * I, b * I]


def try_sir():
    a = 0.0003
    b = 0.1

    X0 = [700, 300, 0]

    time = 20
    n = 10000

    samples = 200
    symbolic_regression_samples = samples

    noise = 0.1
    smoothing_factor = [symbolic_regression_samples * 10000] * 3

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
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[["S", "I"], ["S", "I"], ["I"]],
        MUTATION_SIZE=50,
        XOVER_SIZE=50,
        MAX_DEPTH=5,
        REG_STRENGTH=20,
        verbose=True,
    )

    # results = get_results("SIR")
    best_system = results["system"]
    save_results(results, "SIR")

    integrate_gp = lambda X, t: evaluate(
        best_system, {"t": t, "S": X[0], "I": X[1], "R": X[2]}
    )

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
    try_sir()
