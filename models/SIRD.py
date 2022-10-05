from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from models.utils import (
    add_noise_and_get_data,
    integrate_model,
    plot_data,
    separate_samples,
)
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results

# S' = a - b* S * I/(S+I+R)
# I' = b* S * I/(S+I+R) - c * I - d * I
# R' = c* I
# D' = d * I
def sird_dx(X, t, a, b, c, d):
    S, I, R, D = X

    S_d = a - b * S * I / (S + I + R)
    I_d = b * S * I / (S + I + R) - c * I - d * I
    R_d = c * I
    D_d = d * I

    return [S_d, I_d, R_d, D_d]


def try_sird():
    a = 250
    b = 0.5
    c = 0.1
    d = 0.2

    X0 = [7000, 3000, 0, 0]

    time = 20
    n = 10000

    samples = 200
    symbolic_regression_samples = samples

    noise = 0.1
    smoothing_factor = [
        symbolic_regression_samples * 1000000,
        symbolic_regression_samples * 100000,
        symbolic_regression_samples * 100000,
        symbolic_regression_samples * 1000000,
    ]

    variable_names = ["t", "S", "I", "R", "D"]

    t, *X = integrate_model(sird_dx, time, n, X0, a, b, c, d)

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

    for i in X_samples:
        i["N"] = i["S"] + i["I"] + i["R"]

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[["S", "I", "N"], ["S", "I", "N"], ["I"], ["I"]],
        MUTATION_SIZE=50,
        XOVER_SIZE=50,
        MAX_DEPTH=10,
        REG_STRENGTH=50,
        verbose=True,
    )

    # results = get_results("models_jsons/SIRD")
    best_system = results["system"]
    save_results(results, "models_jsons/SIRD")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {"t": t, "S": X[0], "I": X[1], "R": X[2], "D": X[3], "N": X[0] + X[1] + X[2]},
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
    try_sird()
