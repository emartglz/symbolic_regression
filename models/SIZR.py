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
from src.utils import evaluate, get_results, save_results, take_n_samples_regular

# S Susceptible
# I Infected
# Z Zombie
# R Removed
# alpha Death rate of zombies (caused by destroying its brain or removing its head)
# beta Transmission rate
# delta Death rate of susceptible humans by natural causes (i.e. non-zombie related)
# sita Resurrection rate (susceptible to zombie)
# II Birthrate
# p rate of infection
def zombie_dx(X, t, alpha, beta, delta, sita, II, p):
    S, I, Z, R = X

    D_S = II - beta * S * Z - delta * S
    D_I = beta * S * Z - p * I - delta * I
    D_Z = p * I + sita * R - alpha * S * Z
    D_R = delta * S + delta * I + alpha * S * Z - sita * R

    return [D_S, D_I, D_Z, D_R]


def try_zombie_SIZR():
    alpha = 0.005
    beta = 0.095
    delta = 0.0001
    sita = 0.001
    II = 0
    p = 0.05

    X0 = [500, 0, 1, 0]

    time = 50
    n = 10000

    samples = 200
    symbolic_regression_samples = samples

    noise = 0.1
    smoothing_factor = [
        symbolic_regression_samples * 100,
        symbolic_regression_samples * 500,
        symbolic_regression_samples * 10000,
        symbolic_regression_samples * 1000,
    ]

    variable_names = ["t", "S", "I", "Z", "R"]

    t, *X = integrate_model(zombie_dx, time, n, X0, alpha, beta, delta, sita, II, p)

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
        FEATURES_NAMES=["t", "S", "I", "Z", "R"],
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        XOVER_SIZE=50,
        MUTATION_SIZE=50,
        MAX_DEPTH=10,
        REG_STRENGTH=40,
        verbose=True,
    )

    results = get_results("models_jsons/SIZR")
    best_system = results["system"]
    save_results(results, "models_jsons/SIZR")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {"t": t, "S": X[0], "I": X[1], "Z": X[2], "R": X[3]},
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
    try_zombie_SIZR()
