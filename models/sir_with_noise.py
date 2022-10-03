from matplotlib import pyplot as plt
from scipy import integrate
from models.utils import add_noise_and_get_data, integrate_model
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results, take_n_samples_regular


def sir_dx(X, t, a, b):
    S, I, R = X

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

    t, *X = integrate_model(sir_dx, time, n, X0, a, b)

    data = add_noise_and_get_data(
        t,
        X,
        samples,
        symbolic_regression_samples,
        noise,
        smoothing_factor,
        ["t", "S", "I", "R"],
        0,
    )

    X_samples = data["X_samples"]
    ode = data["ode"]
    t = data["t"]
    S, I, R = data["X"]

    t_noise = data["t_noise"]
    S_noise, I_noise, R_noise = data["X_noise"]

    S_spline, I_spline, R_spline = data["X_spline"]

    plt.plot(t_noise, take_n_samples_regular(samples, S), label="S samples")
    plt.plot(t_noise, take_n_samples_regular(samples, I), label="I samples")
    plt.plot(t_noise, take_n_samples_regular(samples, R), label="R samples")

    plt.plot(t_noise, S_noise, ".", label="S samples noise")
    plt.plot(t_noise, I_noise, ".", label="I samples noise")
    plt.plot(t_noise, R_noise, ".", label="R samples noise")

    ts = list(map(lambda x: x["t"], data["X_samples"]))

    plt.plot(ts, [S_spline(i) for i in ts], "--", label="S samples spline")
    plt.plot(ts, [I_spline(i) for i in ts], "--", label="I samples spline")
    plt.plot(ts, [R_spline(i) for i in ts], "--", label="R samples spline")

    plt.legend()
    plt.show()

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

    # results = get_results("SIR_noise")
    best_system = results["system"]
    save_results(results, "SIR_noise")

    integrate_gp = lambda X, t: evaluate(
        best_system, {"t": t, "S": X[0], "I": X[1], "R": X[2]}
    )

    SIR_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    S_gp, I_gp, R_gp = SIR_gp.T

    plt.plot(t_noise, take_n_samples_regular(samples, S), label="S samples")
    plt.plot(t_noise, take_n_samples_regular(samples, I), label="I samples")
    plt.plot(t_noise, take_n_samples_regular(samples, R), label="R samples")

    plt.plot(t_noise, S_noise, ".", label="S samples noise")
    plt.plot(t_noise, I_noise, ".", label="I samples noise")
    plt.plot(t_noise, R_noise, ".", label="R samples noise")

    ts = list(map(lambda x: x["t"], data["X_samples"]))

    plt.plot(ts, [S_spline(i) for i in ts], "--", label="S samples spline")
    plt.plot(ts, [I_spline(i) for i in ts], "--", label="I samples spline")
    plt.plot(ts, [R_spline(i) for i in ts], "--", label="R samples spline")

    plt.plot(t, S_gp, "-.", label="S symbolic regression")
    plt.plot(t, I_gp, "-.", label="I symbolic regression")
    plt.plot(t, R_gp, "-.", label="R symbolic regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try_sir()
