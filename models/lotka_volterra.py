from matplotlib import pyplot as plt
from scipy import integrate
from models.utils import add_noise_and_get_data, integrate_model
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results, take_n_samples_regular


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

    t, *X = integrate_model(lotka_volterra_dx, time, n, X0, a, b, c, d)

    data = add_noise_and_get_data(
        t,
        X,
        samples,
        symbolic_regression_samples,
        noise,
        smoothing_factor,
        ["t", "X1", "X2"],
        0,
    )

    X_samples = data["X_samples"]
    ode = data["ode"]
    t = data["t"]
    X1, X2 = data["X"]

    t_noise = data["t_noise"]
    X1_noise, X2_noise = data["X_noise"]

    X1_spline, X2_spline = data["X_spline"]

    plt.plot(t_noise, take_n_samples_regular(samples, X1), label="X1 samples")
    plt.plot(t_noise, take_n_samples_regular(samples, X2), label="X2 samples")

    plt.plot(t_noise, X1_noise, ".", label="X1 samples noise")
    plt.plot(t_noise, X2_noise, ".", label="X2 samples noise")

    ts = list(map(lambda x: x["t"], data["X_samples"]))

    plt.plot(ts, [X1_spline(i) for i in ts], "--", label="X1 samples spline")
    plt.plot(ts, [X2_spline(i) for i in ts], "--", label="X2 samples spline")

    plt.legend()
    plt.show()

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

    # results = get_results("LV2")
    best_system = results["system"]
    save_results(results, "LV2")

    integrate_gp = lambda X, t: evaluate(best_system, {"t": t, "X1": X[0], "X2": X[1]})
    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)
    X1_gp, X2_gp = X_gp.T

    plt.plot(t_noise, take_n_samples_regular(samples, X1), label="X1 samples")
    plt.plot(t_noise, take_n_samples_regular(samples, X2), label="X2 samples")

    plt.plot(t_noise, X1_noise, ".", label="X1 samples noise")
    plt.plot(t_noise, X2_noise, ".", label="X2 samples noise")

    ts = list(map(lambda x: x["t"], data["X_samples"]))

    plt.plot(ts, [X1_spline(i) for i in ts], "--", label="X1 samples spline")
    plt.plot(ts, [X2_spline(i) for i in ts], "--", label="X2 samples spline")

    plt.plot(t, X1_gp, "-.", label="X1 symbolic regression")
    plt.plot(t, X2_gp, "-.", label="X2 symbolic regression")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    try_lotka_volterra()
