from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular


def lotka_volterra_dx(X, t, a, b, c, d):
    return [X[0] * (a - b * X[1]), -X[1] * (c - d * X[0])]


def integrate_lotka_volterra(time, samples, X0, a, b, c, d):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(
        lotka_volterra_dx, X0, t, (a, b, c, d), full_output=True
    )

    X1, X2 = X.T

    return (t, X1, X2)


def try_lotka_volterra():
    a = 0.04
    b = 0.0005
    c = 0.2
    d = 0.004

    x1_0 = x2_0 = 20
    X0 = [x1_0, x2_0]

    time = 1000
    n = 100000
    t, X1, X2 = integrate_lotka_volterra(time, n, X0, a, b, c, d)

    samples = 200
    ts = take_n_samples_regular(samples, t)
    X1s = take_n_samples_regular(samples, X1)
    X2s = take_n_samples_regular(samples, X2)
    X_samples = [[ts[i], X1s[i], X2s[i]] for i in range(len(ts))]
    ode = [
        lotka_volterra_dx([X1s[i], X2s[i]], ts[i], a, b, c, d) for i in range(len(ts))
    ]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["t", "X0", "X1"],
        MAX_GENERATIONS=100,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=100,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=20,
    )

    integrate_gp = lambda X, t: evaluate(best_system, {"t": t, "X0": X[0], "X1": X[1]})

    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    X1_gp, X2_gp = X_gp.T

    plt.plot(t, X1, label="X1 samples")
    plt.plot(t, X2, label="X2 samples")
    plt.plot(t, X1_gp, label="X1 symbolic regression")
    plt.plot(t, X2_gp, label="X2 symbolic regression")
    plt.legend()
    plt.show()
