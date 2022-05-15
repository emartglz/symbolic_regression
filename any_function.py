import math
from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular


def f(x, a, b, c, d):
    return a * x + b * x**2 + c * x**3 + d * x**4


def df(x, a, b, c, d):
    return a + 2 * b * x + 3 * c * x**2 + d * x**3


def try_any_function():
    a = 2500
    b = -175
    c = 2
    d = -0.006

    time = 200
    n = 100000

    x = linspace(0, time, n)
    y = [f(i, a, b, c, d) for i in x]

    samples = 501

    xs = take_n_samples_regular(samples, x)
    ys = take_n_samples_regular(samples, y)

    X_samples = [[xs[i], ys[i]] for i in range(len(xs) - 1)]
    ode = [[(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])] for i in range(len(xs) - 1)]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["x", "y"],
        MAX_GENERATIONS=100,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=100,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=20,
    )

    integrate_gp = lambda X, x: evaluate(best_system, {"x": x, "y": X[0]})

    X_gp, infodict = integrate.odeint(integrate_gp, 0, x, full_output=True)

    X1_gp = X_gp.T

    plt.plot(x, y, label="samples")
    plt.plot(x, X1_gp[0], label="symbolic regression")
    plt.legend()
    plt.show()
