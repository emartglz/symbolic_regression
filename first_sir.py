from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular


def sir_dx(X, t, a, b):
    return [-a * X[1] * X[0], a * X[1] * X[0] - b * X[1], b * X[1]]


def integrate_sir(time, samples, X0, a, b):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(sir_dx, X0, t, (a, b), full_output=True)

    X1, X2, X3 = X.T

    return (t, X1, X2, X3)


def try_sir():
    a = 0.3
    b = 0.4

    X0 = [0.7, 0.3, 0]

    time = 20
    n = 10000

    samples = 200

    t, X1, X2, X3 = integrate_sir(time, n, X0, a, b)

    ts = take_n_samples_regular(samples, t)
    X1s = take_n_samples_regular(samples, X1)
    X2s = take_n_samples_regular(samples, X2)
    X3s = take_n_samples_regular(samples, X3)

    X_samples = [[ts[i], X1s[i], X2s[i], X3s[i]] for i in range(len(ts))]

    ode = [sir_dx([X1s[i], X2s[i], X3s[i]], ts[i], a, b) for i in range(len(ts))]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["t", "X0", "X1", "X2"],
        MAX_GENERATIONS=100,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=100,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=20,
    )

    integrate_gp = lambda X, t: evaluate(
        best_system, {"t": t, "X0": X[0], "X1": X[1], "X2": X[2]}
    )

    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    X1_gp, X2_gp, X3_gp = X_gp.T

    plt.plot(t, X1, label="X1 samples")
    plt.plot(t, X2, label="X2 samples")
    plt.plot(t, X3, label="X3 samples")
    plt.plot(t, X1_gp, label="X1 symbolic regression")
    plt.plot(t, X2_gp, label="X2 symbolic regression")
    plt.plot(t, X3_gp, label="X3 symbolic regression")
    plt.legend()
    plt.show()
