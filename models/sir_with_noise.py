from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from models.utils import add_noise, derivate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, get_results, save_results, take_n_samples_regular


def sir_dx(X, t, a, b):
    S = X[0]
    I = X[1]
    R = X[2]

    return [-a * I * S, a * I * S - b * I, b * I]


def integrate_sir(time, samples, X0, a, b):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(sir_dx, X0, t, (a, b), full_output=True)

    S, I, R = X.T

    return (t, S, I, R)


def try_sir():
    a = 0.3
    b = 0.4

    X0 = [0.7, 0.3, 0]

    time = 20
    n = 10000

    samples = 200

    noise = 0.01

    t, S, I, R = integrate_sir(time, n, X0, a, b)

    ts = take_n_samples_regular(samples, t)
    Ss = add_noise(take_n_samples_regular(samples, S), noise)
    Is = add_noise(take_n_samples_regular(samples, I), noise)
    Rs = add_noise(take_n_samples_regular(samples, R), noise)

    X_samples = [
        {"t": ts[i], "S": Ss[i], "I": Is[i], "R": Rs[i]} for i in range(len(ts))
    ]

    convert = [[Ss[i], Is[i], Rs[i]] for i in range(len(ts))]
    ode = derivate(ts, convert)

    results = symbolic_regression(
        X_samples[:-1],
        ode,
        seed_g=0,
        MAX_GENERATIONS=100,
        POP_SIZE=100,
        FEATURES_NAMES=[["S", "I"], ["S", "I"], ["I"]],
        MUTATION_SIZE=100,
        XOVER_SIZE=100,
        MAX_DEPTH=5,
        REG_STRENGTH=20,
        verbose=True,
    )

    # best_system = get_results("SIR")
    best_system = results["system"]
    save_results(results, "SIR")

    integrate_gp = lambda X, t: evaluate(
        best_system, {"t": t, "S": X[0], "I": X[1], "R": X[2]}
    )

    SIR_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    S_gp, I_gp, R_gp = SIR_gp.T

    plt.plot(ts, take_n_samples_regular(samples, S), label="S samples")
    plt.plot(ts, take_n_samples_regular(samples, I), label="I samples")
    plt.plot(ts, take_n_samples_regular(samples, R), label="R samples")
    plt.plot(ts, Ss, label="S samples noise")
    plt.plot(ts, Is, label="I samples noise")
    plt.plot(ts, Rs, label="R samples noise")
    plt.plot(t, S_gp, label="S symbolic regression")
    plt.plot(t, I_gp, label="I symbolic regression")
    plt.plot(t, R_gp, label="R symbolic regression")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try_sir()
