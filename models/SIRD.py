from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, save_results, take_n_samples_regular

# S' = a - b* S * I/(S+I+R)
# I' = b* S * I/(S+I+R) - c * I - d * I
# R' = c* I
# D' = d * I
def sird_dx(X, t, a, b, c, d):
    S = X[0]
    I = X[1]
    R = X[2]
    D = X[3]

    S_d = a - b * S * I / (S + I + R)
    I_d = b * S * I / (S + I + R) - c * I - d * I
    R_d = c * I
    D_d = d * I

    return [S_d, I_d, R_d, D_d]


def integrate_sird(time, samples, X0, a, b, c, d):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(sird_dx, X0, t, (a, b, c, d), full_output=True)

    S, I, R, D = X.T

    return (t, S, I, R, D)


def try_sird():
    a = 250
    b = 0.5
    c = 0.1
    d = 0.2

    X0 = [7000, 3000, 0, 0]

    time = 20
    n = 10000

    samples = 200

    t, S, I, R, D = integrate_sird(time, n, X0, a, b, c, d)

    # plt.plot(t, S, label="S samples")
    # plt.plot(t, I, label="I samples")
    # plt.plot(t, R, label="R samples")
    # plt.plot(t, D, label="D samples")
    # plt.legend()
    # plt.show()

    ts = take_n_samples_regular(samples, t)
    Ss = take_n_samples_regular(samples, S)
    Is = take_n_samples_regular(samples, I)
    Rs = take_n_samples_regular(samples, R)
    Ds = take_n_samples_regular(samples, D)

    X_samples = [
        {
            "t": ts[i],
            "S": Ss[i],
            "I": Is[i],
            "R": Rs[i],
            "D": Ds[i],
            "N": Ss[i] + Is[i] + Rs[i],
        }
        for i in range(len(ts))
    ]

    ode = [
        sird_dx([Ss[i], Is[i], Rs[i], Ds[i]], ts[i], a, b, c, d) for i in range(len(ts))
    ]

    results = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        MAX_GENERATIONS=1000,
        POP_SIZE=100,
        FEATURES_NAMES=[["S", "I", "N"], ["S", "I", "N"], ["I"], ["I"]],
        MUTATION_SIZE=500,
        XOVER_SIZE=500,
        MAX_DEPTH=10,
        REG_STRENGTH=50,
        verbose=True,
    )

    best_system = results["system"]
    save_results(results, "SIRD")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {"t": t, "S": X[0], "I": X[1], "R": X[2], "D": X[3], "N": X[0] + X[1] + X[2]},
    )

    SIR_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    S_gp, I_gp, R_gp, D_gp = SIR_gp.T

    plt.plot(t, S, label="S samples")
    plt.plot(t, I, label="I samples")
    plt.plot(t, R, label="R samples")
    plt.plot(t, D, label="D samples")
    plt.plot(t, S_gp, ":", label="S symbolic regression")
    plt.plot(t, I_gp, ":", label="I symbolic regression")
    plt.plot(t, R_gp, ":", label="R symbolic regression")
    plt.plot(t, D_gp, ":", label="D symbolic regression")
    plt.legend()
    plt.show()
