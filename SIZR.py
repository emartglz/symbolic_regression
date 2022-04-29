from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular

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
    S = X[0]
    I = X[1]
    Z = X[2]
    R = X[3]

    D_S = II - beta * S * Z - delta * S
    D_I = beta * S * Z - p * I - delta * I
    D_Z = p * I + sita * R - alpha * S * Z
    D_R = delta * S + delta * I + alpha * S * Z - sita * R

    return [D_S, D_I, D_Z, D_R]


def integrate_sir(time, samples, X0, alpha, beta, delta, sita, II, p):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(
        zombie_dx,
        X0,
        t,
        (alpha, beta, delta, sita, II, p),
        full_output=True,
    )

    X1, X2, X3, X4 = X.T

    return (t, X1, X2, X3, X4)


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

    t, X1, X2, X3, X4 = integrate_sir(time, n, X0, alpha, beta, delta, sita, II, p)

    ts = take_n_samples_regular(samples, t)
    X1s = take_n_samples_regular(samples, X1)
    X2s = take_n_samples_regular(samples, X2)
    X3s = take_n_samples_regular(samples, X3)
    X4s = take_n_samples_regular(samples, X4)

    X_samples = [[ts[i], X1s[i], X2s[i], X3s[i], X4s[i]] for i in range(len(ts))]

    ode = [
        zombie_dx(
            [X1s[i], X2s[i], X3s[i], X4s[i]],
            ts[i],
            alpha,
            beta,
            delta,
            sita,
            II,
            p,
        )
        for i in range(len(ts))
    ]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["t", "X0", "X1", "X2", "X3"],
        MAX_GENERATIONS=1000,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=1000,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=40,
    )

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {"t": t, "X0": X[0], "X1": X[1], "X2": X[2], "X3": X[3]},
    )

    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    X1_gp, X2_gp, X3_gp, X4_gp = X_gp.T

    plt.plot(t, X1, label="Susceptible population")
    plt.plot(t, X2, label="Infected population")
    plt.plot(t, X3, label="Zombie population")
    plt.plot(t, X4, label="Removed population")

    plt.plot(t, X1_gp, label="Susceptible population symbolic regression")
    plt.plot(t, X2_gp, label="Infected population symbolic regression")
    plt.plot(t, X3_gp, label="Zombie population symbolic regression")
    plt.plot(t, X4_gp, label="Removed population symbolic regression")

    plt.legend()
    plt.show()
