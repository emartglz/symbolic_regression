from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular

# X0 = SZ zombie population
# X1 = SW worker pupulation
# X2 = SM milita population
# X3 = SH mole population
# X4 = R total quantity of supplies
# N The ratio of the total geographic area to the area patrolled by a zombie over the course of one time unit Δt.
# Z1 probability average zombie convert a worker
# Z2 probability average zombie covert a mole (expect Z2 < Z1)
# Z3 probability average zombie covert a milita (expect Z3 > Z1)
# M1 probability milita member will cull a zombie
# M3 probability militia member will accidentally shot and kill another member of the milita
# alpha coefficient of cowardice(that a worker will join the mole community afters seeing a zombie)
# beta coufficient of bravado(that a worker will join the milita)
# f1 and f2 are equations discribing the community's response to a food shortage
# DFL days of food left
# R1 average number of normalized supply units found or produced by a worker over a unit ofo time
# R2 the number of normalizer supply units used by a milita member over a unit of time (R2 > 0)
# R3 the number of normalized supply units used by a mole (0 < R3 < 1)
# F4 population increase-rate per delta t per worker
# F5 probability that a worker will be shot accidentally a militia member (0 <= F5 <= 1)
# H3 probability that a militia member will accidentally shot and kill a member of the mole population


def dfl(R, Sw, Sm, Sh, R2, R3):
    return R / (Sw + (R2 + 1) * Sm + (1 - R3) * Sh)


def f1(dfl):
    if dfl < 4:
        return 1 / 2
    return 0


def f2(dfl):
    if dfl < 2:
        return 1 / 2
    return 0


def zombie_dx(
    X, t, N, Z1, Z2, Z3, M1, M3, alpha, beta, f1, f2, dfl, R1, R2, R3, F4, F5, H3
):
    Sz = X[0]
    Sw = X[1]
    Sm = X[2]
    Sh = X[3]
    R = X[4]
    DFL = dfl(R, Sw, Sm, Sh, R2, R3)

    D_Sz = Z1 * Sw * Sz / N + Z2 * Sh * Sz / N + (Z3 - M1) * Sm * Sz / N
    D_Sw = (
        -(Z1 + alpha * (1 - Z1) + beta * (1 - Z1 - alpha * (1 - Z1))) * Sz * Sw / N
        + f1(DFL) * Sm
        + f2(DFL) * Sh
        + F4 * Sw
        - F5 * Sw * Sm / N
    )
    D_Sm = (
        -Z3 * Sz * Sm / N
        + beta * (1 - Z1 - alpha * (1 - Z1)) * Sz * Sw / N
        - M3 * Sm * Sm / N
        - f1(DFL) * Sm
        - alpha * (1 - Z3) * Sz * Sm / N
        + beta * (1 - Z2) * Sh * Sz / N
    )
    D_Sh = (
        -Z2 * Sz * Sh / N
        + alpha * (1 - Z1) * Sz * Sw / N
        - H3 * Sh * Sm / N
        - f2(DFL) * Sh
        + alpha * (1 - Z3) * Sz * Sm / N
        - beta * (1 - Z2) * Sh * Sz / N
    )
    D_R = (R1 - 1) * Sw - (R2 + 1) * Sm - (1 - R3) * Sh
    return [D_Sz, D_Sw, D_Sm, D_Sh, D_R]


def integrate_sir(
    time,
    samples,
    X0,
    N,
    Z1,
    Z2,
    Z3,
    M1,
    M3,
    alpha,
    beta,
    f1,
    f2,
    dfl,
    R1,
    R2,
    R3,
    F4,
    F5,
    H3,
):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(
        zombie_dx,
        X0,
        t,
        (N, Z1, Z2, Z3, M1, M3, alpha, beta, f1, f2, dfl, R1, R2, R3, F4, F5, H3),
        full_output=True,
    )

    X1, X2, X3, X4, X5 = X.T

    return (t, X1, X2, X3, X4, X5)


def try_zombie():
    alpha = 0.1
    beta = 0.1
    M1 = 0.1
    Z1 = 0.1
    Z2 = 0.05
    Z3 = 0.15
    R1 = 3
    R2 = 1
    R3 = 0.4
    F4 = 6 * 1e-5
    F5 = 0.01
    M3 = 0.01
    H3 = 0.01
    N = 1000

    X0 = [1, 3000, 100, 0, 8000]

    time = 100
    n = 10000

    samples = 200

    t, X1, X2, X3, X4, X5 = integrate_sir(
        time,
        n,
        X0,
        N,
        Z1,
        Z2,
        Z3,
        M1,
        M3,
        alpha,
        beta,
        f1,
        f2,
        dfl,
        R1,
        R2,
        R3,
        F4,
        F5,
        H3,
    )

    plt.plot(t, X1, label="zombie population")
    plt.plot(t, X2, label="worker population")
    plt.plot(t, X3, label="milita population")
    plt.plot(t, X4, label="mole population")
    # plt.plot(t, X5, label="resources")
    plt.legend()
    plt.show()

    ts = take_n_samples_regular(samples, t)
    X1s = take_n_samples_regular(samples, X1)
    X2s = take_n_samples_regular(samples, X2)
    X3s = take_n_samples_regular(samples, X3)
    X4s = take_n_samples_regular(samples, X3)
    X5s = take_n_samples_regular(samples, X3)

    X_samples = [
        [ts[i], X1s[i], X2s[i], X3s[i], X4s[i], X5s[i]] for i in range(len(ts))
    ]

    ode = [
        zombie_dx(
            [X1s[i], X2s[i], X3s[i], X4s[i], X5s[i]],
            ts[i],
            N,
            Z1,
            Z2,
            Z3,
            M1,
            M3,
            alpha,
            beta,
            f1,
            f2,
            dfl,
            R1,
            R2,
            R3,
            F4,
            F5,
            H3,
        )
        for i in range(len(ts))
    ]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["t", "X0", "X1", "X2", "X3", "X4"],
        MAX_GENERATIONS=1000,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=1000,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=100,
    )

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {"t": t, "X0": X[0], "X1": X[1], "X2": X[2], "X3": X[3], "X4": X[4]},
    )

    X_gp, infodict = integrate.odeint(integrate_gp, X0, t, full_output=True)

    X1_gp, X2_gp, X3_gp, X4_gp, X5_gp = X_gp.T

    plt.plot(t, X1, label="zombie population")
    plt.plot(t, X2, label="worker population")
    plt.plot(t, X3, label="milita population")
    plt.plot(t, X4, label="mole population")
    # plt.plot(t, X5, label="resources")
    plt.plot(t, X1_gp, label="zombie population symbolic regression")
    plt.plot(t, X2_gp, label="worker population symbolic regression")
    plt.plot(t, X3_gp, label="militia population symbolic regression")
    plt.plot(t, X4_gp, label="mole population symbolic regression")
    # plt.plot(t, X5_gp, label="rosources symbolic regression")
    plt.legend()
    plt.show()
