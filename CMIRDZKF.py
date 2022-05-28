from matplotlib import pyplot as plt
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.symbolic_regression import symbolic_regression
from src.utils import evaluate, take_n_samples_regular

# X0 C Collector population
# X1 M Millitia population
# X2 I Infected population
# X3 R Recupered population
# X4 D Dead population
# X5 Z Zombie population
# X6 K Kill population
# X7 F Food amount
# alpha_c Z infect C
# alpha_m Z infect M
# beta_m_to_c M convert into C
# beta_c_to_m C convert into M
# nd natura death
# ganma_r I convert into R
# ganma_d I convert into D
# ganma_z I convert into Z
# zeta D convert into Z
# omega milita kill zombie
# delta_f amount of food found
# delta_u amount of food used


def beta_m_to_c(F, C, M):
    # return (1 / (F / (M + C))) / (F * M * C)
    return 0


def beta_c_to_m(F, C, M):
    # return (1 / (F / (M + C))) / (F * M * C)
    return 0


def CMIRDZKF(
    X,
    t,
    alpha_c,
    alpha_m,
    beta_m_to_c,
    beta_c_to_m,
    nd,
    ganma_r,
    ganma_d,
    ganma_z,
    zeta,
    omega,
    delta_f,
    delta_u,
):
    C = X[0]
    M = X[1]
    I = X[2]
    R = X[3]
    D = X[4]
    Z = X[5]
    K = X[6]
    F = X[7]

    beta_evaluated_m_to_c = beta_m_to_c(F, C, M)
    beta_evaluated_c_to_m = beta_c_to_m(F, C, M)

    DC = (
        -1 * alpha_c * C * Z
        + beta_evaluated_m_to_c * M
        - beta_evaluated_c_to_m * C
        - nd * C
    )
    DM = (
        -1 * alpha_m * M * Z
        - beta_evaluated_m_to_c * M
        + beta_evaluated_c_to_m * C
        - nd * M
    )
    DI = alpha_c * C * Z + alpha_m * M * Z - ganma_r * I - ganma_d * I - ganma_z * I
    DR = ganma_r * I - nd * R
    DD = nd * C + nd * M + nd * R + ganma_d * I - zeta * D
    DZ = zeta * D + ganma_z * I - omega * M * Z
    DK = omega * M * Z
    DF = delta_f * C - delta_u * (C + M)
    return [DC, DM, DI, DR, DD, DZ, DK, DF]


def integrate_sir(
    time,
    samples,
    X_init,
    alpha_c,
    alpha_m,
    beta_m_to_c,
    beta_c_to_m,
    nd,
    ganma_r,
    ganma_d,
    ganma_z,
    zeta,
    omega,
    delta_f,
    delta_u,
):
    t = linspace(0, time, samples)

    X, infodict = integrate.odeint(
        CMIRDZKF,
        X_init,
        t,
        (
            alpha_c,
            alpha_m,
            beta_m_to_c,
            beta_c_to_m,
            nd,
            ganma_r,
            ganma_d,
            ganma_z,
            zeta,
            omega,
            delta_f,
            delta_u,
        ),
        full_output=True,
    )

    X0, X1, X2, X3, X4, X5, X6, X7 = X.T

    return (t, X0, X1, X2, X3, X4, X5, X6, X7)


def try_CMIRDZKF():
    alpha_c = 0.4
    alpha_m = 0.2
    nd = 0.01
    ganma_r = 0.3
    ganma_d = 0.3
    ganma_z = 0.4
    zeta = 0.3
    omega = 0.25
    delta_f = 0.6
    delta_u = 0.5

    X_init = [500, 500, 0, 0, 0, 1, 0, 1000]

    time = 20
    n = 10000

    samples = 200

    t, X0, X1, X2, X3, X4, X5, X6, X7 = integrate_sir(
        time,
        n,
        X_init,
        alpha_c,
        alpha_m,
        beta_m_to_c,
        beta_c_to_m,
        nd,
        ganma_r,
        ganma_d,
        ganma_z,
        zeta,
        omega,
        delta_f,
        delta_u,
    )

    # plt.plot(t, X0, label="Collector population")
    # plt.plot(t, X1, label="Millitia population")
    # plt.plot(t, X2, label="Infected population")
    # plt.plot(t, X3, label="Recupered population")
    # plt.plot(t, X4, label="Dead population")
    # plt.plot(t, X5, label="Zombie population")
    # plt.plot(t, X6, label="Kill population")
    # plt.legend()
    # plt.show()

    # plt.plot(t, X7, label="Food amount")
    # plt.legend()
    # plt.show()

    ts = take_n_samples_regular(samples, t)
    X0s = take_n_samples_regular(samples, X0)
    X1s = take_n_samples_regular(samples, X1)
    X2s = take_n_samples_regular(samples, X2)
    X3s = take_n_samples_regular(samples, X3)
    X4s = take_n_samples_regular(samples, X4)
    X5s = take_n_samples_regular(samples, X5)
    X6s = take_n_samples_regular(samples, X6)
    X7s = take_n_samples_regular(samples, X7)

    X_samples = [
        [ts[i], X0s[i], X1s[i], X2s[i], X3s[i], X4s[i], X5s[i], X6s[i], X7s[i]]
        for i in range(len(ts))
    ]

    ode = [
        CMIRDZKF(
            [X0s[i], X1s[i], X2s[i], X3s[i], X4s[i], X5s[i], X6s[i], X7s[i]],
            ts[i],
            alpha_c,
            alpha_m,
            beta_m_to_c,
            beta_c_to_m,
            nd,
            ganma_r,
            ganma_d,
            ganma_z,
            zeta,
            omega,
            delta_f,
            delta_u,
        )
        for i in range(len(ts))
    ]

    best_system = symbolic_regression(
        X_samples,
        ode,
        seed_g=0,
        FEATURES_NAMES=["t", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],
        MAX_GENERATIONS=10000,
        N_GENERATION_OPTIMIZE=1,
        POP_SIZE=100,
        TOURNAMENT_SIZE=1,
        XOVER_PCT=0.5,
        MAX_DEPTH=10,
        REG_STRENGTH=100,
    )

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {
            "t": t,
            "X0": X[0],
            "X1": X[1],
            "X2": X[2],
            "X3": X[3],
            "X4": X[4],
            "X5": X[5],
            "X6": X[6],
            "X7": X[7],
        },
    )

    X_gp, infodict = integrate.odeint(integrate_gp, X_init, t, full_output=True)

    X0_gp, X1_gp, X2_gp, X3_gp, X4_gp, X5_gp, X6_gp, X7_gp = X_gp.T

    plt.plot(t, X0, label="Collector population")
    plt.plot(t, X1, label="Millitia population")
    plt.plot(t, X2, label="Infected population")
    plt.plot(t, X3, label="Recupered population")
    plt.plot(t, X4, label="Dead population")
    plt.plot(t, X5, label="Zombie population")
    plt.plot(t, X6, label="Kill population")

    plt.plot(t, X0_gp, label="Collector population symbolic regression")
    plt.plot(t, X1_gp, label="Millitia population symbolic regression")
    plt.plot(t, X2_gp, label="Infected population symbolic regression")
    plt.plot(t, X3_gp, label="Recupered population symbolic regression")
    plt.plot(t, X4_gp, label="Dead population symbolic regression")
    plt.plot(t, X5_gp, label="Zombie population symbolic regression")
    plt.plot(t, X6_gp, label="Kill population symbolic regression")

    plt.legend()
    plt.show()

    plt.plot(t, X7, label="Food amount")
    plt.plot(t, X7_gp, label="Food symbolic regression")
    plt.legend()
    plt.show()
