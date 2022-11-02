import os
import sys
from models.utils import make_experiment

# S' = a - b* S * I/(S+I+R)
# I' = b* S * I/(S+I+R) - c * I - d * I
# R' = c* I
# D' = d * I
def sird_dx(X, t, a, b, c, d):
    S, I, R, D = X

    S_d = a - b * S * I / (S + I + R)
    I_d = b * S * I / (S + I + R) - c * I - d * I
    R_d = c * I
    D_d = d * I

    return [S_d, I_d, R_d, D_d]


def try_sird(noise, seed, name, save_to, samples=None):
    a = 250
    b = 0.5
    c = 0.1
    d = 0.2

    X0 = [7000, 3000, 0, 0]

    time = 20
    n = 10000

    samples = 300

    if noise == "original_model":
        smoothing_factor = None
    elif noise == 0:
        smoothing_factor = [1] * 4
    else:
        smoothing_factor = [0.1, 0.1, 0.1, 0.1]

    variable_names = ["t", "S", "I", "R", "D"]

    make_experiment(
        sird_dx,
        X0,
        variable_names,
        smoothing_factor,
        noise,
        seed,
        name,
        save_to,
        [a, b, c, d],
        {
            "MAX_GENERATIONS": 100,
            "POP_SIZE": 100,
            "FEATURES_NAMES": [["S", "I", "N"], ["S", "I", "N"], ["I"], ["I"]],
            "MUTATION_SIZE": 50,
            "XOVER_SIZE": 50,
            "MAX_DEPTH": 10,
            "REG_STRENGTH": 40,
            "RANDOM_SELECTION_SIZE": 10,
            # "verbose": True,
        },
        add_N=["S", "I", "R"],
        time=time,
        samples=samples,
        # show_spline=True,
    )


# def try_interval(a, b, noise, save_to):
#     for i in range(a, b):
#         j = i * 10
#         print(j)
#         try_sird(noise, 0, f"{j}", save_to, j)


if __name__ == "__main__":

    noise = "original_model"
    if len(sys.argv) == 2:
        noise = float(sys.argv[1])

    save_to = f"RESULTS/SIRD/noise_{noise}"
    # save_to = f"SAMPLES_TEST/SIRD/noise_{noise}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # repetir a partir del 60
    # import multiprocessing

    # # max_jobs = 16
    # b = 15
    # a = 1
    # ran = b - a
    # # step_size = int(ran / max_jobs) + 1
    # step_size = 1
    # i = a

    # jobs = []
    # while i <= b:
    #     jobs.append(
    #         multiprocessing.Process(
    #             target=try_interval,
    #             args=(i, i + step_size, noise, save_to),
    #         )
    #     )
    #     i += step_size
    #     jobs[-1].start()

    r = 30
    for i in range(r):
        print(i)
        try_sird(noise, i, f"SIRD_{i}", save_to)
