import csv
import json
import numpy as np
from sympy.plotting.textplot import linspace
from scipy import integrate
import matplotlib.pyplot as plt
from src.symbolic_regression import symbolic_regression
from src.utils import (
    evaluate,
    get_results,
    group_with_names,
    load_samples,
    save_results,
    save_samples,
    separate_samples,
    take_n_samples_regular,
)


def integrate_model(model, time, n, X0, *args):
    t = linspace(0, time, n)

    X, _ = integrate.odeint(model, X0, t, args, full_output=True)

    variables = X.T

    return (t, *variables)


def add_noise(target, max_noise, seed=None):
    rng = np.random.default_rng(seed)

    result = [(y + y * max_noise * rng.standard_normal(1)).item() for y in target]

    return result


def plot_data(
    variables_names,
    t_samples=None,
    samples=None,
    t_noise=None,
    samples_noise=None,
    t_spline=None,
    samples_spline=None,
    t_symbolic_regression=None,
    samples_symbolic_regression=None,
    name=None,
):
    plt.clf()
    figure = plt.gcf()
    figure.set_size_inches(15, 8)

    if samples:
        for i, variable_name in enumerate(variables_names):
            plt.plot(t_samples, samples[i], label=f"{variable_name} samples")

    if samples_noise:
        for i, variable_name in enumerate(variables_names):
            plt.plot(
                t_noise, samples_noise[i], ".", label=f"{variable_name} samples noise"
            )
    if samples_spline:
        for i, variable_name in enumerate(variables_names):
            plt.plot(
                t_spline,
                samples_spline[i],
                "--",
                label=f"{variable_name} samples spline",
            )

    if samples_symbolic_regression:
        for i, variable_name in enumerate(variables_names):
            plt.plot(
                t_symbolic_regression,
                samples_symbolic_regression[i],
                "-.",
                label=f"{variable_name} symbolic regression",
            )

    plt.legend()
    if name:
        plt.savefig(
            name,
        )
        return
    plt.show()


def load_oli_params(model, noise, seed):
    file_name = f"OLIVIA_RESULTS/{model}/pso_results_{noise}"

    with open(f"{file_name}.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            if i == seed:
                return [float(x) for x in row[1:]]


def generate_experiment_results(
    model,
    X0,
    variable_names,
    noise,
    seed,
    name,
    save_to,
    params,
    add_N,
    time,
    n,
    samples,
):
    t, *X = integrate_model(model, time, n, X0, *params)

    results = get_results(f"{save_to}/{name}")
    best_system = results["system"]

    def evaluate_symbolic_regression(X, t):
        d = {
            **{"t": t},
            **{v_name: X[i] for i, v_name in enumerate(variable_names[1:])},
        }
        if add_N:
            sum = 0
            for j in add_N:
                sum += d[j]
            d["N"] = sum
        return evaluate(best_system, d)

    X_gp, _ = integrate.odeint(evaluate_symbolic_regression, X0, t, full_output=True)
    X_gp = X_gp.T.tolist()

    t_samples, *X_samples = [take_n_samples_regular(samples, i) for i in [t, *X]]

    t_noise, *X_noise = separate_samples(
        variable_names, load_samples(f"{save_to}/data_{name}")
    )

    t_spline, *X_spline = separate_samples(variable_names, results["X"])

    if isinstance(noise, float):
        model_name = name.split("_")[0]
        X_olivia = [[] for i in range(len(X_noise))]
        X_dx_gp = [[] for i in range(len(X_noise))]
        for i in range(len(t_noise)):
            X_to_eval = [x[i] for x in X_noise]
            temp = model(
                X_to_eval, t_noise[i], *load_oli_params(model_name, noise, seed)
            )
            temp2 = evaluate_symbolic_regression(X_to_eval, t_noise[i])
            for j in range(len(temp)):
                X_olivia[j].append(temp[j])
                X_dx_gp[j].append(temp2[j])

    X_gp_samples = [take_n_samples_regular(samples, i) for i in X_gp]

    # con respecto a los datos de la integración del modelo original
    dif_gp_original = 0
    # con respecto a los datos con ruido
    dif_gp_noise = 0
    # con respecto a los datos del spline
    dif_gp_spline = 0
    # con respecto a los datos de olivia
    dif_gp_olivia = 0
    no_count = False
    for i in range(len(X_spline)):
        for j in range(len(X_spline[i])):
            max_value = 10000
            assert t_samples[j] == t_noise[j] == t_spline[j]
            if (
                abs(X_gp_samples[i][j] - X_samples[i][j]) > max_value
                or abs(X_gp_samples[i][j] - X_noise[i][j]) > max_value
                or abs(X_gp_samples[i][j] - X_spline[i][j]) > max_value
            ):
                no_count = True
                break
            dif_gp_original += abs(X_gp_samples[i][j] - X_samples[i][j])
            dif_gp_noise += abs(X_gp_samples[i][j] - X_noise[i][j])
            dif_gp_spline += abs(X_gp_samples[i][j] - X_spline[i][j])
            if isinstance(noise, float):
                dif_gp_olivia += abs(X_dx_gp[i][j] - X_olivia[i][j])
        if no_count:
            break

    dif_gp_original /= len(X_spline) * len(X_spline[0])
    dif_gp_noise /= len(X_spline) * len(X_spline[0])
    dif_gp_spline /= len(X_spline) * len(X_spline[0])
    if isinstance(noise, float):
        dif_gp_olivia /= len(X_spline) * len(X_spline[0])

    dict_to_save = {
        "dif_gp_original": dif_gp_original,
        "dif_gp_noise": dif_gp_noise,
        "dif_gp_spline": dif_gp_spline,
        "no_count": no_count,
    }

    if isinstance(noise, float):
        dict_to_save["dif_gp_olivia"] = dif_gp_olivia

    file_name = f"{save_to}/results_{name}"
    with open(f"{file_name}.json", "w") as fp:
        json.dump(
            dict_to_save,
            fp,
        )


def make_experiment(
    model,
    X0,
    variable_names,
    smoothing_factor,
    noise,
    seed,
    name,
    save_to,
    params,
    genetic_params,
    add_N=False,
    time=300,
    n=100000,
    samples=300,
    show_spline=False,
):
    t, *X = integrate_model(model, time, n, X0, *params)

    t_samples, *X_samples = [take_n_samples_regular(samples, i) for i in [t, *X]]
    if isinstance(noise, float):
        X_noise = [add_noise(x, noise, seed) for x in X_samples]
    else:
        X_noise = X_samples

    save_samples(
        group_with_names([t_samples, *X_noise], variable_names),
        f"{save_to}/data_{name}",
    )

    if show_spline:
        for i, variable_name in enumerate(variable_names[1:]):
            plt.plot(t_samples, X_samples[i], label=f"{variable_name} samples")

        for i, variable_name in enumerate(variable_names[1:]):
            plt.plot(t_samples, X_noise[i], ".", label=f"{variable_name} samples noise")

    original_model = None
    if not isinstance(noise, float):
        original_model = (model, params)

    results = symbolic_regression(
        [t_samples, *X_noise],
        variable_names,
        smoothing_factor,
        seed_g=seed,
        add_N=add_N,
        show_spline=show_spline,
        original_model=original_model,
        **genetic_params,
    )

    # results = get_results(f"{save_to}/{name}")
    best_system = results["system"]
    save_results(results, f"{save_to}/{name}")

    def evaluate_symbolic_regression(X, t):
        d = {
            **{"t": t},
            **{v_name: X[i] for i, v_name in enumerate(variable_names[1:])},
        }
        if add_N:
            d["N"] = sum(X)
        return evaluate(best_system, d)

    X_gp, _ = integrate.odeint(evaluate_symbolic_regression, X0, t, full_output=True)
    X_gp = X_gp.T.tolist()

    t_spline, *X_spline = separate_samples(variable_names, results["X"])

    plot_data(
        variables_names=variable_names[1:],
        t_samples=t_samples,
        samples=X_samples,
        t_noise=t_samples,
        samples_noise=X_noise,
        t_spline=t_spline,
        samples_spline=X_spline,
        name=f"{save_to}/initial_plot_{name}.pdf",
    )

    plot_data(
        variables_names=variable_names[1:],
        t_samples=t_samples,
        samples=X_samples,
        t_symbolic_regression=t,
        samples_symbolic_regression=X_gp,
        name=f"{save_to}/final_plot_{name}.pdf",
    )

    generate_experiment_results(
        model,
        X0,
        variable_names,
        noise,
        seed,
        name,
        save_to,
        params,
        add_N,
        time,
        n,
        samples,
    )
