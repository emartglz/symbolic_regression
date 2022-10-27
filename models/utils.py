import numpy as np
from sympy.plotting.textplot import linspace
from scipy import integrate
import matplotlib.pyplot as plt
from src.symbolic_regression import symbolic_regression
from src.utils import (
    evaluate,
    group_with_names,
    save_results,
    save_samples,
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
    time=300,
    n=100000,
    samples=300,
):
    t, *X = integrate_model(model, time, n, X0, *params)

    t_samples, *X_samples = [take_n_samples_regular(samples, i) for i in [t, *X]]
    X_noise = [add_noise(x, noise, seed) for x in X_samples]

    save_samples(
        group_with_names([t_samples, *X_noise], variable_names),
        f"{save_to}/data_{name}",
    )

    plot_data(
        variables_names=variable_names[1:],
        t_samples=t_samples,
        samples=X_samples,
        t_noise=t_samples,
        samples_noise=X_noise,
        name=f"{save_to}/initial_plot_{name}.pdf",
    )

    results = symbolic_regression(
        [t_samples, *X_noise],
        variable_names,
        smoothing_factor,
        seed_g=seed,
        **genetic_params,
    )

    # results = get_results("models_jsons/LV")
    best_system = results["system"]
    save_results(results, f"{save_to}/LV_{name}")

    integrate_gp = lambda X, t: evaluate(
        best_system,
        {**{"t": t}, **{v_name: X[i] for i, v_name in enumerate(variable_names[1:])}},
    )
    X_gp, _ = integrate.odeint(integrate_gp, X0, t, full_output=True)
    X_gp = X_gp.T.tolist()

    plot_data(
        variables_names=variable_names[1:],
        t_samples=t_samples,
        samples=X_samples,
        # t_noise=data["t_noise"],
        # samples_noise=data["X_noise"],
        # t_spline=t_spline,
        # samples_spline=X_spline,
        t_symbolic_regression=t,
        samples_symbolic_regression=X_gp,
        name=f"{save_to}/final_plot_{name}.svg",
    )
