import numpy as np
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.utils import take_n_samples_regular
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def add_noise(target, max_noise, seed=None):
    rng = np.random.default_rng(seed)

    result = list(
        map(
            lambda y: (y + y * max_noise * rng.standard_normal(1)).item(),
            target,
        )
    )

    return result


def derivate(x, y):
    result = []
    for i in range(len(y)):
        if i == len(y) - 1:
            break

        result_i = []
        for j in range(len(y[i])):
            result_i.append((y[i + 1][j] - y[i][j]) / (x[i + 1] - x[i]))

        result.append(result_i)

    return result


def integrate_model(model, time, n, X0, *args):
    t = linspace(0, time, n)

    X, infodict = integrate.odeint(model, X0, t, args, full_output=True)

    variables = X.T

    return (t, *variables)


def get_data_from_samples(
    t_total,
    t_samples,
    X_samples,
    smoothing_factor,
    symbolic_regression_samples,
    variable_names,
):
    X_spline = [
        UnivariateSpline(t_samples, x, s=smoothing_factor[i])
        for i, x in enumerate(X_samples)
    ]

    t_simbolic_regression_samples = take_n_samples_regular(
        symbolic_regression_samples, t_total
    )

    X_samples = [
        {
            **{variable_names[0]: i},
            **{
                variable_names[j + 1]: X_spline[j](i).item()
                for j in range(len(variable_names[1:]))
            },
        }
        for i in t_simbolic_regression_samples
    ]

    X_spline_derivate = [x.derivative() for x in X_spline]

    ode = [
        [dx(i).item() for dx in X_spline_derivate]
        for i in t_simbolic_regression_samples
    ]

    return {
        "X_spline": X_spline,
        "X_samples": X_samples,
        "ode": ode,
    }


def add_noise_and_get_data(
    t,
    X,
    samples,
    symbolic_regression_samples,
    noise,
    smoothing_factor,
    variable_names,
    noise_seed=None,
):
    t_noise = take_n_samples_regular(samples, t)
    X_noise = [
        add_noise(take_n_samples_regular(samples, x), noise, noise_seed) for x in X
    ]

    ret = get_data_from_samples(
        t,
        t_noise,
        X_noise,
        smoothing_factor,
        symbolic_regression_samples,
        variable_names,
    )

    return {**ret, **{"t": t, "X": X, "t_noise": t_noise, "X_noise": X_noise}}


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


def separate_samples(variable_names, X_samples):
    ret = []

    for i in variable_names:
        actual = []
        for j in X_samples:
            actual.append(j[i])
        ret.append(actual)

    return ret
