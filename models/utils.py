import numpy as np
from sympy.plotting.textplot import linspace
from scipy import integrate
from src.utils import take_n_samples_regular
from scipy.interpolate import UnivariateSpline


def add_noise(target, max_noise):
    rng = np.random.default_rng()

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


def get_data_from_model(
    model,
    X0,
    time,
    n,
    samples,
    symbolic_regression_samples,
    noise,
    smoothing_factor,
    variable_names,
    *args
):
    t, *X = integrate_model(model, time, n, X0, *args)

    t_noise = take_n_samples_regular(samples, t)
    X_noise = [add_noise(take_n_samples_regular(samples, x), noise) for x in X]

    X_spline = [UnivariateSpline(t_noise, x, s=smoothing_factor) for x in X_noise]

    t_simbolic_regression_samples = take_n_samples_regular(
        symbolic_regression_samples, t
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
        "t": t,
        "X": X,
        "t_noise": t_noise,
        "X_noise": X_noise,
        "X_spline": X_spline,
        "X_samples": X_samples,
        "ode": ode,
    }
