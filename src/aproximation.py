from csaps import csaps


def derivate(x, y):
    result = []
    for i in range(len(y)):
        result_i = []
        for j in range(len(y[i])):
            if j == len(y[i]) - 1:
                break
            result_i.append((y[i][j + 1] - y[i][j]) / (x[j + 1] - x[j]))

        result.append(result_i)

    return result


def smoothing_spline(x, y, smoothing_factor):
    y_spline = [
        csaps(x, y_i, smooth=smoothing_factor[i]).spline for i, y_i in enumerate(y)
    ]

    result = []
    for i in range(len(y)):
        result_i = []
        for j in x[:-1]:
            result_i.append(y_spline[i](j).item())

        result.appen(result_i)

    return result


def generate_dx(t, X, smoothing_factor):
    if smoothing_factor[0] == 1:
        dx = derivate(t, X)
    else:
        dx = smoothing_spline(t, X, smoothing_factor)

    return dx
