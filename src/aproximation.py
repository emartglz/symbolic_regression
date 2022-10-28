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

    y_spline_derivate = [x.derivative(nu=1) for x in y_spline]

    result_X = [x]
    result_Dx = []
    for i in range(len(y)):
        result_i = []
        result_i_dx = []
        for j in x:
            result_i.append(y_spline[i](j).item())
            result_i_dx.append(y_spline_derivate[i](j).item())

        result_X.append(result_i)
        result_Dx.append(result_i_dx)

    return result_X, result_Dx
