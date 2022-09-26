import random


def add_noise(target, max_noise):
    result = list(
        map(
            lambda y: map(lambda x: x + random.uniform(-1 * max_noise, max_noise), y)
            if type(y) == list
            else y + random.uniform(-1 * max_noise, max_noise),
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
