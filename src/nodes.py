def system(*list):
    return list


def system_str(*list):
    s = ""
    for i, exp in enumerate(list):
        s += f"{i + 1} : {exp} \n"
    return s


def population_edo_ecuation(*list):
    sum = 0
    for i in list:
        sum += i
    return sum


def population_edo_ecuation_str(*list):
    s = ""
    for exp in list:
        if s != "":
            s += f" + {exp}"
        else:
            s = f"{exp}"
    return s


def population_edo_term(constant, exp):
    return constant * exp


def population_edo_term_str(constant, exp):
    return f"{constant} * {exp}"
