import operator
from random import randint, random, seed
from copy import deepcopy
from functools import reduce


def render_prog(node):
    if "children" not in node:
        if "feature_name" in node:
            return node["feature_name"]
        return node["value"]
    return node["format_str"].format(*[render_prog(c) for c in node["children"]])


def evaluate(node, row):
    if "children" not in node:
        if "feature_name" in node:
            return row[node["feature_name"]]
        return node["value"]
    return node["func"](*[evaluate(c, row) for c in node["children"]])


def safe_div(a, b):
    return a / b if b else a


def system(*list):
    return list


def random_prog(
    depth,
    system_lenght,
    operations,
    features_names,
    MAX_DEPTH,
    CONSTANT_PROBABILITY,
    MAX_CONSTANT,
):
    if depth == 0:
        s = ""
        for i in range(system_lenght):
            s += f"{i + 1}" + ": {} \n"
        return {
            "func": system,
            "children": [
                random_prog(
                    depth + 1,
                    system_lenght,
                    operations,
                    features_names,
                    MAX_DEPTH,
                    CONSTANT_PROBABILITY,
                    MAX_CONSTANT,
                )
                for _ in range(system_lenght)
            ],
            "format_str": s,
        }

    # favor adding function nodes near the tree root and
    # leaf nodes as depth increases
    if depth < MAX_DEPTH and randint(0, MAX_DEPTH) >= depth:
        op = operations[randint(0, len(operations) - 1)]
        return {
            "func": op["func"],
            "children": [
                random_prog(
                    depth + 1,
                    system_lenght,
                    operations,
                    features_names,
                    MAX_DEPTH,
                    CONSTANT_PROBABILITY,
                    MAX_CONSTANT,
                )
                for _ in range(op["arg_count"])
            ],
            "format_str": op["format_str"],
        }
    else:
        if random() < CONSTANT_PROBABILITY:
            return {"value": round(random(), 3) * MAX_CONSTANT}
        return {"feature_name": features_names[randint(0, len(features_names) - 1)]}


def select_random_node(selected, parent, depth, MAX_DEPTH):
    if "children" not in selected:
        return (parent, depth - 1)
    # favor nodes near the root
    if randint(0, MAX_DEPTH) > depth:
        return (selected, depth)
    child_count = len(selected["children"])
    return select_random_node(
        selected["children"][randint(0, child_count - 1)],
        selected,
        depth + 1,
        MAX_DEPTH,
    )


def do_mutate(
    selected,
    system_lenght,
    operations,
    features_names,
    MAX_DEPTH,
    CONSTANT_PROBABILITY,
    MAX_CONSTANT,
):
    offspring = deepcopy(selected)
    mutate_point, depth = select_random_node(offspring, None, 0, MAX_DEPTH)

    child_count = len(mutate_point["children"])

    children = randint(0, child_count - 1)

    if "children" not in mutate_point["children"][children]:
        if random() < CONSTANT_PROBABILITY:
            mutate_point["children"][children] = {
                "value": round(random(), 3) * MAX_CONSTANT
            }
        mutate_point["children"][children] = {
            "feature_name": features_names[randint(0, len(features_names) - 1)]
        }
    else:
        possibles_func = [
            i
            for i in filter(
                lambda x: x["arg_count"]
                == len(mutate_point["children"][children]["children"]),
                operations,
            )
        ]
        op = possibles_func[randint(0, len(possibles_func) - 1)]

        mutate_point["children"][children]["func"] = op["func"]
        mutate_point["children"][children]["format_str"] = op["format_str"]

    return offspring


# TODO ahora mismo el xover puede darme un arbol con mayor profundidad máxima que MAX_DEPTH
def do_xover(selected1, selected2, MAX_DEPTH):
    offspring = deepcopy(selected1)
    xover_point1, _ = select_random_node(offspring, None, 0, MAX_DEPTH)
    xover_point2, _ = select_random_node(selected2, None, 0, MAX_DEPTH)

    child_count1 = len(xover_point1["children"])
    child_count2 = len(xover_point2["children"])
    xover_point1["children"][randint(0, child_count1 - 1)] = xover_point2["children"][
        randint(0, child_count2 - 1)
    ]
    return offspring


def get_random_parent(population, fitness, TOURNAMENT_SIZE):
    # randomly select population members for the tournament
    tournament_members = [
        randint(0, len(population) - 1) for _ in range(TOURNAMENT_SIZE)
    ]
    # select tournament member with best fitness
    member_fitness = [(fitness[i], population[i]) for i in tournament_members]
    return min(member_fitness, key=lambda x: x[0])[1]


def get_offspring(
    population,
    fitness,
    system_lenght,
    operations,
    features_names,
    MAX_DEPTH,
    CONSTANT_PROBABILITY,
    MAX_CONSTANT,
    XOVER_PCT,
    TOURNAMENT_SIZE,
):
    parent1 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
    if random() < XOVER_PCT:
        parent2 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
        return do_xover(parent1, parent2, MAX_DEPTH)
    else:
        return do_mutate(
            parent1,
            system_lenght,
            operations,
            features_names,
            MAX_DEPTH,
            CONSTANT_PROBABILITY,
            MAX_CONSTANT,
        )


def node_count(x):
    if "children" not in x:
        return 1
    return sum([node_count(c) for c in x["children"]])


def compute_fitness(program, prediction, target, REG_STRENGTH):
    mse = 0
    for i in range(len(prediction)):
        mse_2 = 0
        for j in range(len(prediction[i])):
            mse_2 += (prediction[i][j] - target[i][j]) ** 2
        mse += mse_2 / len(prediction[i])
    mse /= len(prediction)

    penalty = node_count(program) ** REG_STRENGTH
    return mse * penalty


def symbolic_regression(
    X,
    target,
    MAX_GENERATIONS=10,
    seed_g=random(),
    MAX_DEPTH=10,
    POP_SIZE=300,
    CONSTANT_PROBABILITY=0.5,
    MAX_CONSTANT=100,
    TOURNAMENT_SIZE=3,
    XOVER_PCT=0.7,
    REG_STRENGTH=0.5,
    EPSILON=0.001,
    PROPORTION_OF_BESTS=1 / 3,
):
    seed(seed_g)
    feature_lenght = len(X[0])
    system_lenght = len(target[0])

    features_names = []
    for i in range(feature_lenght):
        features_names.append(f"X{i + 1}")

    X2 = []
    for x in X:
        temp = {}
        for i, k in enumerate(features_names):
            temp[k] = x[i]
        X2.append(temp)
    X = X2

    operations = (
        {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
        {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
        {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
        {"func": safe_div, "arg_count": 2, "format_str": "({} / {})"},
        {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
    )

    population = [
        random_prog(
            0,
            system_lenght,
            operations,
            features_names,
            MAX_DEPTH,
            CONSTANT_PROBABILITY,
            MAX_CONSTANT,
        )
        for _ in range(POP_SIZE)
    ]

    global_best = float("inf")
    for gen in range(MAX_GENERATIONS):
        fitness = []
        for prog in population:
            prediction = [evaluate(prog, Xi) for Xi in X]
            score = compute_fitness(prog, prediction, target, REG_STRENGTH)
            fitness.append(score)
            if score < global_best:
                global_best = score
                best_prog = prog

        mean = reduce(lambda a, b: a + b, fitness) / len(fitness)

        print(
            f"Generation: {gen}\nBest Score: {global_best}\nMean score: {mean}\nBest program:\n{render_prog(best_prog)}\n"
        )

        if global_best < EPSILON:
            break

        member_fitness = [(fitness[i], i, population[i]) for i in range(POP_SIZE)]
        member_fitness.sort()

        best_amount = int(POP_SIZE * PROPORTION_OF_BESTS)
        population_next_gen = [i[2] for i in member_fitness[:best_amount]]
        for i in range(best_amount, POP_SIZE):
            population_next_gen.append(
                get_offspring(
                    population,
                    fitness,
                    system_lenght,
                    operations,
                    features_names,
                    MAX_DEPTH,
                    CONSTANT_PROBABILITY,
                    MAX_CONSTANT,
                    XOVER_PCT,
                    TOURNAMENT_SIZE,
                )
            )

        population = population_next_gen

    print(f"Best score: {global_best}")
    print(f"Best program:\n{render_prog(best_prog)}")

    return render_prog(best_prog)
