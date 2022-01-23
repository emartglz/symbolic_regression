import operator
from random import randint, random, seed
from copy import deepcopy
from functools import reduce
from scipy.optimize import least_squares
import numpy as np
from pprint import pprint


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


def evaluate_least_squares(theta, node, row):
    if "children" not in node:
        if "feature_name" in node:
            return row[node["feature_name"]]
        return theta[int(node["value"])]
    return node["func"](
        *[evaluate_least_squares(theta, c, row) for c in node["children"]]
    )


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
            return {"value": None}
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
    VARIABLE_PROBABILITY,
    MAX_CONSTANT,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
):
    offspring = deepcopy(selected)
    mutate_point, depth = select_random_node(offspring, None, 0, MAX_DEPTH)

    child_count = len(mutate_point["children"])

    children = randint(0, child_count - 1)

    if "children" not in mutate_point["children"][children]:
        r = random()
        # add constant
        if r < CONSTANT_PROBABILITY:
            mutate_point["children"][children] = {"value": None}
        r -= CONSTANT_PROBABILITY
        if r < VARIABLE_PROBABILITY:
            mutate_point["children"][children] = {
                "feature_name": features_names[randint(0, len(features_names) - 1)]
            }
        else:
            r -= VARIABLE_PROBABILITY
            mutate_point["children"][children] = random_prog(
                depth + 1,
                system_lenght,
                operations,
                features_names,
                MAX_DEPTH,
                CONSTANT_PROBABILITY,
                MAX_CONSTANT,
            )
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

        r = random()

        # Change operation same "aridad"
        if r < CHANGE_OPERATION_PROBABILITY:
            mutate_point["children"][children]["func"] = op["func"]
            mutate_point["children"][children]["format_str"] = op["format_str"]
        r -= CHANGE_OPERATION_PROBABILITY
        # Delete node
        if r < DELETE_NODE_PROBABILITY:
            r2 = random()
            # add constant
            if r2 < CONSTANT_PROBABILITY:
                mutate_point["children"][children] = {"value": None}
            r2 -= CONSTANT_PROBABILITY
            if r2 < VARIABLE_PROBABILITY:
                mutate_point["children"][children] = {
                    "feature_name": features_names[randint(0, len(features_names) - 1)]
                }
            else:
                r2 -= VARIABLE_PROBABILITY
                mutate_point["children"][children] = random_prog(
                    depth + 1,
                    system_lenght,
                    operations,
                    features_names,
                    MAX_DEPTH,
                    CONSTANT_PROBABILITY,
                    MAX_CONSTANT,
                )
        else:
            r -= DELETE_NODE_PROBABILITY
            # Add operation
            r_operation = operations[randint(0, len(operations) - 1)]
            node = deepcopy(mutate_point["children"][children])

            mutate_point["children"][children]["children"] = [
                {"value": None} for _ in range(r_operation["arg_count"])
            ]
            mutate_point["children"][children]["children"][-1] = node
            mutate_point["children"][children]["func"] = r_operation["func"]
            mutate_point["children"][children]["format_str"] = r_operation["format_str"]

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
    VARIABLE_PROBABILITY,
    MAX_CONSTANT,
    CHANGE_OPERATION_PROBABILITY,
    DELETE_NODE_PROBABILITY,
    ADD_OPERATION_PROBABILITY,
    XOVER_PCT,
    TOURNAMENT_SIZE,
):
    parent1 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
    if random() < XOVER_PCT:
        parent2 = get_random_parent(population, fitness, TOURNAMENT_SIZE)
        return constant_name_assign(do_xover(parent1, parent2, MAX_DEPTH), 0)[0]
    else:
        return constant_name_assign(
            do_mutate(
                parent1,
                system_lenght,
                operations,
                features_names,
                MAX_DEPTH,
                CONSTANT_PROBABILITY,
                VARIABLE_PROBABILITY,
                MAX_CONSTANT,
                CHANGE_OPERATION_PROBABILITY,
                DELETE_NODE_PROBABILITY,
                ADD_OPERATION_PROBABILITY,
            ),
            0,
        )[0]


def node_count(x):
    if "children" not in x:
        return 1
    return sum([node_count(c) for c in x["children"]])


def constant_count(x):
    if "children" not in x:
        if "value" in x:
            return 1
        return 0
    return sum([constant_count(c) for c in x["children"]])


def constant_name_assign(selected, number=0):
    offspring = deepcopy(selected)
    if "children" not in offspring:
        if "value" in offspring:
            offspring["value"] = str(number)
            number += 1
        return offspring, number

    child_number = len(offspring["children"])
    for c in range(child_number):
        offspring["children"][c], number = constant_name_assign(
            offspring["children"][c], number
        )

    return offspring, number


def compute_fitness(program, prediction, target, REG_STRENGTH):
    mse = 0
    for i in range(len(prediction)):
        mse_2 = 0
        for j in range(len(prediction[i])):
            mse_2 += (prediction[i][j] - target[i][j]) ** 2
        mse += mse_2 / len(prediction[i])
    mse /= len(prediction)

    # penalty = node_count(program) ** REG_STRENGTH
    penalty = 1
    return mse * penalty


def fun(theta, prog, ts, ys, REG_STRENGTH):
    prediction = [evaluate_least_squares(theta, prog, t) for t in ts]
    return compute_fitness(prog, prediction, ys, REG_STRENGTH)


def symbolic_regression(
    X,
    target,
    MAX_GENERATIONS=10,
    seed_g=random(),
    MAX_DEPTH=10,
    POP_SIZE=300,
    CONSTANT_PROBABILITY=0.5,
    VARIABLE_PROBABILITY=0.3,
    MAX_CONSTANT=100,
    CHANGE_OPERATION_PROBABILITY=0.3,
    DELETE_NODE_PROBABILITY=0.3,
    ADD_OPERATION_PROBABILITY=0.4,
    TOURNAMENT_SIZE=3,
    XOVER_PCT=0.7,
    REG_STRENGTH=2,
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
        constant_name_assign(
            random_prog(
                0,
                system_lenght,
                operations,
                features_names,
                MAX_DEPTH,
                CONSTANT_PROBABILITY,
                MAX_CONSTANT,
            )
        )[0]
        for _ in range(POP_SIZE)
    ]

    global_best = float("inf")
    for gen in range(MAX_GENERATIONS):
        fitness = []
        for prog in population:
            constant = constant_count(prog)

            if constant != 0:
                constant_ini = [random() * MAX_CONSTANT for _ in range(constant)]
                prediction = least_squares(
                    fun,
                    constant_ini,
                    bounds=(0, MAX_CONSTANT),
                    kwargs={
                        "prog": prog,
                        "ts": X,
                        "ys": target,
                        "REG_STRENGTH": REG_STRENGTH,
                    },
                )

                score = prediction.cost

                # print(score)
                # print(constant_count(prog), prediction.x)
                # print(render_prog(prog))

                if score < global_best:
                    global_best = score
                    best_prog = prog
                    best_const = prediction.x

            else:
                prediction = [evaluate(prog, Xi) for Xi in X]
                score = compute_fitness(prog, prediction, target, REG_STRENGTH)

                # print(score)
                # print(render_prog(prog))

                if score < global_best:
                    global_best = score
                    best_prog = prog
                    best_const = []

            fitness.append(score)

        mean = reduce(lambda a, b: a + b, fitness) / len(fitness)

        print(
            f"Generation: {gen}\nBest Score: {global_best}\nMean score: {mean}\nBest program:\n{render_prog(best_prog)}\nBest Constant:\n{best_const}\n"
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
                    VARIABLE_PROBABILITY,
                    MAX_CONSTANT,
                    CHANGE_OPERATION_PROBABILITY,
                    DELETE_NODE_PROBABILITY,
                    ADD_OPERATION_PROBABILITY,
                    XOVER_PCT,
                    TOURNAMENT_SIZE,
                )
            )

        population = population_next_gen

    print(f"Best score: {global_best}")
    print(f"Best program:\n{render_prog(best_prog)}")
    print(f"Best Constant: {best_const}")

    return render_prog(best_prog)