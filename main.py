import json
from src.utils import get_results


def analise_tests(model_name, noises, amount_of_tests):
    ret = {}
    print("MODEL NAME: ", model_name)
    for i in noises:
        print("NOISE: ", i)

        sum = 0
        score_min = 1e30
        score_max = 0

        sum_dif_gp_original = 0
        sum_dif_gp_noise = 0
        sum_dif_gp_spline = 0
        sum_dif_gp_olivia = 0

        evaluated_systems = 0

        min_system = 0
        min_system_value = 1e30
        for j in range(amount_of_tests):
            results = get_results(f"RESULTS/{model_name}/noise_{i}/{model_name}_{j}")

            sum += results["score"]
            score_min = min(score_min, results["score"])
            score_max = max(score_max, results["score"])

            with open(
                f"RESULTS/{model_name}/noise_{i}/results_{model_name}_{j}.json"
            ) as json_file:
                data = json.load(json_file)

            if data["no_count"] == True:
                continue

            if data["dif_gp_original"] < min_system_value:
                min_system = j
                min_system_value = data["dif_gp_original"]

            sum_dif_gp_original += data["dif_gp_original"]
            sum_dif_gp_noise += data["dif_gp_noise"]
            sum_dif_gp_spline += data["dif_gp_spline"]
            sum_dif_gp_olivia += data["dif_gp_olivia"]
            evaluated_systems += 1

        mean = sum / amount_of_tests

        mean_dif_gp_original = sum_dif_gp_original / evaluated_systems
        mean_dif_gp_noise = sum_dif_gp_noise / evaluated_systems
        mean_dif_gp_spline = sum_dif_gp_spline / evaluated_systems
        mean_dif_gp_olivia = sum_dif_gp_olivia / evaluated_systems

        ret[i] = {
            "mean": round(mean, 5),
            "min": round(score_min, 5),
            "max": round(score_max, 5),
            "dif_gp_original": round(mean_dif_gp_original, 5),
            "dif_gp_noise": round(mean_dif_gp_noise, 5),
            "dif_gp_spline": round(mean_dif_gp_spline, 5),
            "dif_gp_olivia": round(mean_dif_gp_olivia, 5),
            "evaluated_systems": evaluated_systems,
        }

        print("SCORE MIN: ", score_min)
        print("SCORE MAX: ", score_max)
        print("SCORE MEAN: ", mean)
        print("MIN SYSTEM: ", min_system_value, " ", min_system)

    return ret


def print_latex_table(results, noises):
    text = """\\begin{tabular}{|c|c|c|c|c|c|}
\hline  
"""

    for noise, percent in noises:
        text += "& \\textbf{ruido de " + str(percent) + "\%}"

    text += """\\\\
\hline
"""

    for eng, esp in [
        ("evaluated_systems", "cantidad de sistemas"),
        ("dif_gp_original", "original"),
        ("dif_gp_noise", "original con ruido"),
        ("dif_gp_spline", "spline"),
        ("dif_gp_olivia", "otro método"),
    ]:
        text += esp

        for noise, percent in noises:
            text += " & " + str(results[noise][eng])

        text += """\\\\
\hline
"""
    text += "\\end{tabular}"

    return text


def print_another_latex_table(results, noises):
    text = """\\begin{tabular}{|c|c|c|c|}
\hline  
"""

    for noise, percent in noises:
        text += "& \\textbf{ruido de " + str(percent) + "\%}"

    text += """\\\\
\hline
"""

    for eng, esp in [("mean", "media"), ("min", "mínimo"), ("max", "máximo")]:
        text += esp

        for noise, percent in noises:
            text += " & " + str(results[noise][eng])

        text += """\\\\
\hline
"""
    text += "\\end{tabular}"

    return text


def print_systems(model_name, noises, amount_of_tests):
    print("MODEL NAME: ", model_name)
    for i in noises:
        print("NOISE: ", i)
        for j in range(amount_of_tests):
            results = get_results(f"RESULTS/{model_name}/noise_{i}/{model_name}_{j}")
            print(j)
            print(results["system_representation"])


if __name__ == "__main__":
    model_name = "SVVEIR"
    noises = [0.0, 0.05, 0.1]
    amount_of_tests = 30
    # noises = [0.1]

    # print_systems(model_name, noises, amount_of_tests)

    results = analise_tests(model_name, noises, amount_of_tests)
    # print(print_another_latex_table(results, [(0.0, 0), (0.05, 5), (0.1, 10)]))
    # print()
    # print(print_latex_table(results, [(0.0, 0), (0.05, 5), (0.1, 10)]))
