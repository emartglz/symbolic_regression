from src.utils import get_results


def analise_tests(model_name, noises, amount_of_tests):
    ret = {}
    print("MODEL NAME: ", model_name)
    for i in noises:
        print("NOISE: ", i)

        sum = 0
        score_min = 1e30
        score_max = 0

        for j in range(amount_of_tests):
            results = get_results(f"RESULTS/{model_name}/noise_{i}/{model_name}_{j}")

            sum += results["score"]
            score_min = min(score_min, results["score"])
            score_max = max(score_max, results["score"])

        mean = sum / amount_of_tests

        ret[i] = {
            "mean": round(mean, 5),
            "min": round(score_min, 5),
            "max": round(score_max, 5),
        }

        print("SCORE MIN: ", score_min)
        print("SCORE MAX: ", score_max)
        print("SCORE MEAN: ", mean)

    return ret


def print_latex_table(results, noises):
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


if __name__ == "__main__":
    model_name = "SVVEIR"
    noises = [0.0, 0.05, 0.1]
    amount_of_tests = 30

    results = analise_tests(model_name, noises, amount_of_tests)
    print(print_latex_table(results, [(0.0, 0), (0.05, 5), (0.1, 10)]))
