from src.utils import get_results


def analise_tests(model_name, noises, amount_of_tests):
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

        print("SCORE MIN: ", score_min)
        print("SCORE MAX: ", score_max)
        print("SCORE MEAN: ", mean)


if __name__ == "__main__":
    model_name = "SIR"
    noises = [0, 0.05, 0.1]
    amount_of_tests = 30

    analise_tests(model_name, noises, amount_of_tests)
