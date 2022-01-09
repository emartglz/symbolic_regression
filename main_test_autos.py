import pandas as pd
from symbolic_regression import symbolic_regression
from random import seed


def main():
    names = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
        "origin",
        "car name",
    ]

    # Read in data
    data = pd.read_csv(
        "auto-mpg.data",
        delim_whitespace=True,
        na_values="?",
        header=None,
        names=names,
    )
    # Drop string feature
    data.pop("car name")
    # Replace N/A values in horsepower
    data["horsepower"].fillna(data["horsepower"].median(), inplace=True)
    # Separate target feature
    target = data.pop("mpg")

    names = names[1:-1]
    X = []

    for _, row in data.iterrows():
        r = []
        for n in names:
            r.append(row[n])
        X.append(r)

    target = [[i] for i in target]

    print(
        symbolic_regression(
            X,
            target,
            CONSTANT_PROBABILITY=0,
            MAX_GENERATIONS=50,
            TOURNAMENT_SIZE=10,
            XOVER_PCT=0.5,
        )
    )


if __name__ == "__main__":
    main()
