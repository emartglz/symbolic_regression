from models.lotka_volterra import try_lotka_volterra
from models.SIR import try_sir
from models.SIRD import try_sird
from models.SIQRD import try_siqrd
from models.SVVEIR import try_svveir

noises = [0.0, 0.05, 0.1, "original_model"]
models_func = [try_lotka_volterra, try_sir, try_sird, try_siqrd, try_svveir]
models_names = ["LV", "SIR", "SIRD", "SIQRD", "SVVEIR"]

experiments = 30

for m, model in enumerate(models_names):
    print(models_names[m])
    for n in noises:
        save_to = f"RESULTS_WITH_NO_FEATURES/{models_names[m]}/noise_{n}"
        print(n)
        for i in range(experiments):
            print(i)
            models_func[m](n, i, f"{models_names[m]}_{i}", save_to)
