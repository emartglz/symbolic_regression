from models.SIZR import try_zombie_SIZR
from models.any_function import try_any_function
from models.lotka_volterra import try_lotka_volterra
from models.first_sir import try_sir
from models.zombie_militia_workers_moles import try_zombie
from models.CMIRDZKF import try_CMIRDZKF


def main():
    # try_lotka_volterra()
    # try_sir()
    # try_zombie()
    # try_zombie_SIZR()
    # try_any_function()
    try_CMIRDZKF()


if __name__ == "__main__":
    main()
