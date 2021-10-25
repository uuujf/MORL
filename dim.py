import random
import numpy as np
from mdp import MOMDP
from mdp import MOUCBVI

if __name__ == '__main__':
    S = 20
    A = 5
    H = 10
    # d = 15
    K = 5000

    # ds = [50, 30, 15, 5]
    ds = [1, 5, 15, 20, 30]
    regrets = {}

    for d in ds:
        momdp = MOMDP(S, A, H, d)
        moucbvi = MOUCBVI(momdp)

        Rs, Pis = moucbvi.online_game(K, coeff=0.1)
        regrets[d] = momdp.regret(Rs, Pis)
        # regrets_stay = momdp.regret_opt_stationary_policy(Rs)

        print(regrets[d][-10:])
    # print(regrets_stay[-10:])


    import numpy as np
    import matplotlib.pyplot as plt

    for d in ds:
        plt.plot(range(K+1), regrets[d], "-", label="d="+str(d))
        # plt.plot(range(K+1), regrets_stay, "--b")

    plt.xlabel("number of episodes", fontsize=15)
    plt.ylabel("total regret", fontsize=15)
    plt.legend(fontsize=15)

    plt.savefig("dim.pdf")
