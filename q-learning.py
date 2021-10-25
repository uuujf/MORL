import random
from matplotlib.pyplot import step
import numpy as np
from mdp import MOMDP
from mdp import MOUCBVI

class MOQLearning(object):
    def __init__(self, momdp):
        super(MOQLearning, self).__init__()
        self.momdp = momdp

        self.S = self.momdp.S
        self.A = self.momdp.A
        self.H = self.momdp.H
        self.d = self.momdp.d
        # self.rewards = self.momdp.rewards

        self.histogram = np.ones((self.H, self.S, self.A, self.S)) # (H, S, A, S) # initialize to 1 to avoid boundary cases
        self.Q_table = np.ones((self.H + 1, self.S, self.A)) * H # (H + 1, S, A) # initialized to H

    def update_history(self, episode_trajectory):
        for h in range(self.H):
            x,a,y = episode_trajectory[h]
        # for (x,a,y) in episode_trajectory:
            self.histogram[h,x,a,y] += 1
    
    # def get_empi_transit(self):
    #     # (S, A, S)
    #     empi_transit = np.copy(self.histogram)
    #     empi_transit /= np.sum(empi_transit, axis=2, keepdims=True)
    #     return empi_transit
    
    def get_bonus(self, coeff=1.0):
        # (H, S, A)
        # bonus = np.sqrt(  np.minimum(self.d, self.S) * self.H **2 / np.sum(self.histogram, axis=2))
        bonus = np.sqrt(  self.H **3 / np.sum(self.histogram, axis=3))
        return bonus * coeff

    # def get_Q_table(self, reward, coeff=1.0):
    #     # reward: (H, S, A)
    #     bonus = self.get_bonus(coeff) # (S, A)
    #     empi_transit = self.get_empi_transit() # (S, A, S)

    #     pi = np.zeros((self.H, self.S, self.A))
    #     Q = np.zeros((self.H, self.S, self.A))
    #     V = np.zeros(self.S)
    #     for h in np.arange(self.H-1, -1, -1):
    #         Q[h] = reward[h] + bonus + np.sum(empi_transit * V.reshape((1, 1, self.S)), axis=2)
    #         Q[h] = np.minimum(Q[h], self.H)
    #         V = np.amax(Q[h], axis=1)
    #         pi[h] = np.eye(self.A)[np.argmax(Q[h], axis=1)]
    #     return Q, pi
    
    def get_policy(self):
        pi = np.zeros((self.H, self.S, self.A))
        for h in np.arange(self.H-1, -1, -1):
            pi[h] = np.eye(self.A)[np.argmax(self.Q_table[h], axis=1)]
        return pi

    def update_Q_table(self, reward, trajectory, coeff=1.0):
        bonus = self.get_bonus(coeff) # (H, S, A)
        for h in range(self.H):
            x,a,y = trajectory[h]
            Vx = np.amax(self.Q_table[h+1, y]) + 0.0
            stepsize = (self.H + 1) / (self.H + np.sum(self.histogram[h,x,a]))
            self.Q_table[h,x,a] = (1-stepsize) * self.Q_table[h,x,a] + stepsize * (Vx + reward[h,x,a] + bonus[h,x,a])
            self.Q_table[h,x,a] = np.minimum(self.Q_table[h,x,a], self.H)

    def online_game(self, rewards, coeff=1.0):
        # Rs = []
        Pis = []
        # stepsize = 0.9
        for reward in rewards:
            # reward = self.momdp.get_reward()
            policy = self.get_policy()
            _, trajectory = self.momdp.play(reward, policy)
            self.update_Q_table(reward, trajectory, coeff)
            self.update_history(trajectory)
            # Rs.append(np.copy(reward))
            Pis.append(np.copy(policy))
        return Pis


if __name__ == '__main__':
    S = 20
    A = 5
    H = 10
    d = 15
    K = 5000

    momdp = MOMDP(S, A, H, d)
    moucbvi = MOUCBVI(momdp)
    moq = MOQLearning(momdp)

    
    Rs, Pis = moucbvi.online_game(K, coeff=0.1)
    Pis_Q = moq.online_game(Rs, coeff=1.0)
    regrets = momdp.regret(Rs, Pis)
    # regrets_stay = momdp.regret_opt_stationary_policy(Rs)
    regrets_Q = momdp.regret(Rs, Pis_Q)

    print(regrets[-10:])
    print(regrets_Q[-10:])


    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(range(K+1), regrets, "-r")
    plt.plot(range(K+1), regrets_Q, "--g")

    plt.xlabel("number of episodes", fontsize=15)
    plt.ylabel("total regret", fontsize=15)
    plt.legend(["MO-UCBVI", "Q-Learning"], fontsize=15)

    plt.savefig("q-learning.pdf")
