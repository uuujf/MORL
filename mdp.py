import random
import numpy as np

class MOMDP(object):
    def __init__(self, S, A, H, d):
        super(MOMDP, self).__init__()
        self.S = S
        self.A = A
        self.H = H
        self.d = d

        self.transit = None
        self.rewards = None

        self._init_transit("uniform")
        self._init_rewards("random")


    def _init_transit(self, dist=None):
        # (x,a,y)
        if dist == "uniform":
            self.transit = np.random.rand(self.S, self.A, self.S)
            self.transit /= np.sum(self.transit, axis=(2), keepdims=True)

    def _init_rewards(self, dist=None):
        # (d, H, S, A)
        if dist == "random":
            self.rewards = np.random.rand(self.d, self.H, self.S, self.A)

    def get_reward(self):
        # (H, S, A)
        pref = np.random.rand(self.d)
        pref /= np.sum(pref)
        return np.sum(self.rewards * np.reshape(pref, (self.d, 1, 1, 1)), axis=0)

    def play(self, reward, policy):
        # reward: (H, S, A)
        # policy: (H, S, A)
        trajectory = []
        x = 0
        V = 0
        for h in range(self.H):
            a = np.random.choice(self.A, p=policy[h, x, :])
            y = np.random.choice(self.S, p=self.transit[x,a,:])
            V += reward[h, x, a]
            trajectory.append( (x,a,y) )
            x = y
        return V, trajectory

    def _get_rand_policy(self):
        pi = np.random.rand(self.H, self.S, self.A)
        pi /= np.sum(pi, axis=1, keepdims=True)
        return pi

    def _get_V_pi(self, reward, policy):
        # reward: (H, S, A)
        # policy: (H, S, A)
        p_x = np.zeros(self.S)
        p_x[0] = 1
        V = 0
        for h in range(self.H):
            p_xa = policy[h] * p_x.reshape((self.S, 1)) # (S, A)
            p_y = np.sum(p_xa.reshape((self.S, self.A, 1)) * self.transit, axis=(0,1))
            V += np.sum(p_xa * reward[h])
            p_x = np.copy(p_y)
            # print(p_y.shape, p_y.sum())
        return V

    def _get_V_opt(self, reward):
        # reward: (H, S, A)
        V = np.zeros(self.S)
        pi = np.zeros((self.H, self.S, self.A))
        for h in np.arange(self.H-1, -1, -1):
            Q = reward[h] + np.sum(self.transit * V.reshape((1, 1, self.S)), axis=2)
            V = np.amax(Q, axis=1)
            pi[h] = np.eye(self.A)[np.argmax(Q, axis=1)]
        return V[0], pi

    def regret_opt_stationary_policy(self, Rs):
        r_bar = np.zeros((self.H, self.S, self.A))
        for R in Rs:
            r_bar += R
        r_bar /= len(Rs)

        _, pi = self._get_V_opt(r_bar)

        regret = 0
        regrets = [0]
        for R in Rs:
            V_opt, _ = self._get_V_opt(R)
            V_pi = self._get_V_pi(R, pi)
            regret += (V_opt - V_pi)
            regrets.append(regret)
        return regrets

    def regret(self, Rs, Pis):
        regret = 0
        regrets = [0]
        for (R, Pi) in zip(Rs, Pis):
            V_opt, _ = self._get_V_opt(R)
            V_pi = self._get_V_pi(R, Pi)
            regret += (V_opt - V_pi)
            regrets.append(regret)
        return regrets

class MOUCBVI(object):
    def __init__(self, momdp):
        super(MOUCBVI, self).__init__()
        self.momdp = momdp

        self.S = self.momdp.S
        self.A = self.momdp.A
        self.H = self.momdp.H
        self.d = self.momdp.d
        # self.rewards = self.momdp.rewards

        self.histogram = np.ones((self.S, self.A, self.S)) # (S, A, S) # initialize to 1 to avoid boundary cases

    def update_history(self, episode_trajectory):
        for (x,a,y) in episode_trajectory:
            self.histogram[x,a,y] += 1
    
    def get_empi_transit(self):
        # (S, A, S)
        empi_transit = np.copy(self.histogram)
        empi_transit /= np.sum(empi_transit, axis=2, keepdims=True)
        return empi_transit
    
    def get_bonus(self, coeff=1.0):
        # (S, A)
        bonus = np.sqrt(  np.minimum(self.d, self.S) * self.H **2 / np.sum(self.histogram, axis=2))
        return bonus * coeff

    def get_Q_table(self, reward, coeff=1.0):
        # reward: (H, S, A)
        bonus = self.get_bonus(coeff) # (S, A)
        empi_transit = self.get_empi_transit() # (S, A, S)

        pi = np.zeros((self.H, self.S, self.A))
        Q = np.zeros((self.H, self.S, self.A))
        V = np.zeros(self.S)
        for h in np.arange(self.H-1, -1, -1):
            Q[h] = reward[h] + bonus + np.sum(empi_transit * V.reshape((1, 1, self.S)), axis=2)
            Q[h] = np.minimum(Q[h], H)
            V = np.amax(Q[h], axis=1)
            pi[h] = np.eye(self.A)[np.argmax(Q[h], axis=1)]
        return Q, pi

    def online_game(self, K, coeff=1.0):
        Rs = []
        Pis = []
        for _ in range(K):
            reward = self.momdp.get_reward()
            _, policy = self.get_Q_table(reward, coeff)
            _, trajectory = self.momdp.play(reward, policy)
            self.update_history(trajectory)
            Rs.append(np.copy(reward))
            Pis.append(np.copy(policy))
        return Rs, Pis


if __name__ == '__main__':
    S = 20
    A = 5
    H = 10
    d = 15
    K = 5000

    momdp = MOMDP(S, A, H, d)
    moucbvi = MOUCBVI(momdp)

    
    Rs, Pis = moucbvi.online_game(K, coeff=0.1)
    regrets = momdp.regret(Rs, Pis)
    regrets_stay = momdp.regret_opt_stationary_policy(Rs)

    print(regrets[-10:])
    print(regrets_stay[-10:])


    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(range(K+1), regrets, "-r")
    plt.plot(range(K+1), regrets_stay, "-b")

    plt.xlabel("# episodes")
    plt.ylabel("total regret")
    plt.legend(["MO-UCBVI", "Best stationary policy"], fontsize=15)

    plt.savefig("regret.pdf")