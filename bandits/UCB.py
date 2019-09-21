import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal, entropy
from utils.helper import Game
import os

ex = Experiment("UCB-regret-minimisation")
ex.add_config("configs/base-config.yaml")


def UCB(args, game_i, alpha=0.2, pbar=None, verbose=False):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    if verbose:
        print(f"\nRunning game {game_i} with UCB-alpha = {alpha}")
        print(f"Arms distribution used {game_ll}")

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    for exp in range(args.repeat):

        # ---------------------------------------------------------------------------- #
        # 0. Rewards collected and optimal arm pull (for experiment)
        # ---------------------------------------------------------------------------- #

        regret_exp_ll = np.zeros(args.n)
        pulls_ll = np.zeros(len(game))
        sample_mean_ll = np.zeros(len(game))

        # ---------------------------------------------------------------------------- #
        # 1. UCB Initialisation
        # ---------------------------------------------------------------------------- #
        for j in range(len(game)):
            reward = game.get_reward(j)
            pulls_ll[j] += 1
            sample_mean_ll[j] += (reward - sample_mean_ll[j]) / pulls_ll[j]
            regret_exp_ll[j] = best_reward - reward
            arm_ll[j, j] += 1

        for j in range(len(game), args.n):
            # ---------------------------------------------------------------------------- #
            # 2. Compute UCB metric
            # ---------------------------------------------------------------------------- #

            UCB = sample_mean_ll + np.sqrt(alpha * np.log(j + 1) / pulls_ll)

            ucb_arm = np.argmax(UCB)
            reward = game.get_reward(ucb_arm)
            regret_exp_ll[j] = best_reward - reward
            arm_ll[ucb_arm, j] += 1

            # ---------------------------------------------------------------------------- #
            # 3. Recompute Sample Mean
            # ---------------------------------------------------------------------------- #
            pulls_ll[ucb_arm] += 1
            sample_mean_ll[ucb_arm] += (reward - sample_mean_ll[ucb_arm]) / pulls_ll[
                ucb_arm
            ]

        if pbar:
            pbar.set_description(
                f"Game: {game_i + 1} UCB_alpha_{alpha} exp: {exp+1} arm: {ucb_arm}"
            )
            pbar.update(1)

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        var_regret_ll += regret_exp_ll ** 2

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)

    arm_ll /= args.repeat

    if verbose:
        print(f"Arms pulled matrix {arm_ll}")
        print(f"Regret {regret_ll}")

    return regret_ll, var_regret_ll, arm_ll


def Asymp_UCB(args, game_i, l=0.2, pbar=None):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    for exp in range(args.repeat):
        ###########################################
        # 0. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        reward_exp_ll = np.zeros(args.n)
        arm_exp_ll = np.zeros(args.n)
        regret_exp_ll = np.zeros(args.n)

        times = np.ones(len(game))
        samples = np.zeros(len(game))
        for t in range(len(game)):
            samples[t] = game.get_reward(t)
            arm_exp_ll[t] = t
            reward_exp_ll[t] = samples[t]
            regret_exp_ll[t] = best_reward - samples[t]

        for t in range(len(game) + 1, args.n):
            conf_vector = np.zeros(len(game))
            # for i in range(len(game)):
            #     conf_vector[i] = (samples[i]/times[i])+np.sqrt(2*np.log(1+t*np.log(t)*np.log(t))/times[i])
            conf_vector = (samples / times) + np.sqrt(
                2 * np.log(1 + t * np.log(t) * np.log(t)) / times
            )

            arm = np.argmax(conf_vector)
            arm_exp_ll[t] = arm
            reward_exp_ll[t] = game.get_reward(arm)
            regret_exp_ll[t] = best_reward - reward_exp_ll[t]
            samples[arm] += reward_exp_ll[t]
            times[arm] += 1

        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_Asym_UCB_{l}_exp_{exp}_arm_{arm}")

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        # print(f"Regret {regret_exp_ll}")

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)
    print(f"Overall regret {regret_ll}")

    return regret_ll, var_regret_ll


def KL_div(a, b):
    pk = [a, 1 - a]
    qk = [b, 1 - b]
    return entropy(pk, qk)


def maxQ(p, C):
    # print (p)
    qlist = []
    eps = 0.01
    for i in range(1, 100):
        a = KL_div(p, i * eps)
        qlist.append(a)

    q1 = np.argwhere(np.array(qlist) < C)
    qlist1 = np.array(qlist)[q1]
    q2 = np.argmax(qlist1)
    return (q1[q2] + 1) * eps


def KL_UCB(args, game_i, l=0.2, pbar=None):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    for exp in range(args.repeat):
        ###########################################
        # 0. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        reward_exp_ll = np.zeros(args.n)
        arm_exp_ll = np.zeros(args.n)
        regret_exp_ll = np.zeros(args.n)

        times = np.ones(len(game))
        samples = np.zeros(len(game))
        for t in range(len(game)):
            samples[t] = game.get_reward(t)
            arm_exp_ll[t] = t
            reward_exp_ll[t] = samples[t]
            regret_exp_ll[t] = best_reward - samples[t]

        for t in range(len(game) + 1, args.n):
            conf_vector = np.zeros(len(game))
            for i in range(len(game)):
                C = np.log(1 + t * np.log(t) * np.log(t)) / times[i]
                p = samples[i] / times[i]
                conf_vector[i] = maxQ(p, C)
            print(conf_vector)
            arm = np.argmax(conf_vector)
            arm_exp_ll[t] = arm
            reward_exp_ll[t] = game.get_reward(arm)
            regret_exp_ll[t] = best_reward - reward_exp_ll[t]
            samples[arm] += reward_exp_ll[t]
            times[arm] += 1

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        # print(f"Regret {regret_exp_ll}")

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)
    print(f"Overall regret {regret_ll}")


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    regret_ll, var_regret_ll = UCB(
        args, game_i=0, alpha=2, pbar=tqdm(range(args.repeat))
    )
    print(f"UCB regret {regret_ll}")
