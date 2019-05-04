import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
import os

ex = Experiment("MRAS-regret-minimisation")
ex.add_config('configs/base-config.yaml')


def get_phases_pulls(n):
    n_ll = np.arange(5, n + 1)[1:]
    sum = 0
    phases = []
    pulls = []
    i = 0
    while True:
        if sum + n_ll[i] > n:
            n_ll[i] = n - sum
            phases.append(n_ll[i])
            pulls.append(1)
            break
        else:
            sum += n_ll[i]
            phases.append(n_ll[i])
            pulls.append(1)
        i += 1

    return np.array(phases), np.array(pulls)


def G(phase, sample_mean, theta, C=1):
    '''
    Vectorised op
    :param sample_means:
    :param theta:
    :param C:
    :return:
    '''
    num = np.exp(C * phase * sample_mean)
    theta[np.where(theta == 0)] = C
    return num / theta


def MRAS_Categorical(args, game_i, l=0.2, pbar=None):
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
        sample_mean = np.zeros(len(game))
        n_pulls = np.zeros(len(game))

        theta_0 = 1 / len(game) * np.ones(len(game))
        theta = 1 / len(game) * np.ones(len(game))

        phases = args.n // len(game) * [len(game)]
        if args.n % len(game):
            phases += [args.n % len(game)]
        phases = np.array(phases)
        pulls = np.ones_like(phases)
        # phases, pulls = get_phases_pulls(args.n)

        t = 0
        for j in range(len(phases)):
            # Between 0 and len(game) (not included upper)
            dist = (1 - l) * theta + l * theta_0
            print(f"Arms dist {dist}")
            arms = np.random.choice(len(game), phases[j], p=dist)
            mask = np.zeros(len(game))

            for arm in arms:
                for _ in range(pulls[j]):
                    reward = game.get_reward(arm)
                    reward_exp_ll[t] = reward
                    mask[arm] += 1
                    regret_exp_ll[t] = best_reward - reward
                    n_pulls[arm] += 1
                    sample_mean[arm] += (reward - sample_mean[arm]) / n_pulls[arm]
                    t += 1

                    if t >= args.n:
                        break

            ###########################################
            # 1. Update theta
            ###########################################
            theta = G(j + 1, sample_mean, theta) * mask
            theta = theta / theta.sum()
            # theta_0 += (theta - theta_0) / (j + 1)

            print(f"Phase {j + 1} Arms {arms} theta {theta}")

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        # print(f"Regret {regret_exp_ll}")

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)
    print(f"Overall regret {regret_ll}")


def MRAS_Dirichlet(args, game_i, l=0.2, pbar=None, summation=1000):
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
        sample_mean = np.zeros(len(game))
        n_pulls = np.zeros(len(game))

        theta_0 = np.ones(len(game))
        theta = theta_0

        phases = args.n // len(game) * [len(game)]
        if args.n % len(game):
            phases += [args.n % len(game)]
        phases = np.array(phases)
        pulls = np.ones_like(phases)
        # phases, pulls = get_phases_pulls(args.n)

        t = 0
        for j in range(len(phases)):
            # Between 0 and len(game) (not included upper)
            dist = (1 - l) * theta + l * theta_0
            print(f"Arms dist {dist}")
            # arms = np.random.choice(len(game), phases[j], p=dist)
            arms1 = np.random.dirichlet(dist, phases[j])
            arms = np.argmax(arms1, axis=1)
            mask = np.zeros(len(game))

            for arm in arms:
                for _ in range(pulls[j]):
                    reward = game.get_reward(arm)
                    reward_exp_ll[t] = reward
                    mask[arm] += 1
                    regret_exp_ll[t] = best_reward - reward
                    n_pulls[arm] += 1
                    sample_mean[arm] += (reward - sample_mean[arm]) / n_pulls[arm]
                    t += 1

                    if t >= args.n:
                        break

            ###########################################
            # 1. Update theta
            ##########################################
            C = 1
            # for i in range(len(theta)):
            #     num = 0
            #     den = 0
            #     for arm in arms1:
            #         a = np.argmax(arm)
            #         num += np.exp(C * sample_mean[a]) * (np.log(arms1[i][a])) / dirichlet.pdf(arm, theta)
            #         den += np.exp(C * sample_mean[a]) / dirichlet.pdf(arm, theta)
            #     theta[i] = summation * num / den

            num = np.zeros_like(theta)
            denom = np.zeros_like(theta)
            for arm in arms1:
                a = np.argmax(arm)
                num += np.exp(C * sample_mean[a]) * (np.log(arm)) / dirichlet.pdf(arm, theta)
                denom += np.exp(C * sample_mean[a]) / dirichlet.pdf(arm, theta)

            theta = summation * np.exp(num / denom)

            print(f"Phase {j + 1} Arms {arms} theta {theta}")

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
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)
    MRAS_Categorical(args, game_i=1, l=0.1)
