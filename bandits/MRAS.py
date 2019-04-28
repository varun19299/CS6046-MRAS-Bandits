import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
import os

ex = Experiment("MRAS-regret-minimisation")


@ex.config
def config():
    n = 2000  # event horizon
    repeat = 10  # repeat the experiment 100 times.
    games = [0, 0, 0, 0]  # Bernouli distributed
    games[0] = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    games[1] = [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
    games[2] = [0.5, 0.2, 0.1]
    games[3] = [0.5, 0.4, 0.3, 0.42, 0.35, 0.22, 0.33]


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


@ex.automain
def main(_run):
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)
    MRAS(args, game_i=0, l=0.1)
