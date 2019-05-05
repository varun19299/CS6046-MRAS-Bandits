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


def update_Categorical(phase, theta, sample_mean, mask, C=10, epsilon=1e-12, upper=1e12):
    '''
    Vectorised op
    :param sample_means:
    :param theta: Categorical params
    :param C: constant for exp
    :return: theta
    '''
    num = np.exp(C * phase * sample_mean, dtype = np.float128)
    theta[np.where(theta == 0)] = C

    # small epsilon for ill conditioned
    theta = num / theta
    theta = theta * mask + epsilon
    theta = theta / theta.sum()

    return theta.astype(np.float64)


def update_Dirchlet(phase, theta, sample_mean, arms1, C=1, summation=1000, epsilon=1e-12):
    '''
    Dirchelt update
    :param phase: current phase
    :param theta: dirichlet params
    :param sample_mean:
    :param arms1: contains n_k * K array
    :param C: constant for exp
    :param summation: homogenity assumption
    :param epsilon: small float value to prevent ill conditioning
    :return: thetas
    '''
    num = np.zeros_like(theta)
    denom = np.zeros_like(theta)
    # a_ll = np.argmax(arms1, axis=1)
    # num = np.sum(np.exp(C * (phase + 1) * sample_mean[a_ll], dtype = np.float128) * np.log(arms1) / (dirichlet.pdf(arms1.T, theta) + epsilon), axis = 0)
    # denom = np.sum(np.exp(C * (phase + 1) * sample_mean[a_ll], dtype=np.float128) / dirichlet.pdf(arms1.T, theta) + epsilon, axis = 0)

    for arm in arms1:
        a = np.argmax(arm)
        num += np.exp(C * (phase + 1) * sample_mean[a], dtype = np.float128) * (np.log(arm)) / (dirichlet.pdf(arm, theta) + epsilon)
        denom += np.exp(C * (phase + 1) * sample_mean[a], dtype = np.float128) / (dirichlet.pdf(arm, theta) + epsilon)

    theta = summation * np.exp(num / denom)
    theta = np.around(theta)
    theta[np.where(theta == 0)] = epsilon
    return theta.astype(np.float64)

def MRAS_Categorical(args, game_i, l=0.2, pbar=None):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))
    theta_0 = 1 / len(game) * np.ones(len(game))

    for exp in range(args.repeat):
        ############################################################
        # 1. Rewards collected and optimal arm pull (for experiment)
        ############################################################

        reward_exp_ll = np.zeros(args.n)
        arm_exp_ll = np.zeros(args.n)
        regret_exp_ll = np.zeros(args.n)
        sample_mean = np.zeros(len(game))
        n_pulls = np.zeros(len(game))

        ############################################################
        # 2. Initialise Theta
        ############################################################
        theta = theta_0

        ############################################################
        # 3. Set phase
        ############################################################
        phases = args.n // len(game) * [len(game)]
        if args.n % len(game):
            phases += [args.n % len(game)]
        phases = np.array(phases)
        pulls = np.ones_like(phases)
        # phases, pulls = get_phases_pulls(args.n)

        t = 0
        for j in range(len(phases)):
            # Between 0 and len(game) (not included upper)
            ############################################################
            # 4. Pull n_k arms, M_k times each
            ############################################################
            dist = (1 - l) * theta + l * theta_0
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
            # 5. Update theta
            ##########################################
            theta = update_Categorical(j + 1, theta, sample_mean, mask)

            if not pbar:
                pass
                # print(f"Phase {j + 1} Arms {arms} theta {theta}")
            else:
                pbar.set_description(f"Game {game_i + 1} Exp {exp} Phase {j + 1} Arms {arms}")

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        print(f"Phase {j + 1} Arms {arms} theta {theta}")

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)
    print(f"Overall regret {regret_ll}")


def MRAS_Dirichlet(args, game_i, l=0.2, pbar=None):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    theta_0 = np.ones(len(game)) * 10

    for exp in range(args.repeat):
        ############################################################
        # 1. Rewards collected and optimal arm pull (for experiment)
        ############################################################

        reward_exp_ll = np.zeros(args.n)
        regret_exp_ll = np.zeros(args.n)
        sample_mean = np.zeros(len(game))
        n_pulls = np.zeros(len(game))

        ############################################################
        # 2. Initialise Theta
        ############################################################
        theta = theta_0

        ############################################################
        # 3. Set phase
        ############################################################
        phases = args.n // len(game) * [len(game)]
        if args.n % len(game):
            phases += [args.n % len(game)]
        phases = np.array(phases)
        pulls = np.ones_like(phases)
        # phases, pulls = get_phases_pulls(args.n)

        t = 0
        for j in range(len(phases)):
            # Between 0 and len(game) (not included upper)

            ############################################################
            # 4. Pull n_k arms, M_k times each
            ############################################################
            dist = (1 - l) * theta + l * theta_0
            # print(f"Arms dist {dist}")
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
            # 5. Update theta
            ##########################################
            theta = update_Dirchlet(j + 1, theta, sample_mean, arms1)

        print(f"Phase {j + 1} Arms {arms} theta {theta}")

        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll

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
    # MRAS_Dirichlet(args, game_i=0, l=0.1, pbar=None)
    MRAS_Categorical(args, game_i=0, l=0.1, pbar=None)
