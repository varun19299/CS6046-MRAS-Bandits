import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
import os

ex = Experiment("Thomson-Sampling-regret-minimisation")


@ex.config
def config():
    n = 2000  # event horizon
    repeat = 10  # repeat the experiment 100 times.
    games = [0, 0, 0, 0]  # Bernouli distributed
    games[0] = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    games[1] = [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
    games[2] = [0.5, 0.2, 0.1]
    games[3] = [0.5, 0.4, 0.3, 0.42, 0.35, 0.22, 0.33]


def TS_beta(args, game_i, params, pbar=None):
    alpha, beta = params
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    # print(f"\n\nRunning game {game_i} with TS_Beta {params}")
    # print(f"Arms distribution used {game_ll}")
    #####################################################
    # 1. Rewards collected and optimal arm pull (overall)
    #####################################################

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    # pbar = tqdm(range(args.repeat))
    for exp in range(args.repeat):
        ###########################################
        # 2. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        regret_exp_ll = np.zeros(args.n)

        alpha_arms = np.ones(len(game)) * alpha
        beta_arms = np.ones(len(game)) * beta

        ##################################################
        # 3. Initialisation
        ##################################################
        for j in range(len(game)):
            arm_ll[j, j] += 1
            reward = game.get_reward(j)
            regret_exp_ll[j] = best_reward - reward
            alpha_arms[j] += reward
            beta_arms[j] += 1 - reward

        for j in range(len(game), args.n):
            ##################################################
            # 2. pull samples
            ##################################################

            samples = np.random.beta(alpha_arms, beta_arms)
            arm_to_pull = np.argmax(samples)

            # print(f"Arm pulled {arm_to_pull}")

            arm_ll[arm_to_pull, j] += 1
            reward = game.get_reward(arm_to_pull)
            regret_exp_ll[j] = best_reward - reward

            alpha_arms[arm_to_pull] += reward
            beta_arms[arm_to_pull] += 1 - reward

        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_TS_Beta_{params}_exp_{exp}_arm_{arm_to_pull}")
        regret_exp_ll = np.cumsum(regret_exp_ll)
        regret_ll += regret_exp_ll
        var_regret_ll += regret_exp_ll ** 2

    regret_ll = regret_ll / args.repeat

    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.arange(1, args.n + 1)
    var_regret_ll = np.sqrt(var_regret_ll)

    arm_ll /= args.repeat

    return regret_ll, var_regret_ll, arm_ll


@ex.automain
def main(_run):
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)

    regret_ll, _, _ = TS_beta(args, game_i=0, params=(1, 1), pbar=tqdm(range(3)))
    print(f"TS regret {regret_ll}")
