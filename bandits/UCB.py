import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
import os

ex = Experiment("UCB-regret-minimisation")
ex.add_config('configs/base-config.yaml')


def UCB(args, game_i, alpha=0.2, pbar=None):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    # print(f"\n\nRunning game {game_i} with UCB-alpha = {alpha}")
    # print(f"Arms distribution used {game_ll}")
    #####################################################
    # 0. Rewards collected and optimal arm pull (overall)
    #####################################################

    reward_ll = np.zeros(args.n)
    var_reward_ll = np.zeros(args.n)
    optimal_ll = np.zeros(args.n)
    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)

    # pbar = tqdm(range(args.repeat))
    for exp in range(args.repeat):

        ###########################################
        # 0. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        reward_exp_ll = np.zeros(args.n)
        n_pulls = np.zeros(len(game))
        optimal_exp_ll = np.zeros(args.n)
        sample_mean = np.zeros(len(game))

        ##################################################
        # 1. UCB Initialisation
        ##################################################
        for j in range(len(game)):
            reward = game.get_reward(j)
            reward_exp_ll[j] = reward
            n_pulls[j] += 1
            sample_mean[j] += (reward - sample_mean[j]) / n_pulls[j]
            optimal_exp_ll += j == best_arm

        for j in range(len(game), args.n):
            ##################################################
            # 2. Compute UCB metric
            ##################################################

            UCB = sample_mean + np.sqrt(alpha * np.log(j + 1) / n_pulls)

            ucb_arm = np.argmax(UCB)
            reward = game.get_reward(ucb_arm)
            reward_exp_ll[j] = reward

            ##################################################
            # 3. Recompute Sample Mean
            ##################################################
            n_pulls[ucb_arm] += 1
            sample_mean[ucb_arm] += (reward - sample_mean[ucb_arm]) / n_pulls[ucb_arm]

            optimal_exp_ll += ucb_arm == best_arm

        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_UCB_alpha_{alpha}_exp_{exp}_arm_{ucb_arm}")

        reward_ll += reward_exp_ll
        var_reward_ll += reward_exp_ll ** 2
        optimal_ll += optimal_exp_ll
        regret_exp_ll = np.arange(1, args.n + 1) * best_reward - np.cumsum(reward_exp_ll)
        regret_ll += regret_exp_ll
        var_regret_ll += regret_exp_ll ** 2

    # optimal_ll[990:] += 23*len(game)/10
    reward_ll = reward_ll / args.repeat
    regret_ll = regret_ll / (1 * args.repeat)
    var_reward_ll /= args.repeat
    var_reward_ll -= reward_ll ** 2
    var_reward_ll = np.sqrt(var_reward_ll)
    var_regret_ll /= args.repeat
    var_regret_ll -= regret_ll ** 2
    var_regret_ll /= np.sqrt(np.arange(1, args.n + 1))
    var_regret_ll = np.sqrt(var_regret_ll)

    # print("No of optimal pulls per trial", optimal_ll)
    # print(f"Regret {regret_ll}")

    return regret_ll, var_regret_ll


@ex.automain
def main(_run):
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)

    regret_ll, var_regret_ll = UCB(args, game_i=0, alpha=2, pbar=None)
    print(f"UCB regret {regret_ll}")
