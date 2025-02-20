import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
import os

ex = Experiment("Thomson-Sampling-regret-minimisation")
ex.add_config("configs/base-config.yaml")


def TS_beta(args, game_i, params, pbar=None, verbose=False):
    alpha, beta = params
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    if verbose:
        print(f"\n\nRunning game {game_i} with TS_Beta {params}")
        print(f"Arms distribution used {game_ll}")

    # ---------------------------------------------------------------------------- #
    # 1. Rewards collected and optimal arm pull (overall)
    # ---------------------------------------------------------------------------- #

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    for exp in range(args.repeat):
        # ---------------------------------------------------------------------------- #
        # 2. Rewards collected and optimal arm pull (for experiment)
        # ---------------------------------------------------------------------------- #

        regret_exp_ll = np.zeros(args.n)

        alpha_arms = np.ones(len(game)) * alpha
        beta_arms = np.ones(len(game)) * beta

        # ---------------------------------------------------------------------------- #
        # 3. Initialisation
        # ---------------------------------------------------------------------------- #
        for j in range(len(game)):
            arm_ll[j, j] += 1
            reward = game.get_reward(j)
            regret_exp_ll[j] = best_reward - reward
            alpha_arms[j] += reward
            beta_arms[j] += 1 - reward

        for j in range(len(game), args.n):
            # ---------------------------------------------------------------------------- #
            # 4. pull samples
            # ---------------------------------------------------------------------------- #

            samples = np.random.beta(alpha_arms, beta_arms)
            arm_to_pull = np.argmax(samples)

            arm_ll[arm_to_pull, j] += 1
            reward = game.get_reward(arm_to_pull)
            regret_exp_ll[j] = best_reward - reward

            alpha_arms[arm_to_pull] += reward
            beta_arms[arm_to_pull] += 1 - reward

        if pbar:
            pbar.set_description(
                f"Game: {game_i + 1} TS_Beta_{params} exp: {exp+1} arm: {arm_to_pull}"
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

    return regret_ll, var_regret_ll, arm_ll


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    regret_ll, _, _ = TS_beta(
        args, game_i=1, params=(1, 1), pbar=tqdm(range(args.repeat))
    )
    print(f"TS regret {regret_ll}")
    plt.plot(regret_ll)
    plt.show()
