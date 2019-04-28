import matplotlib.pyplot as plt
import numpy as np
import os
from sacred import Experiment
from tqdm import tqdm

from utils import helper
from utils.helper import Game
from utils.tupperware import tupperware
from bandits.UCB import UCB
from bandits.MRAS import MRAS_Categorical
from bandits.TS_beta import TS_beta

ex = Experiment("regret-minimisation")
ex.add_config('configs/base-config.yaml')

def plot_regret(D, game, args, supress=False):
    '''
    D dictionary of algorithm names:
    (with keys)
    : regret 
    : var 
    '''

    l = np.arange(1, args.n + 1)

    for i, alg in enumerate(D):
        # plt.plot(l,D[alg]['regret'])
        plt.errorbar(l[::100], D[alg]['regret'][::100], D[alg]['var'][i * 5::100])

    plt.legend(list(D.keys()))
    plt.xlabel("Trial (n) Averaged over 100 experiments")
    plt.ylabel("Regret averaged over exps")
    plt.title(f"Regret Curves for game {game + 1}")
    plt.savefig(f"logs/Regret-game-{game + 1}.png")

    if supress:
        plt.close()
    else:
        plt.show()


def plot_prob_arm(D, game, args, supress=False):
    '''
    D dictionary of algorithm names:
    (with keys)
    : regret
    : var
    :armpull
    '''

    f, ax = plt.subplots(2, 1, sharex=True)
    D = {key: D[key] for key in D if "TS" in key}

    for i, alg in enumerate(D):
        legend = []
        for e, arm_pull in enumerate(D[alg]['arms_ll']):
            ax[i].plot(np.arange(1, args.n + 1), arm_pull, alpha=0.5)
            legend.append(f"{alg}_arm_{e + 1}")
            # ax1.legend(legend)
            ax[i].set_title(f"alg_{alg} for game {game + 1}")

    plt.xlabel("Trial (n) Averaged over 100 experiments")
    plt.ylabel("Posterior probability of arm pulls")
    plt.savefig(f"logs/Ppulls-game-{game + 1}.png")

    if supress:
        plt.close()
    else:
        plt.show()


def instance_lower_bound(n, gaps):
    gaps = gaps[np.where(gaps > 0)]

    lb = 0
    for gap in gaps:
        lb_add = (0.5 * np.log(n) + np.log(gap / 2)) / gap
        lb_add[np.where(lb_add < 0)] = 0
        lb += lb_add
    return lb * 8 / 9


def regret_lower_bounds(args, game_i):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_mean, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    gap_indep = 1 / 27 * np.sqrt((len(game) - 1) * np.arange(1, args.n + 1))
    gap_dep = instance_lower_bound(np.arange(1, args.n + 1), best_mean - game_ll)

    return gap_indep, gap_dep


@ex.automain
def main(_run):
    args = _run.config
    print(f"Configs used {args}")
    args = tupperware(args)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    MRAS_Categorical(args, game_i=0, l=0.1)
    regret_ll, _, _ = TS_beta(args, game_i=0, params=(1, 1), pbar=tqdm(range(3)))
    print(f"TS regret {regret_ll}")

    # pbar = tqdm(range(3))
    # for game_i in pbar:
    #     D = {}
    #
    #     for alpha in [0.5, 2, 5]:
    #         regret_ll, var_regret_ll = UCB(args, game_i, alpha, pbar)
    #         d = {}
    #         d['regret'] = regret_ll
    #         d['var'] = var_regret_ll
    #         D[f'UCB-alpha={alpha}'] = d
    #
    #     for params in [(1, 1), (0.2, 0.8)]:
    #         regret_ll, var_regret_ll, arms_ll = TS_beta(args, game_i, params, pbar)
    #         d = {}
    #         d['regret'] = regret_ll
    #         d['var'] = var_regret_ll
    #         d['arms_ll'] = arms_ll
    #         D[f'TS-beta-params-{params}'] = d
    #
    #     gap_indep, gap_dep = regret_lower_bounds(args, game_i)
    #
    #     d['regret'] = gap_indep
    #     d['var'] = np.zeros_like(gap_indep)
    #     D['gap-indpendent-minimax'] = d
    #
    #     d['regret'] = gap_dep
    #     d['var'] = np.zeros_like(gap_dep)
    #     D['gap-dependent-minimax'] = d
    #
    #     plot_regret(D, game_i, args, supress=True)
    #     plot_prob_arm(D, game_i, args, supress=True)
