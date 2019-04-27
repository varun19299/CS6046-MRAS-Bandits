import numpy as np
from sacred import Experiment
from tupperware import tupperware
import matplotlib.pyplot as plt
import helper
from tqdm import tqdm
from helper import Game
import os

ex = Experiment("regret-minimisation")


@ex.config
def config():
    n = 10000  # event horizon
    repeat = 100  # repeat the experiment 100 times.
    games = [0, 0, 0, 0]  # Bernouli distributed
    games[0] = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    games[1] = [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
    games[2] = [0.5, 0.2, 0.1]
    games[3] = [0.5, 0.4, 0.3, 0.42, 0.35, 0.22, 0.33]


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
        arm_exp_ll = np.zeros(args.n)

        ##################################################
        # 1. UCB Initialisation
        ##################################################
        for j in range(len(game)):
            arm_exp_ll[j] = j
            reward = game.get_reward(j)
            reward_exp_ll[j] = reward

        sample_mean_ll = helper.get_sample_mean(reward_exp_ll, arm_exp_ll, n_arms=len(game), upto=len(game))

        for j in range(len(game), args.n):
            ##################################################
            # 2. Compute UCB metric
            ##################################################

            T_arm_ll = np.sum(arm_exp_ll[:j, None] == np.arange(len(game)), axis=0)
            UCB = sample_mean_ll + np.sqrt(alpha * np.log(j + 1) / T_arm_ll)

            ucb_arm = np.argmax(UCB)
            arm_exp_ll[j] = ucb_arm
            reward = game.get_reward(ucb_arm)
            reward_exp_ll[j] = reward

            ##################################################
            # 3. Recompute Sample Mean
            ##################################################
            sample_mean_ll[ucb_arm] += (reward - sample_mean_ll[ucb_arm]) / (T_arm_ll[ucb_arm] + 1)

        pbar.set_description(f"Game_{game_i + 1}_UCB_alpha_{alpha}_exp_{exp}_arm_{ucb_arm}")

        reward_ll += reward_exp_ll
        var_reward_ll += reward_exp_ll ** 2
        optimal_ll += arm_exp_ll == best_arm
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

    pbar = tqdm(range(3))
    for game_i in pbar:
        D = {}

        for alpha in [0.5, 2, 5]:
            regret_ll, var_regret_ll = UCB(args, game_i, alpha, pbar)
            d = {}
            d['regret'] = regret_ll
            d['var'] = var_regret_ll
            D[f'UCB-alpha={alpha}'] = d

        for params in [(1, 1), (0.2, 0.8)]:
            regret_ll, var_regret_ll, arms_ll = TS_beta(args, game_i, params, pbar)
            d = {}
            d['regret'] = regret_ll
            d['var'] = var_regret_ll
            d['arms_ll'] = arms_ll
            D[f'TS-beta-params-{params}'] = d

        gap_indep, gap_dep = regret_lower_bounds(args, game_i)

        d['regret'] = gap_indep
        d['var'] = np.zeros_like(gap_indep)
        D['gap-indpendent-minimax'] = d

        d['regret'] = gap_dep
        d['var'] = np.zeros_like(gap_dep)
        D['gap-dependent-minimax'] = d

        plot_regret(D, game_i, args, supress=True)
        plot_prob_arm(D, game_i, args, supress=True)
