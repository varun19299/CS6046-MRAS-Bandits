import numpy as np
from sacred import Experiment
from utils.tupperware import tupperware
import matplotlib.pyplot as plt
from utils import helper
from tqdm import tqdm
from utils.helper import Game
from scipy import stats
import os

ex = Experiment("Thomson-Sampling-regret-minimisation")
ex.add_config("configs/base-config.yaml")


def MRAS_categ(args, game_i, pbar=None, verbose=False):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    if verbose:
        print(f"\n\nRunning game {game_i} with MRAS Categorical")
        print(f"Arms distribution used {game_ll}")
    #####################################################
    # 1. Rewards collected and optimal arm pull (overall)
    #####################################################

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    pbar = tqdm(range(args.repeat))
    for exp in range(args.repeat):
        ###########################################
        # 2. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        regret_exp_ll = np.zeros(args.n)
        sample_mean_ll = np.zeros(len(game))
        theta = np.ones(len(game)) / len(game)
        pulls_ll = np.zeros(len(game))
        last_pulled_ll = np.zeros(len(game))

        ##################################################
        # 3. Initialisation
        ##################################################
        for k in range(10):
            for j in range(len(game)):
                arm_ll[j, k * len(game) + j] += 1
                reward = game.get_reward(j)
                regret_exp_ll[j] = best_reward - reward

                # Update sample mean
                pulls_ll[j] += 1
                last_pulled_ll[j] = k * len(game) + j + 1
                sample_mean_ll[j] = (reward - sample_mean_ll[j]) / pulls_ll[j]

        for j in range(len(game) * 10, args.n):
            # pull sample
            samples = np.random.choice(len(game), 20, p=theta)
            arm = stats.mode(samples)[0]
            arm_ll[arm, j] += 1

            # Update sample mean
            reward = game.get_reward(arm)
            regret_exp_ll[j] = best_reward - reward
            pulls_ll[arm] += 1
            sample_mean_ll[arm] += (reward - sample_mean_ll[arm]) / pulls_ll[arm]

            last_pulled_ll[arm] = j + 1

            # Update theta
            theta = np.exp(pulls_ll * sample_mean_ll) / theta
            theta = theta / np.sum(theta)

            # if verbose:
            #     print(f"Turn {j+1} of {args.n}  Arm pulled {arm}")
            #     print(f"Updated sample means {sample_mean_ll}")
            #     print(f"New theta: {theta}")

        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_MRAS_Categ_exp_{exp}_arm_{arm}")
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

    print(f"Final theta: {theta}")

    return regret_ll, var_regret_ll, arm_ll


def MRAS_categ_elite(args, game_i, pbar=None, verbose=False):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    if verbose:
        print(f"\n\nRunning game {game_i} with MRAS Categorical")
        print(f"Arms distribution used {game_ll}")
    #####################################################
    # 1. Rewards collected and optimal arm pull (overall)
    #####################################################

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    pbar = tqdm(range(args.repeat))
    for exp in range(args.repeat):
        ###########################################
        # 2. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        regret_exp_ll = np.zeros(args.n)
        sample_mean_ll = np.zeros(len(game))
        theta = np.ones(len(game)) / len(game)
        pulls_ll = np.zeros(len(game))

        count = 0

        ##################################################
        # 3. Initialisation
        ##################################################
        for k in range(10):
            for j in range(len(game)):
                arm_ll[j, k * len(game) + j] += 1
                reward = game.get_reward(j)
                regret_exp_ll[j] = best_reward - reward

                # Update sample mean
                pulls_ll[j] += 1
                sample_mean_ll[j] = (reward - sample_mean_ll[j]) / pulls_ll[j]

                count += 1

        # Samples to take
        N = 400
        alpha = 4
        kai = 0
        kai_bar = 0
        rho = 0.7
        epi_J = 1e-6
        phase = 1

        while count < args.n:
            # pull sample
            arms = np.random.choice(len(game), N, p=theta)
            mask = np.zeros(len(game))
            J_vec = np.zeros(N)
            J_vec_arms = np.zeros(N, dtype=np.int)

            for e, arm in enumerate(arms):
                arm_ll[arm, count] += 1

                # Update sample mean
                reward = game.get_reward(arm)
                regret_exp_ll[count] = best_reward - reward
                pulls_ll[arm] += 1
                sample_mean_ll[arm] += (reward - sample_mean_ll[arm]) / pulls_ll[arm]
                J_vec[e] = sample_mean_ll[arm]
                J_vec_arms[e] = arm
                count += 1

                if count >= args.n:
                    break

            # Update elite set
            k = int((1 - rho) * N) - 1
            kai_bar = np.partition(J_vec, k)[k - 1]

            if kai_bar - kai > epi_J:
                kai = kai_bar
            else:
                attempts = 0
                rho_bar = rho
                while rho_bar > epi_J:
                    attempts += 1
                    rho_bar = 0.9 * rho_bar
                    k = int((1 - rho_bar) * N) - 1
                    kai_bar = np.partition(J_vec, k)[k]
                    if kai_bar - kai > epi_J:
                        kai = kai_bar
                        rho = rho_bar
                        break
                    # if attempts > 20:
                    #     break
                if rho > rho_bar:
                    N = int(alpha * N)

            # Find elite mask
            if len(J_vec_arms[np.where(J_vec >= kai)]) > 0:
                for arm in J_vec_arms[np.where(J_vec >= kai)]:
                    mask[arm] += 1

                # for arm in J_vec_arms[np.where(J_vec >= kai)]:
                #     mask[arm] += 1
                #     mult[arm] = 1

                # Update theta
                theta[mask > 0] = (
                    np.exp(phase * sample_mean_ll[mask > 0])
                    * mask[mask > 0]
                    / theta[mask > 0]
                )
                theta[mask == 0] = 0
                theta = theta / np.sum(theta[mask > 0])
                phase += 1

            # if verbose:
            #     print(f"\nTurn {count} of {args.n}  Arm pulled {arms}")
            #     print(f"Updated sample means {sample_mean_ll}")
            #     print(f"Mask {mask}")
            #     print(f"New theta: {theta}")
        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_MRAS_Categ_exp_{exp}_arm_{arm}")
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

    print(f"Final theta: {theta}")

    return regret_ll, var_regret_ll, arm_ll


def gaussian(x, mu=0, sigma=0.1):
    num = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    denom = 2 * np.pi * sigma
    return num / denom


def MRAS_gaussian(args, game_i, pbar=None, verbose=False):
    game_ll = args.games[game_i]
    game = Game(game_ll)
    best_reward, best_arm = np.max(game.game_ll), np.argmax(game.game_ll)

    if verbose:
        print(f"\n\nRunning game {game_i} with MRAS Categorical")
        print(f"Arms distribution used {game_ll}")
    #####################################################
    # 1. Rewards collected and optimal arm pull (overall)
    #####################################################

    regret_ll = np.zeros(args.n)
    var_regret_ll = np.zeros(args.n)
    arm_ll = np.zeros((len(game), args.n))

    pbar = tqdm(range(args.repeat))
    for exp in range(args.repeat):
        ###########################################
        # 2. Rewards collected and optimal arm pull (for experiment)
        ###########################################

        regret_exp_ll = np.zeros(args.n)
        sample_mean_ll = np.zeros(len(game))
        pulls_ll = np.zeros(len(game))
        num_ll = np.zeros(len(game))
        num_var_ll = np.zeros(len(game))
        denom_ll = np.zeros(len(game))

        ##################################################
        # 3. Initialisation
        ##################################################

        mu_ll = 0.5 * np.ones(len(game))
        sigma_ll = 0.1 * np.ones_like(mu_ll)

        for k in range(10):
            for j in range(len(game)):
                arm_ll[j, k * len(game) + j] += 1
                reward = game.get_reward(j)
                regret_exp_ll[j] = best_reward - reward

                # Update sample mean
                pulls_ll[j] += 1
                sample_mean_ll[j] = (reward - sample_mean_ll[j]) / pulls_ll[j]

                num_ll[j] += (
                    np.exp(sample_mean_ll[j] * (j + 1))
                    * reward
                    / gaussian(reward, mu=mu_ll[j], sigma=sigma_ll[j])
                )
                num_var_ll[j] += (
                    np.exp(sample_mean_ll[j] * (j + 1))
                    * (reward - sample_mean_ll[j])
                    * (reward - sample_mean_ll[j])
                    / gaussian(reward, mu=mu_ll[j], sigma=sigma_ll[j])
                )
                denom_ll[j] += np.exp(sample_mean_ll[j] * (j + 1)) / gaussian(
                    reward, mu=mu_ll[j], sigma=sigma_ll[j]
                )

        for j in range(len(game) * 10, args.n):
            # pull sample
            samples = np.random.normal(loc=mu_ll, scale=0.1)
            arm = np.argmax(samples)
            arm_ll[arm, j] += 1

            # Update sample mean
            reward = game.get_reward(arm)
            regret_exp_ll[j] = best_reward - reward
            pulls_ll[arm] += 1
            sample_mean_ll[arm] = (reward - sample_mean_ll[arm]) / pulls_ll[arm]

            # Update params
            num_ll[arm] += (
                np.exp(sample_mean_ll[arm] * (j + 1))
                * sample_mean_ll[arm]
                / gaussian(sample_mean_ll[arm], mu=mu_ll[arm], sigma=0.1)
            )

            denom_ll[arm] += np.exp(sample_mean_ll[arm] * (j + 1)) / gaussian(
                sample_mean_ll[arm], mu=mu_ll[arm], sigma=0.1
            )

            mu_ll = num_ll / denom_ll
            # if verbose:
            #     print(f"Turn {j+1} of {args.n}  Arm pulled {arm}")
            #     print(f"Updated sample means {sample_mean_ll}")
            #     print(f"New theta: {theta}")

        if pbar:
            pbar.set_description(f"Game_{game_i + 1}_MRAS_Categ_exp_{exp}_arm_{arm}")
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

    print(f"Final mu: {mu_ll}")

    return regret_ll, var_regret_ll, arm_ll


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    regret_ll, _, arm_ll = MRAS_categ_elite(args, game_i=0, verbose=True)
    print(f"Arms pulled {arm_ll}")
    print(f"MRAS regret {regret_ll}")
    plt.plot(regret_ll)
    plt.show()
