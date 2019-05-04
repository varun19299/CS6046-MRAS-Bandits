# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:03:07 2019

@author: Dell
"""

import numpy as np
from scipy.stats import multivariate_normal
from sacred import Experiment
from utils.tupperware import tupperware
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


def MRAS_gaussian(args, game_i, l=0.2, pbar=None):
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
        sigma_0 = np.ones(len(game))
        theta = 1 / len(game) * np.ones(len(game))
        sigma = sigma_0

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
            cov = np.diag(sigma, k = 0)
            arms1 = np.random.multivariate_normal(dist, cov, phases[j])
            arms = np.argmax(arms1, axis = 1)

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
            C = 1
            """
            num = np.exp(C*sample_mean)*(arms1)/multivariate_normal.pdf(arms1, theta, cov)
            den = np.exp(C*sample_mean)/multivariate_normal.pdf(arms1, theta, cov)
            theta = np.sum(num)/np.sum(den)
            
            # theta_0 += (theta - theta_0) / (j + 1)
            num = np.exp(C*sample_mean)*(sigma-theta)*(sigma-theta)/multivariate_normal.pdf(arms1, theta, cov)
            den = np.exp(C*sample_mean)/multivariate_normal.pdf(arms1, theta, cov)
            sigma = np.sum(num)/np.sum(den)
            """
            
            
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
