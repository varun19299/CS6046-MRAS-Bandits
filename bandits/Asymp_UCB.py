# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:08:59 2019

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


def Asymp_UCB(args, game_i, l=0.2, pbar=None):
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
        
        times = np.ones(len(game))
        samples = np.zeros(len(game))
        for t in range(len(game)):
           samples[t] = game.get_reward(t)
           arm_exp_ll[t] = t
           reward_exp_ll[t] = samples[t]
           regret_exp_ll[t] = best_reward- samples[t]
        
        for t in range(len(game)+1, args.n):
            conf_vector = np.zeros(len(game))
            for i in range(len(game)):
                conf_vector[i] = (samples[i]/times[i])+np.sqrt(2*np.log(1+t*np.log(t)*np.log(t))/times[i])
            arm = np.argmax(conf_vector)
            arm_exp_ll[t] = arm
            reward_exp_ll[t] = game.get_reward(arm)
            regret_exp_ll[t] = best_reward- reward_exp_ll[t]
            
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
    Asymp_UCB(args, game_i=0, l=0.1)
