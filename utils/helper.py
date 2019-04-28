import numpy as np
from sacred import Experiment

ex = Experiment("bandits_helper")

class Game(object):
    '''
    Constructs a game instance
    '''

    def __init__(self, game_ll):
        '''
        Initialise a game instance with
        : game_ll as list of arms
        : distribution as bernoulli
        '''
        self.game_ll = np.array(game_ll)

    def get_reward(self, arm):
        '''
        Get a reward for arm i

        :arm from 0 to len(game) - 1
        '''
        reward = np.random.binomial(n = 1, p = self.game_ll[arm])
        return reward

    def __len__(self):
        '''
        Returns number of arms
        '''
        return len(self.game_ll)

@ex.config
def config():
    reward_ll = np.array([0,1,1,2])
    arms_ll = np.array([0,1,0,1])

@ex.capture
def get_sample_mean(reward_ll, arms_ll, n_arms = None, upto = None):
    '''
    Computes sample mean per arm
    : reward_ll: upto time n
    : arm_ll: which arm was picked, size n

    Returns
    : sample_mean_ll : Size K
    '''
    if upto:
        reward_ll = reward_ll[:upto]
        arms_ll = arms_ll[:upto]
    if not n_arms:
        n_arms = np.max(arms_ll) + 1

    sample_mean_ll = np.zeros(n_arms)

    for i in range(n_arms):
        ii = np.where(arms_ll == i)[0]
        sample_mean = np.sum(reward_ll[ii])/ len(reward_ll[ii])
        sample_mean_ll[i] = sample_mean

    return sample_mean_ll

@ex.automain
def main():
    print(get_sample_mean())
