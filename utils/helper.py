import numpy as np
from sacred import Experiment

ex = Experiment("bandits_helper")


class Game(object):
    """
    Constructs a game instance
    """

    def __init__(self, game_ll):
        """
        Initialise a game instance with
        : game_ll as list of arms
        : distribution as bernoulli
        """
        self.game_ll = np.array(game_ll)

    def get_reward(self, arm):
        """
        Get a reward for arm i

        :arm from 0 to len(game) - 1
        """
        reward = np.random.binomial(n=1, p=self.game_ll[arm])
        return reward

    def __len__(self):
        """
        Returns number of arms
        """
        return len(self.game_ll)
