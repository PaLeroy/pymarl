from src.matchmaking.matchmaking import Matchmaking
import numpy as np


class RandomMatchmaking(Matchmaking):

    def list_combat(self, agent_dict):
        rand_ = np.random.random_sample()
        if rand_ < 0.5:
            return [(0, 1), ]
        else:
            return [(1, 0), ]
