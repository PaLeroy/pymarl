from src.matchmaking.matchmaking import Matchmaking


class SingleMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        return [(0, 0), ]

    def update_elo(self, agent_dict, list_episode_matches, win_list):
        pass