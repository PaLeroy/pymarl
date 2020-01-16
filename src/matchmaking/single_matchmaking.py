from src.matchmaking.matchmaking import Matchmaking


class SingleMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        return [(0, 0), ]
