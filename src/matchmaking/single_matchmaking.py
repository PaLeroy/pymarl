from src.matchmaking.matchmaking import Matchmaking


class SingleMatchmaking(Matchmaking):

    def list_combat(self, agent_dict):
        return [(0, 0), ]
