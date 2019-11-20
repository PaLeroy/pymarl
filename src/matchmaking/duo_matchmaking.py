from src.matchmaking.matchmaking import Matchmaking


class DuoMatchmaking(Matchmaking):

    def list_combat(self, agent_dict):
        return [(0, 1), ]
