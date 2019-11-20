class Matchmaking:
    def __init__(self, agent_dict):
        for k, v in agent_dict.items():
            agent_dict[k]["elo"] = 0

    def list_combat(self, agent_dict):
        return NotImplementedError
