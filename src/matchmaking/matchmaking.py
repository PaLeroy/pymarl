import numpy as np

class Matchmaking:
    def __init__(self, agent_dict):
        for k, v in agent_dict.items():
            agent_dict[k]["elo"] = 1000

    def list_combat(self, agent_dict):
        return NotImplementedError

    def update_elo(self, agent_dict, list_episode_matches, win_list):
        for idx, match in enumerate(list_episode_matches):
            win_1 = win_list[idx][0]
            win_2 = win_list[idx][1]
            if win_1 is None or win_2 is None:
                # The match did not end
                continue

            id_team_1 = match[0]
            id_team_2 = match[1]
            elo_team_1 = agent_dict[id_team_1]["elo"]
            elo_team_2 = agent_dict[id_team_2]["elo"]
            q_1 = np.power(10, elo_team_1/400)
            q_2 = np.power(10, elo_team_2/400)
            q_t = q_1 + q_2
            e_1 = q_1 / q_t
            e_2 = q_2 / q_t
            if win_1:
                s_1 = 1
                s_2 = 0
            elif win_2:
                s_1 = 0
                s_2 = 1
            else:
                s_1 = 0.5
                s_2 = 0.5
            agent_dict[id_team_1]["elo"] = max(0, elo_team_1 + 32 * (s_1 - e_1))
            agent_dict[id_team_2]["elo"] = max(0, elo_team_2 + 32 * (s_2 - e_2))