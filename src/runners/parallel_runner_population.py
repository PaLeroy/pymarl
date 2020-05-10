from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from runners import ParallelRunner


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py


class ParallelRunnerPopulation(ParallelRunner):
    def __init__(self, args, logger, agent_dict):
        super().__init__(args, logger)
        self.agent_dict = agent_dict
        self.mac = None
        self.agent_timer = None
        self.batches = None
        self.team_id = None

        self.t = None

        self.train_returns = {}
        self.test_returns = {}
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = {}

        for k, _ in agent_dict.items():
            self.train_returns[k] = []
            self.test_returns[k] = []
            self.train_stats[k] = {}
            self.test_stats[k] = {}
            self.log_train_stats_t[k] = -1000000

        # Log the first run


    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, 1,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)

    def setup_agents(self, list_match, agent_dict):
        # To be called between each episode
        # Define which agents play with each other.
        # This will be a list of pair of agent_id
        self.mac = []
        self.team_id = []
        self.agent_timer = {}
        self.agent_dict = agent_dict
        self.list_match = list_match

        for k, v in agent_dict.items():
            self.agent_timer[k] = v['t_total']

        heuristic_list = []
        for idx_match, match in enumerate(list_match):
            team_id1 = match[0]
            team_id2 = match[1]
            self.team_id.append([team_id1, team_id2])
            self.mac.append(
                [agent_dict[team_id1]["mac"], agent_dict[team_id2]["mac"]])

            heuristic_t1 = type(
                agent_dict[team_id1]["mac"]).__name__ == 'DoNothingMAC'
            heuristic_t2 = type(
                agent_dict[team_id2]["mac"]).__name__ == 'DoNothingMAC'
            heuristic_list.append([heuristic_t1, heuristic_t2])

        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("setup_heuristic", heuristic_list[idx]))



    def reset(self):
        self.batches = []
        for _ in self.list_match:
            self.batches.append([self.new_batch(), self.new_batch()])

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data_1 = []
        pre_transition_data_2 = []
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()

            state = data["state"]
            observations = data["obs"]
            obs_team_1 = observations[:self.args.n_agents]
            obs_team_2 = observations[self.args.n_agents:]
            avail_actions = data["avail_actions"]
            avail_actions_team_1 = avail_actions[:self.args.n_agents]
            avail_actions_team_2 = avail_actions[self.args.n_agents:]
            pre_transition_data_team_1 = {"state": [state[0]],
                                          "avail_actions": [
                                              avail_actions_team_1],
                                          "obs": [obs_team_1]}
            pre_transition_data_1.append(pre_transition_data_team_1)
            pre_transition_data_team_2 = {"state": [state[1]],
                                          "avail_actions": [
                                              avail_actions_team_2],
                                          "obs": [obs_team_2]}
            pre_transition_data_2.append(pre_transition_data_team_2)
        self.t = 0
        for idx, _ in enumerate(self.batches):
            self.batches[idx][0].update(pre_transition_data_1[idx], ts=self.t)
            self.batches[idx][1].update(pre_transition_data_2[idx], ts=self.t)

        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()
        all_terminated = False
        episode_returns = [np.zeros(2) for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        for idx_match, _ in enumerate(self.list_match):
            for idx_team in range(2):
                self.mac[idx_match][idx_team].init_hidden(batch_size=1)

        terminated = [False for _ in range(self.batch_size)]
        final_env_infos = [{} for _ in range(self.batch_size)]  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            actions = []
            cpu_actions = []
            for idx_match, match in enumerate(self.list_match):
                if terminated[idx_match]:
                    continue
                actions_match = []
                for idx_team in range(2):
                    action = self.mac[idx_match][idx_team].select_actions(
                        self.batches[idx_match][idx_team],
                        t_ep=self.t,
                        t_env=self.agent_timer[match[idx_team]],
                        test_mode=test_mode)
                    actions_match.append(action)

                actions.append(actions_match)
                cpu_action = np.concatenate(
                    [actions_match[0][0].to("cpu").numpy(),
                     actions_match[1][0].to("cpu").numpy()])
                cpu_actions.append(cpu_action)
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    # Only send the actions to the env if it hasn't terminated
                    parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data_1 = []
            post_transition_data_2 = []

            # Data for the next step we will insert in order to select an action
            pre_transition_data_1 = []
            pre_transition_data_2 = []

            # Receive data back for each unterminated env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if terminated[idx]:
                    post_transition_data_1.append([])
                    post_transition_data_2.append([])
                    pre_transition_data_1.append([])
                    pre_transition_data_2.append([])
                else:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    reward_team_1 = data["reward"][0]
                    reward_team_2 = data["reward"][-1]

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos[idx] = data["info"]
                        if not data["info"].get("episode_limit", False):
                            env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data_team_1 = {
                        "actions": actions[action_idx][0],
                        "reward": [(reward_team_1,)],
                        "terminated": [(env_terminated,)]}
                    post_transition_data_team_2 = {
                        "actions": actions[action_idx][1],
                        "reward": [(reward_team_2,)],
                        "terminated": [(env_terminated,)]}
                    post_transition_data_1.append(post_transition_data_team_1)
                    post_transition_data_2.append(post_transition_data_team_2)

                    action_idx += 1
                    episode_returns[idx] += [reward_team_1, reward_team_2]
                    episode_lengths[idx] += 1

                    if not test_mode:
                        self.agent_timer[self.team_id[idx][0]] += 1
                        self.agent_timer[self.team_id[idx][1]] += 1

                    # Data for the next timestep needed to select an action
                    state = data["state"]
                    observations = data["obs"]
                    obs_team_1 = observations[:self.args.n_agents]
                    obs_team_2 = observations[self.args.n_agents:]
                    avail_actions = data["avail_actions"]
                    avail_actions_team_1 = avail_actions[:self.args.n_agents]
                    avail_actions_team_2 = avail_actions[self.args.n_agents:]
                    pre_transition_data_team_1 = {"state": [state[0]],
                                                  "avail_actions": [
                                                      avail_actions_team_1],
                                                  "obs": [obs_team_1]}
                    pre_transition_data_1.append(pre_transition_data_team_1)
                    pre_transition_data_team_2 = {"state": [state[1]],
                                                  "avail_actions": [
                                                      avail_actions_team_2],
                                                  "obs": [obs_team_2]}
                    pre_transition_data_2.append(pre_transition_data_team_2)

            # Add post_transiton data into the batch
            for idx, _ in enumerate(self.batches):
                if not terminated[idx]:
                    self.batches[idx][0].update(post_transition_data_1[idx],
                                                ts=self.t)
                    self.batches[idx][1].update(post_transition_data_2[idx],
                                                ts=self.t)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            for idx, _ in enumerate(self.batches):
                if not terminated[idx]:
                    self.batches[idx][0].update(pre_transition_data_1[idx],
                                                ts=self.t)
                    self.batches[idx][1].update(pre_transition_data_2[idx],
                                                ts=self.t)

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # Pre analyze. Exclude information from bugged environments: env that
        list_win = []
        list_time = []
        list_canceled_match = []
        for idx, d in enumerate(final_env_infos):
            won_team_1 = d['battle_won_team_1']
            won_team_2 = d['battle_won_team_2']
            episode_length = episode_lengths[idx]
            if not won_team_1 and not won_team_2 and episode_length < self.episode_limit:
                self.agent_timer[self.list_match[idx][0]] -= episode_length
                self.agent_timer[self.list_match[idx][1]] -= episode_length
                list_win.append(None)
                list_time.append(None)
                self.batches[idx] = None
                list_canceled_match.append(True)
            else:
                list_win.append([won_team_1, won_team_2])
                list_time.append([episode_length, episode_length])
                list_canceled_match.append(False)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        log_prefix = "test_" if test_mode else ""
        for idx_match, match in enumerate(self.list_match):
            if list_canceled_match[idx_match]:
                continue
            team_id1 = match[0]
            team_id2 = match[1]
            env_info = final_env_infos[idx_match]
            env_info_team1 = {
                "battle_won_team_1": env_info["battle_won_team_1"],
                "return_team_1": episode_returns[idx_match][0]}
            env_info_team2 = {
                "battle_won_team_2": env_info["battle_won_team_2"],
                "return_team_2": episode_returns[idx_match][1]}
            del env_info["battle_won_team_1"]
            del env_info["battle_won_team_2"]
            cur_stats[team_id1].update(
                {k: cur_stats[team_id1].get(k, 0) + env_info.get(k,
                                                                 0) + env_info_team1.get(
                    k, 0) for
                 k
                 in
                 set(cur_stats[team_id1]) | set(env_info) | set(
                     env_info_team1)})

            cur_stats[team_id2].update(
                {k: cur_stats[team_id2].get(k, 0) + env_info.get(k,
                                                                 0) + env_info_team2.get(
                    k, 0) for
                 k
                 in
                 set(cur_stats[team_id2]) | set(env_info) | set(
                     env_info_team2)})
            if env_info_team1["battle_won_team_1"]:
                cur_stats[team_id1]["won"] \
                    = 1 + cur_stats[team_id1].get(
                    "won", 0)
                cur_stats[team_id2]["defeat"] \
                    = 1 + cur_stats[team_id2].get(
                    "defeat", 0)

            elif env_info_team2["battle_won_team_2"]:
                cur_stats[team_id2]["won"] \
                    = 1 + cur_stats[team_id2].get(
                    "won", 0)
                cur_stats[team_id1]["defeat"] \
                    = 1 + cur_stats[team_id1].get(
                    "defeat", 0)
            else:
                cur_stats[team_id1]["draw"] \
                    = 1 + cur_stats[team_id1].get(
                    "draw", 0)
                cur_stats[team_id2]["draw"] \
                    = 1 + cur_stats[team_id2].get(
                    "draw", 0)

            cur_stats[team_id1]["n_episodes"] \
                = 1 + cur_stats[team_id1].get(
                "n_episodes", 0)
            cur_stats[team_id2]["n_episodes"] \
                = 1 + cur_stats[team_id2].get(
                "n_episodes", 0)

            cur_stats[team_id1]["ep_length"] \
                = episode_lengths[idx_match] + cur_stats[team_id1].get(
                "ep_length", 0)
            cur_stats[team_id2]["ep_length"] \
                = episode_lengths[idx_match] + cur_stats[team_id2].get(
                "ep_length", 0)
            cur_returns[team_id1].append(episode_returns[idx_match][0])
            cur_returns[team_id2].append(episode_returns[idx_match][1])

        if test_mode:
            n_test_runs = max(1,
                              self.args.test_nepisode // self.batch_size) * self.batch_size
            n_tests_returns = 0
            for k, v in self.test_returns.items():
                n_tests_returns += len(v)
            if (n_tests_returns >= n_test_runs * 2):
                for k, _ in self.agent_dict.items():
                    id = k
                    time = self.agent_timer[id]
                    log_prefix_ = log_prefix + "agent_id_" + str(id) + "_"
                    self._log(cur_returns[id], cur_stats[id], log_prefix_, time)

        for id, _ in self.agent_dict.items():
            time = self.agent_timer[id]
            if time - self.log_train_stats_t[id] >= self.args.runner_log_interval:
                log_prefix_ = log_prefix + "agent_id_" + str(id) + "_"
                self._log(cur_returns[id], cur_stats[id], log_prefix_, time)

                if hasattr(self.agent_dict[id]["mac"], "action_selector") and \
                        hasattr(self.agent_dict[id]["mac"].action_selector,
                           "epsilon"):
                    self.logger.log_stat("agent_id_" + str(id) + "_epsilon",
                                         self.agent_dict[id][
                                             "mac"].action_selector.epsilon,
                                         time)
                self.log_train_stats_t[id] = time

        return self.batches, list_time, list_win

    def _log(self, returns, stats, prefix, time):

        if len(returns) > 0:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns),
                                 time)
            self.logger.log_stat(prefix + "return_std", np.std(returns),
                                 time)
            returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], time)
        stats.clear()