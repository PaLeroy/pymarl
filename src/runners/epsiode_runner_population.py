from functools import partial

from runners import EpisodeRunner

import numpy as np
import torch as th

from src.components.episode_buffer import EpisodeBatch


class EpisodeRunnerPopulation(EpisodeRunner):
    def __init__(self, args, logger, agent_dict):
        super().__init__(args, logger)
        self.agent_dict = agent_dict
        self.mac_team1 = None
        self.mac_team2 = None
        self.t_total_team1 = None
        self.t_total_team2 = None
        self.batch_team_1 = None
        self.batch_team_2 = None
        self.t = None
        self.team_id1 = None
        self.team_id2 = None

        self.train_returns = {}
        self.test_returns = {}
        self.train_stats = {}
        self.test_stats = {}

        for k, _ in agent_dict.items():
            self.train_returns[k] = []
            self.test_returns[k] = []
            self.train_stats[k] = {}
            self.test_stats[k] = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)

    def setup_agents(self, list_pair, agent_dict):
        # To be called between each episode
        # Define which agents play with each other.
        # This will be a list of pair of agent_id
        self.team_id1 = list_pair[0][0]
        self.team_id2 = list_pair[0][1]
        self.mac_team1 = agent_dict[self.team_id1]["mac"]
        self.mac_team2 = agent_dict[self.team_id2]["mac"]
        self.t_total_team1 = agent_dict[self.team_id1]["t_total"]
        self.t_total_team2 = agent_dict[self.team_id2]["t_total"]

        # TODO: for parallel execution, need to handle t_total differently
        # Reload last information (in case, not usefuel atm)
        self.agent_dict = agent_dict

    def reset(self):
        self.batch_team_1 = self.new_batch()
        self.batch_team_2 = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()
        terminated = False
        episode_return = np.zeros(2)

        self.mac_team1.init_hidden(batch_size=self.batch_size)
        self.mac_team2.init_hidden(batch_size=self.batch_size)

        while not terminated:
            observations = self.env.get_obs()
            obs_team_1 = observations[:self.args.n_agents]
            obs_team_2 = observations[self.args.n_agents:]
            avail_actions = self.env.get_avail_actions()
            avail_actions_team_1 = avail_actions[:self.args.n_agents]
            avail_actions_team_2 = avail_actions[self.args.n_agents:]

            pre_transition_data_team_1 = {
                "state": [self.env.get_state()],
                "avail_actions": [avail_actions_team_1],
                "obs": [obs_team_1],
            }

            pre_transition_data_team_2 = {
                "state": [self.env.get_state()],
                "avail_actions": [avail_actions_team_2],
                "obs": [obs_team_2],
            }
            self.batch_team_1.update(pre_transition_data_team_1, ts=self.t)
            self.batch_team_2.update(pre_transition_data_team_2, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions_team_1 = self.mac_team1.select_actions(self.batch_team_1,
                                                           t_ep=self.t,
                                                           t_env=self.t_total_team1,
                                                           test_mode=test_mode)

            actions_team_2 = self.mac_team2.select_actions(self.batch_team_2,
                                                           t_ep=self.t,
                                                           t_env=self.t_total_team2,
                                                           test_mode=test_mode)
            actions = th.cat((actions_team_1[0], actions_team_2[0]))
            reward, terminated, env_info = self.env.step(actions)
            episode_return += (reward[0], reward[self.args.n_agents])
            post_transition_data_team_1 = {
                "actions": actions_team_1,
                "reward": [(reward[0],)],
                "terminated": [
                    (terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data_team_2 = {
                "actions": actions_team_2,
                "reward": [(reward[1],)],
                "terminated": [
                    (terminated != env_info.get("episode_limit", False),)],
            }

            self.batch_team_1.update(post_transition_data_team_1, ts=self.t)
            self.batch_team_2.update(post_transition_data_team_2, ts=self.t)

            self.t += 1

        last_observations = self.env.get_obs()
        obs_team_1 = last_observations[:self.args.n_agents]
        obs_team_2 = last_observations[self.args.n_agents:]

        last_avail_actions = self.env.get_avail_actions()
        avail_actions_team_1 = last_avail_actions[:self.args.n_agents]
        avail_actions_team_2 = last_avail_actions[self.args.n_agents:]

        last_data_team_1 = {
            "state": [self.env.get_state()],
            "avail_actions": [avail_actions_team_1],
            "obs": [obs_team_1],
        }
        last_data_team_2 = {
            "state": [self.env.get_state()],
            "avail_actions": [avail_actions_team_2],
            "obs": [obs_team_2],
        }
        self.batch_team_1.update(last_data_team_1, ts=self.t)
        self.batch_team_2.update(last_data_team_2, ts=self.t)

        # Select actions in the last stored state
        actions_team_1 = self.mac_team1.select_actions(self.batch_team_1,
                                                       t_ep=self.t,
                                                       t_env=self.t_total_team1,
                                                       test_mode=test_mode)

        actions_team_2 = self.mac_team2.select_actions(self.batch_team_2,
                                                       t_ep=self.t,
                                                       t_env=self.t_total_team2,
                                                       test_mode=test_mode)

        self.batch_team_1.update({"actions": actions_team_1}, ts=self.t)
        self.batch_team_2.update({"actions": actions_team_2}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns

        cur_stats[self.team_id1].update(
            {k: cur_stats[self.team_id1].get(k, 0) + env_info.get(k, 0) for k
             in
             set(cur_stats[self.team_id1]) | set(env_info)})
        cur_stats[self.team_id2].update(
            {k: cur_stats[self.team_id2].get(k, 0) + env_info.get(k, 0) for k
             in
             set(cur_stats[self.team_id2]) | set(env_info)})

        cur_stats[self.team_id1]["n_episodes"] \
            = 1 + cur_stats[self.team_id1].get("n_episodes", 0)
        cur_stats[self.team_id2]["n_episodes"] \
            = 1 + cur_stats[self.team_id2].get("n_episodes", 0)

        cur_stats[self.team_id1]["ep_length"] \
            = self.t + cur_stats[self.team_id1].get("ep_length", 0)
        cur_stats[self.team_id2]["ep_length"] \
            = self.t + cur_stats[self.team_id2].get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.t_total_team1 += self.t
            self.t_total_team2 += self.t

        cur_returns[self.team_id1].append(episode_return[0])
        cur_returns[self.team_id2].append(episode_return[1])
        log_prefix = "test_" if test_mode else ""

        for k, _ in self.agent_dict.items():
            id = k
            if test_mode and (len(
                    self.test_returns[k]) == self.args.test_nepisode):
                log_prefix_ = log_prefix + "agent_id_" + str(id) + "_"
                self._log(cur_returns[id], cur_stats[id], log_prefix_)

            elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                log_prefix_ = log_prefix + "agent_id_" + str(id) + "_"
                self._log(cur_returns[id], cur_stats[id], log_prefix_)
        if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            if hasattr(self.mac_team1.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon_id" + "agent_id_" + str(self.team_id1) + "_",
                    self.mac_team1.action_selector.epsilon,
                    self.t_env)
            if hasattr(self.mac_team2.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon_id" + "agent_id_" + str(self.team_id2) + "_",
                    self.mac_team1.action_selector.epsilon,
                    self.t_env)

        self.log_train_stats_t = self.t_env

        return [[self.batch_team_1, self.batch_team_2], ], \
               [[self.t_total_team1, self.t_total_team2], ]

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns),
                             self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns),
                             self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], self.t_env)
        stats.clear()
