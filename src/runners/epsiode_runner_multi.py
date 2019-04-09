from runners import EpisodeRunner

import numpy as np


class EpisodeRunnerMulti(EpisodeRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = np.zeros(self.batch.scheme['reward']['vshape'])

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            observations = self.env.get_obs()
            obs_team_1 = observations[:self.args.n_agents_team1]
            obs_team_2 = observations[self.args.n_agents_team1:]

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs_team_1": [obs_team_1],
                "obs_team_2": [obs_team_2]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                              t_env=self.t_env,
                                              test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [
                    (terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        observations = self.env.get_obs()
        obs_team_1 = observations[:self.args.n_agents_team1]
        obs_team_2 = observations[self.args.n_agents_team1:]
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs_team_1": [obs_team_1],
            "obs_team_2": [obs_team_2]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                          t_env=self.t_env,
                                          test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in
                          set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon",
                                     self.mac.action_selector.epsilon,
                                     self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch

    def _log(self, returns, stats, prefix):
        for idx, rets in enumerate(returns):
            self.logger.log_stat(prefix + "return_mean" + str(idx),
                                 np.mean(rets), self.t_env)
            self.logger.log_stat(prefix + "return_std" + str(idx),
                                 np.std(rets), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], self.t_env)
        stats.clear()
