import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma_multi import COMACriticMulti
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class COMALearnerMulti:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents_team1 = args.n_agents_team1
        self.n_agents_team2 = args.n_agents_team2
        self.n_agents = [args.n_agents_team1, args.n_agents_team2]

        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        scheme_team_1 = scheme.copy()
        scheme_team_1['obs'] = scheme_team_1['obs_team_1']
        scheme_team_2 = scheme.copy()
        scheme_team_2['obs'] = scheme_team_2['obs_team_2']

        self.critic = [
            COMACriticMulti(scheme_team_1, args, 1),
            COMACriticMulti(scheme_team_2, args, 2)
        ]

        self.target_critic = [
            copy.deepcopy(critic_) for critic_ in self.critic
        ]

        self.agent_params = [
            list(param_) for param_ in mac.parameters()
        ]
        self.critic_params = [
            list(critic_.parameters()) for critic_ in self.critic
        ]

        self.params = [
            agent_params_ + critic_params_
            for agent_params_, critic_params_ in
            zip(self.agent_params, self.critic_params)
        ]

        self.agent_optimiser = [
            RMSprop(params=agent_params_, lr=args.lr,
                    alpha=args.optim_alpha,
                    eps=args.optim_eps)
            for agent_params_ in self.agent_params
        ]
        self.critic_optimiser = [
            RMSprop(params=critic_params_,
                    lr=args.critic_lr,
                    alpha=args.optim_alpha,
                    eps=args.optim_eps)
            for critic_params_ in self.critic_params
        ]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        rewards = [rewards[:, :, :self.n_agents_team1],
                   rewards[:, :, self.n_agents_team1:]]

        actions = batch["actions"][:, :]
        actions_teams = [actions[:, :, :self.n_agents_team1],
                         actions[:, :, self.n_agents_team1:]]

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        avail_actions = batch["avail_actions"][:, :-1]
        avail_actions = [avail_actions[:, :, :self.n_agents_team1],
                         avail_actions[:, :, self.n_agents_team1:]]

        critic_mask = mask.clone()

        mask = [mask.repeat(1, 1, self.n_agents_team1).view(-1),
                mask.repeat(1, 1, self.n_agents_team2).view(-1)]

        q_vals, critic_train_stats = self._train_critic(batch, rewards,
                                                        terminated,
                                                        actions_teams,
                                                        avail_actions,
                                                        critic_mask, bs, max_t)

        actions = actions[:, :-1]
        actions = [actions[:, :, :self.n_agents_team1],
                   actions[:, :, self.n_agents_team1:]]

        mac_out = [[] for _ in rewards]
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs_array = self.mac.forward(batch, t=t)
            for idx, _ in enumerate(rewards):
                mac_out[idx].append(agent_outs_array[idx])

        mac_out = [
            th.stack(mac_out_, dim=1)
            for mac_out_ in mac_out
        ]  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        for idx, (mac_out_, mask_) in enumerate(zip(mac_out, mask)):
            mac_out[idx][avail_actions == 0] = 0
            mac_out[idx] = mac_out_ / mac_out_.sum(dim=-1, keepdim=True)
            mac_out[idx][avail_actions == 0] = 0

            # Calculated baseline
            q_vals[idx] = q_vals[idx].reshape(-1, self.n_actions)
            pi = mac_out_.view(-1, self.n_actions)
            baseline = (pi * q_vals[idx]).sum(-1).detach()

            # Calculate policy grad with mask
            q_taken = th.gather(q_vals[idx], dim=1,
                                index=actions[idx].reshape(-1, 1)).squeeze(1)
            pi_taken = th.gather(pi, dim=1,
                                 index=actions[idx].reshape(-1, 1)).squeeze(1)

            pi_taken[mask_ == 0] = 1.0
            log_pi_taken = th.log(pi_taken)

            advantages = (q_taken - baseline).detach()

            coma_loss = - ((
                                       advantages * log_pi_taken) * mask_).sum() / mask_.sum()

            # Optimise agents
            self.agent_optimiser[idx].zero_grad()
            coma_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params[idx],
                                                    self.args.grad_norm_clip)
            self.agent_optimiser[idx].step()

            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                ts_logged = len(
                    critic_train_stats["critic_loss_" + str(idx + 1)])
                for key in ["critic_loss_" + str(idx + 1),
                            "critic_grad_norm_" + str(idx + 1),
                            "td_error_abs_" + str(idx + 1),
                            "q_taken_mean_" + str(idx + 1),
                            "target_mean_" + str(idx + 1)]:
                    self.logger.log_stat(key,
                                         sum(critic_train_stats[
                                                 key]) / ts_logged,
                                         t_env)

                self.logger.log_stat("advantage_mean_" + str(idx + 1), (
                        advantages * mask_).sum().item() / mask_.sum().item(),
                                     t_env)
                self.logger.log_stat("coma_loss_" + str(idx + 1),
                                     coma_loss.item(), t_env)
                self.logger.log_stat("agent_grad_norm_" + str(idx + 1),
                                     grad_norm, t_env)
                self.logger.log_stat("pi_max_" + str(idx + 1),
                                     (pi.max(dim=1)[
                                          0] * mask_).sum().item() / mask_.sum().item(),
                                     t_env)
                self.log_stats_t = t_env

        if (self.critic_training_steps - self.last_target_update_step) \
                / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions,
                      mask, bs, max_t):
        # Optimise critic
        target_q_vals = []
        for idx, target_critic_ in enumerate(self.target_critic):
            target_q_vals.append(target_critic_(batch)[:, :])

        targets_taken = [
            th.gather(target_q_vals_, dim=3, index=actions_).squeeze(3)
            for target_q_vals_, actions_ in zip(target_q_vals, actions)
        ]

        # Calculate td-lambda targets
        targets = [
            build_td_lambda_targets(rewards_, terminated, mask,
                                    targets_taken_, n_agents_,
                                    self.args.gamma,
                                    self.args.td_lambda)
            for rewards_, targets_taken_, n_agents_
            in zip(rewards, targets_taken, self.n_agents)
        ]

        q_vals = [th.zeros_like(target_q_vals_)[:, :-1]
                  for target_q_vals_ in target_q_vals]

        running_log = {
            "critic_loss_1": [],
            "critic_loss_2": [],
            "critic_grad_norm_1": [],
            "critic_grad_norm_2": [],
            "td_error_abs_1": [],
            "td_error_abs_2": [],
            "target_mean_1": [],
            "target_mean_2": [],
            "q_taken_mean_1": [],
            "q_taken_mean_2": [],

        }
        for idx, (rewards_, n_agents_, critic_, targets_,
                  critic_optimiser_, critic_params_) in enumerate(
            zip(rewards, self.n_agents, self.critic, targets,
                self.critic_optimiser, self.critic_params)):

            for t in reversed(range(rewards_.size(1))):
                mask_t = mask[:, t].expand(-1, n_agents_)
                if mask_t.sum() == 0:
                    continue

                q_t = critic_(batch, t)

                q_vals[idx][:, t] = q_t.view(bs, n_agents_, self.n_actions)
                q_taken = th.gather(q_t, dim=3,
                                    index=actions[idx][:, t:t + 1]).squeeze(
                    3).squeeze(1)
                targets_t = targets_[:, t]

                td_error = (q_taken - targets_t.detach())

                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask_t

                # Normal L2 loss, take mean over actual data
                loss = (masked_td_error ** 2).sum() / mask_t.sum()
                critic_optimiser_.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(critic_params_,
                                                        self.args.grad_norm_clip)
                critic_optimiser_.step()
                self.critic_training_steps += 1

                running_log["critic_loss_" + str(idx + 1)].append(loss.item())
                running_log["critic_grad_norm_" + str(idx + 1)].append(
                    grad_norm)
                mask_elems = mask_t.sum().item()
                running_log["td_error_abs_" + str(idx + 1)].append(
                    (masked_td_error.abs().sum().item() / mask_elems))
                running_log["q_taken_mean_" + str(idx + 1)].append(
                    (q_taken * mask_t).sum().item() / mask_elems)
                running_log["target_mean_" + str(idx + 1)].append(
                    (targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        for target_critic_, critic_ in zip(self.target_critic, self.critic):
            target_critic_.load_state_dict(critic_.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        for target_critic_, critic_ in zip(self.target_critic, self.critic):
            critic_.cuda()
            target_critic_.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        for idx, critic_ in enumerate(self.critic):
            th.save(critic_.state_dict(),
                    "{}/critic".format(path) + str(idx) + ".th")
        for idx, agent_optimiser_ in enumerate(self.agent_optimiser):
            th.save(agent_optimiser_.state_dict(),
                    "{}/agent_opt".format(path) + str(idx) + ".th")
        for idx, critic_optimiser_ in enumerate(self.critic_optimiser):
            th.save(critic_optimiser_.state_dict(),
                    "{}/critic_opt".format(path) + str(idx) + ".th")

    def load_models(self, path):
        self.mac.load_models(path)
        for idx, critic_ in enumerate(self.critic):
            self.critic[idx].load_state_dict(
                th.load("{}/critic".format(path) + str(idx) + ".th",
                        map_location=lambda storage,
                                            loc: storage))

        # Not quite right but I don't want to save target networks
        for idx, critic_ in enumerate(self.critic):
            self.target_critic[idx].load_state_dict(critic_.state_dict())
            self.agent_optimiser[idx].load_state_dict(
                th.load("{}/agent_opt".format(path) + str(idx) + ".th",
                        map_location=lambda storage, loc: storage))
            self.critic_optimiser[idx].load_state_dict(
                th.load("{}/critic_opt".format(path) + str(idx) + ".th",
                        map_location=lambda storage, loc: storage))
