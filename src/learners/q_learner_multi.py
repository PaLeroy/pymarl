import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearnerMulti:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents_team1 = args.n_agents_team1
        self.n_agents_team2 = args.n_agents_team2
        self.mac = mac
        self.logger = logger
        self.params = [list(param) for param in mac.parameters()]

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = [
            RMSprop(params=param, lr=args.lr, alpha=args.optim_alpha,
                    eps=args.optim_eps) for param in self.params]

        # a little wasteful to deepcopy (e.g. duplicates action selector),
        # but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]

        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # TODO: automate that
        rewards = [rewards[:, :, :self.n_agents_team1],
                   rewards[:, :, self.n_agents_team1:]]
        actions = [actions[:, :, :self.n_agents_team1],
                   actions[:, :, self.n_agents_team1:]]
        avail_actions = [avail_actions[:, :, :self.n_agents_team1],
                         avail_actions[:, :, self.n_agents_team1:]]

        # Calculate estimated Q-Values
        mac_out = [[] for _ in rewards]

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs_array = self.mac.forward(batch, t=t)
            for idx, _ in enumerate(rewards):
                mac_out[idx].append(agent_outs_array[idx])

        mac_out = [
            th.stack(mac_out_team_, dim=1)
            for mac_out_team_ in mac_out
        ]  # concat over time

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = []
        for idx, mac_out_ in enumerate(mac_out):
            # Remove the last dim
            chosen_action_qvals.append(
                th.gather(mac_out_[:, :-1], dim=3, index=actions[idx])
                    .squeeze(3))

        # Calculate the Q-Values necessary for the target
        target_mac_out = [[] for _ in mac_out]

        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs_array = self.target_mac.forward(batch, t=t)
            for idx, _ in enumerate(rewards):
                target_mac_out[idx].append(target_agent_outs_array[idx])

        # We don't need the first timesteps Q-Value estimate
        # for calculating targets
        target_mac_out = [
            th.stack(target_mac_out_team_[1:], dim=1)
            for target_mac_out_team_ in target_mac_out
        ]

        # Mask out unavailable actions
        for idx, target_mac_out_ in enumerate(target_mac_out):
            target_mac_out[idx][avail_actions[idx][:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            for idx, mac_out_ in enumerate(mac_out):
                mac_out[idx][
                    avail_actions[idx] == 0
                    ] = -9999999

            cur_max_actions = [
                mac_out_[:, 1:].max(dim=3, keepdim=True)[1]
                for mac_out_ in mac_out
            ]

            target_max_qvals = [
                th.gather(target_mac_out_team_, 3,
                          cur_max_actions_team_).squeeze(3)
                for target_mac_out_team_, cur_max_actions_team_ in
                zip(target_mac_out, cur_max_actions)
            ]
        else:
            target_max_qvals = [
                target_mac_out_team_.max(dim=3)[0]
                for target_mac_out_team_ in target_mac_out
            ]

        # Mix
        # Todo
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals,
                                             batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals,
                                                 batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = [
            rewards_ + self.args.gamma * (1 - terminated) * target_max_qvals_
            for rewards_, target_max_qvals_ in
            zip(rewards, target_max_qvals)
        ]

        # Td-error

        td_error = [
            (chosen_action_qvals_ - targets_.detach())
            for chosen_action_qvals_, targets_ in
            zip(chosen_action_qvals, targets)
        ]

        mask = [
            mask.expand_as(td_error_)
            for idx, td_error_ in enumerate(td_error)
        ]

        # 0-out the targets that came from padded data
        masked_td_error = [
            td_error_ * mask_
            for td_error_, mask_ in zip(td_error, mask)
        ]

        # Normal L2 loss, take mean over actual data
        loss = [
            (masked_td_error_ ** 2).sum() / mask_.sum()
            for masked_td_error_, mask_ in zip(masked_td_error, mask)
        ]

        # Optimise
        grad_norm = []
        for idx, optimiser in enumerate(self.optimiser):
            optimiser.zero_grad()
            loss[idx].backward()
            grad_norm.append(th.nn.utils.clip_grad_norm_(self.params[idx],
                                                         self.args.grad_norm_clip))
            optimiser.step()

        if (
                episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for idx, loss_ in enumerate(loss):
                self.logger.log_stat("loss_" + str(idx), loss_.item(), t_env)

                self.logger.log_stat("grad_norm_" + str(idx), grad_norm[idx],
                                     t_env)

                mask_elems = mask[idx].sum().item()
                self.logger.log_stat("td_error_abs_" + str(idx), (
                        masked_td_error[idx].abs().sum().item() / mask_elems),
                                     t_env)
                self.logger.log_stat("q_taken_mean_" + str(idx),
                                     (chosen_action_qvals[idx] * mask[
                                         idx]).sum().item() / (
                                             mask_elems * self.args.n_agents),
                                     t_env)
                self.logger.log_stat("target_mean_" + str(idx),
                                     (targets[idx] * mask[
                                         idx]).sum().item() / (
                                             mask_elems * self.args.n_agents),
                                     t_env)
                self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                               map_location=lambda storage,
                                                                   loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path),
                                               map_location=lambda storage,
                                                                   loc: storage))
