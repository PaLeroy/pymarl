import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.vmix import VMixer
import torch as th
from torch.optim import RMSprop

from src.modules.agents import RNNVAgent


class VLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        self.mixer = QMixer(args)

        self.v_mixer = VMixer(args)
        self.target_v_mixer = copy.deepcopy(self.v_mixer)

        input_v = self.mac._get_input_shape(scheme)
        self.v_agent = RNNVAgent(input_v,
                                 self.args)
        self.v_hidden_states = None
        self.target_v_agent = copy.deepcopy(self.v_agent)
        self.target_v_hidden_states = None

        self.params += list(self.mixer.parameters())
        self.params += list(self.v_mixer.parameters())
        self.params += list(self.v_agent.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr,
                                 alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3,
                                        index=actions).squeeze(
            3)  # Remove the last dim

        # Calculate estimated V-Values to update the Q
        v_agent_out = []
        self.v_init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            v_outs = self.v_forward(batch, t=t)
            v_agent_out.append(v_outs)
        v_agent_out_for_q = th.stack(v_agent_out[1:], dim=1)
        v_agent_out_for_v = th.stack(v_agent_out[:-1], dim=1)

        # Calculate target V-Values
        target_mv_agent_out = []
        self.target_v_init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            v_outs = self.target_v_forward(batch, t=t)
            target_mv_agent_out.append(v_outs)
        target_mv_agent_out = th.stack(target_mv_agent_out[1:], dim=1)

        # Max over target Q-Values
        chosen_action_qvals = self.mixer(chosen_action_qvals,
                                         batch["state"][:, :-1])
        v_agent_out_for_q = self.v_mixer(v_agent_out_for_q, batch["state"][:, 1:])
        v_agent_out_for_v = self.v_mixer(v_agent_out_for_v, batch["state"][:, :-1])
        target_mv_agent_out = self.target_v_mixer(target_mv_agent_out, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets_q = rewards + self.args.gamma * (
                1 - terminated) * v_agent_out_for_q

        # Calculte 1-step V
        targets_v = rewards + self.args.gamma * (
                1 - terminated) * target_mv_agent_out

        # Td-error
        td_error_q = (chosen_action_qvals - targets_q.detach())
        td_error_v = (v_agent_out_for_v - targets_v.detach())

        mask_q = mask.expand_as(td_error_q)
        mask_v = mask.expand_as(td_error_v)

        # 0-out the targets that came from padded data
        masked_td_error = td_error_q * mask_q
        masked_td_error_v = td_error_v * mask_v

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask_q.sum()
        loss_v = (masked_td_error_v ** 2).sum() / mask_v.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        loss_v.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params,
                                                self.args.grad_norm_clip)
        self.optimiser.step()

        if (
                episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_v", loss_v.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (
                    masked_td_error.abs().sum().item() / mask_elems),
                                 t_env)
            self.logger.log_stat("td_error_abs_v", (
                    masked_td_error_v.abs().sum().item() / mask_elems),
                                 t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (
                                         mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("q_target_mean",
                                 (targets_q * mask).sum().item() / (
                                         mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("v_target_mean",
                                 (targets_v * mask).sum().item() / (
                                         mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_v_agent.load_state_dict(self.v_agent.state_dict())
        self.target_v_mixer.load_state_dict(self.v_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.v_agent.cuda()
        self.target_v_agent.cuda()

        self.mixer.cuda()
        self.v_mixer.cuda()
        self.target_v_agent.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.v_agent.state_dict(), "{}/v_agent.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.v_mixer.state_dict(), "{}/v_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.v_agent.load_state_dict(th.load("{}/v_agent.th".format(path),
                                             map_location=lambda storage,
                                                                 loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                           map_location=lambda storage,
                                                               loc: storage))

        self.v_mixer.load_state_dict(th.load("{}/v_mixer.th".format(path),
                                             map_location=lambda storage,
                                                                 loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path),
                                               map_location=lambda storage,
                                                                   loc: storage))

    def v_init_hidden(self, batch_size):
        self.v_hidden_states = self.v_agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1)

    def target_v_init_hidden(self, batch_size):
        self.target_v_hidden_states = self.target_v_agent.init_hidden().unsqueeze(
            0).expand(
            batch_size, self.n_agents, -1)

    def v_forward(self, ep_batch, t):
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        agent_outs, self.v_hidden_states = self.v_agent(agent_inputs,
                                                        self.v_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def target_v_forward(self, ep_batch, t):
        agent_inputs = self.mac._build_inputs(ep_batch, t)
        agent_outs, self.target_v_hidden_states = self.target_v_agent(
            agent_inputs,
            self.target_v_hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
