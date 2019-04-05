from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from controllers.basic_controller import BasicMAC


import datetime

class MultiMAC(BasicMAC):
    def __init__(self, scheme, groups, args):

        self.n_agents_team1 = args.n_agents_team1
        self.n_agents_team2 = args.n_agents_team2
        self.n_agents = [self.n_agents_team1, self.n_agents_team2]
        self.args = args

        input_shape_team_1, input_shape_team_2 = self._get_input_shape(scheme)

        self.agent_team1 = self._build_agents(input_shape_team_1)
        self.agent_team2 = self._build_agents(input_shape_team_2)
        self.agents = [self.agent_team1, self.agent_team2]

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None),
                       test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions_array = [
            ep_batch["avail_actions"][:, t_ep, :self.n_agents_team1],
            ep_batch["avail_actions"][:, t_ep, self.n_agents_team1:]]

        agent_outputs_array = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = []
        for agent_outputs, avail_actions in zip(agent_outputs_array,
                                                avail_actions_array):
            chosen_actions.append(
                self.action_selector.select_action(agent_outputs[bs],
                                                   avail_actions[bs],
                                                   t_env,
                                                   test_mode=test_mode))


        return th.cat(chosen_actions, 1)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs_array = self._build_inputs(ep_batch, t)
        avail_actions = [ep_batch["avail_actions"][:, t, :self.n_agents_team1],
                         ep_batch["avail_actions"][:, t, self.n_agents_team1:]]
        agent_outs_array = []
        for idx, (hidden_states, agent_inputs) in enumerate(
                zip(self.hidden_states, agent_inputs_array)):
            agent_outs, self.hidden_states[idx] \
                = self.agents[idx](agent_inputs, hidden_states)

            # Softmax the agent outputs if they're policy logits
            if self.agent_output_type == "pi_logits":
                # TODO: check here when logits
                print("TO CHECK")
                if getattr(self.args, "mask_before_softmax", True):
                    # Make the logits for unavailable actions very negative
                    # to minimise their affect on the softmax
                    reshaped_avail_actions = avail_actions[idx].reshape(
                        ep_batch.batch_size * self.n_agents[idx], -1)
                    agent_outs[reshaped_avail_actions == 0] = -1e10

                agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
                if not test_mode:
                    # Epsilon floor
                    epsilon_action_num = agent_outs.size(-1)
                    if getattr(self.args, "mask_before_softmax", True):
                        # With probability epsilon, we will pick an available action uniformly
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1,
                                                                        keepdim=True).float()

                    agent_outs = (
                            (1 - self.action_selector.epsilon) * agent_outs
                            +
                            th.ones_like(agent_outs)
                            * self.action_selector.epsilon / epsilon_action_num
                    )

                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0
            agent_outs_array.append(agent_outs)
        agent_outs_array = [
            agent_outs.view(ep_batch.batch_size, self.n_agents[idx], -1)
            for idx, agent_outs in enumerate(agent_outs_array)
        ]
        return agent_outs_array

    def init_hidden(self, batch_size):
        self.hidden_states = [
            agent.init_hidden().unsqueeze(0).expand(batch_size,
                                                    self.n_agents[idx],
                                                    -1) for idx, agent in
            enumerate(self.agents)]

    def parameters(self):
        return [agent.parameters() for agent in self.agents]

    def load_state(self, other_mac):
        for agent, other_mac_agent in zip(self.agents, other_mac.agents):
            agent.load_state_dict(other_mac_agent.state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()

    def save_models(self, path):
        for idx, agent in self.agents:
            th.save(agent.state_dict(),
                    "{}/agent_" + str(idx) + ".th".format(path))

    def load_models(self, path):
        for idx, _ in self.agents:
            self.agents[idx]. \
                load_state_dict(
                th.load("{}/agent_" + str(idx) + ".th".format(path),
                        map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        return agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes 2 teams of homogenous agents with flat observations.

        bs = batch.batch_size
        inputs_team_1 = []
        inputs_team_2 = []

        inputs_team_1.append(batch["obs_team_1"][:, t])  # b1av
        inputs_team_2.append(batch["obs_team_2"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs_team_1.append(th.zeros_like(
                    batch["actions_onehot"][:, t, :self.n_agents_team1]))
                inputs_team_2.append(th.zeros_like(
                    batch["actions_onehot"][:, t, :self.n_agents_team2]))
            else:
                inputs_team_1.append(
                    batch["actions_onehot"][:, t, :self.n_agents_team1])
                inputs_team_2.append(th.zeros_like(
                    batch["actions_onehot"][:, t, :self.n_agents_team2]))

        if self.args.obs_agent_id:
            inputs_team_1.append(
                th.eye(self.n_agents_team1, device=batch.device).unsqueeze(
                    0).expand(bs, -1, -1))
            inputs_team_2.append(
                th.eye(self.n_agents_team2, device=batch.device).unsqueeze(
                    0).expand(bs, -1, -1))

        inputs_team_1 = th.cat([x.reshape(bs * self.n_agents_team1, -1)
                                for x in inputs_team_1], dim=1)
        inputs_team_2 = th.cat([x.reshape(bs * self.n_agents_team2, -1)
                                for x in inputs_team_2], dim=1)

        return inputs_team_1, inputs_team_2

    def _get_input_shape(self, scheme):
        input_shape_team_1 = scheme["obs_team_1"]["vshape"]
        input_shape_team_2 = scheme["obs_team_2"]["vshape"]

        if self.args.obs_last_action:
            input_shape_team_1 += scheme["actions_onehot"]["vshape"][0]
            input_shape_team_2 += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape_team_1 += self.n_agents_team1
            input_shape_team_2 += self.n_agents_team2

        return input_shape_team_1, input_shape_team_2
