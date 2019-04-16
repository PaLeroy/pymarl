import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACriticMulti(nn.Module):
    def __init__(self, scheme, args, team):
        super().__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.team = team
        if team == 1:
            self.n_agents = args.n_agents_team1
        else:
            self.n_agents = args.n_agents_team2

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs.append(
            batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        if self.team == 1:
            inputs.append(batch["obs_team_1"][:, ts])
            slice_ = slice(0, self.args.n_agents_team1)
            actions = batch["actions_onehot"][:, ts, slice_] \
                .view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        else:
            inputs.append(batch["obs_team_2"][:, ts])
            slice_ = slice(self.args.n_agents_team1,
                           self.args.n_agents_team1 + self.args.n_agents_team2)
            actions = batch["actions_onehot"][:, ts, slice_] \
                .view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)

        # actions (masked out by agent)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(
            self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        # last actions
        if t == 0:
            inputs.append(
                th.zeros_like(batch["actions_onehot"][:, 0:1, slice_]).view(bs,
                                                                            max_t,
                                                                            1,
                                                                            -1).repeat(
                    1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(
                batch["actions_onehot"][:, slice(t - 1, t), slice_].view(bs,
                                                                         max_t,
                                                                         1,
                                                                         -1).repeat(
                    1,
                    1,
                    self.n_agents,
                    1))
        else:
            last_actions = th.cat(
                [th.zeros_like(batch["actions_onehot"][:, 0:1, slice_]),
                 batch["actions_onehot"][:, :-1, slice_]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1,
                                                                      self.n_agents,
                                                                      1)
            inputs.append(last_actions)

        inputs.append(
            th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(
                0).expand(bs, max_t, -1, -1))

        inputs = th.cat(
            [x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.team == 1:
            input_shape += scheme["obs_team_1"]["vshape"]
        else:
            input_shape += scheme["obs_team_2"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][
                           0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape
