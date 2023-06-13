import torch
from torch import nn
from torch.distributions import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        ).float()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).float()

        self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)    #(4, )

    def act(self, state, memory):       # state (1,24)
        action_mean = self.actor(state)                     # (1,4)
        cov_mat = torch.diag(self.action_var).to(device)    # (4,4)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()                              # (1,4)
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):      # state (4000, 24); action (4000, 4)
        state_value = self.critic(state)    # (4000, 1)

        # to calculate action score(logprobs) and distribution entropy
        action_mean = self.actor(state)                     # (4000,4)
        action_var = self.action_var.expand_as(action_mean) # (4000,4)
        cov_mat = torch.diag_embed(action_var).to(device)   # (4000,4,4)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
