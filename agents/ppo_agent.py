from agents.actor_critic import ActorCritic
import torch
from torch import nn
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Memory: 
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]

class PPO:
    def __init__(self, state_dim, action_dim, config, restore=False, ckpt=None, use_wandb=False):

        self.lr = config.lr
        self.betas = config.betas
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.K_epochs = config.K_epochs
        self.use_wandb = use_wandb

        # current policy
        self.policy = ActorCritic(state_dim, action_dim, config.action_std).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr, betas=config.betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(state_dim, action_dim, config.action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss().float()

    def select_action(self, state, memory):
        ''' 
        Agents' action
        '''
        state = torch.FloatTensor(state).to(device)
        return self.old_policy.act(state, memory).cpu().numpy().flatten()

    def update(self, memory):
        '''
        Updates the agent
        '''


        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio: p/q
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()

            # Actor loss using Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1.float(), surr2.float())

            # Critic loss: critic loss - entropy
            critic_loss = 0.5 * self.MSE_loss(rewards.float(), state_values.float()) - 0.01 * dist_entropy

            # Total loss
            loss = actor_loss + critic_loss

            # Visualize debug variables
            if self.use_wandb:
                wandb.log(
                    {'total_loss': loss.mean(),
                    'actor_loss': actor_loss.mean(),
                    'critic_loss': critic_loss.mean(),
                    'advantage_loss': advantages.mean(),
                    'dist_entropy': dist_entropy.mean()
                    }
                    )
            # Backward gradients
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())