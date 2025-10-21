import torch
import torch.nn as nn
import torch.optim as optim

from config import PG_LEARNING_RATE, PG_GAMMA, PG_HIDDEN_LAYER_SIZE

# Neural network for the Policy Gradient agent.
class PolicyNetwork(nn.Module):

    def __init__(self, state_size, num_plants, hidden=PG_HIDDEN_LAYER_SIZE):
        super().__init__()
        # Shared layer processes the overall state of the garden
        self.shared = nn.Sequential(nn.Linear(state_size, hidden), nn.ReLU())
        # Each plant gets its own output layer (head) to predict action probabilities
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(num_plants)])

    def forward(self, x):
       
        x = self.shared(x)
        # Get action logits for each plant from its respective head
        return [h(x) for h in self.heads]
    
# a policy Gradient agent for the multi-plant garden environment.
class PolicyGradientAgent:

    def __init__(self, env, lr=PG_LEARNING_RATE, gamma=PG_GAMMA):
        self.env = env
        self.gamma = gamma
        self.model = PolicyNetwork(
            state_size=env.num_plants + 1, num_plants=env.num_plants)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        # memory to store log probabilities and rewards for the current episode
        self.log_probs, self.rewards = [], []

    def choose_action(self, state):

        s = torch.FloatTensor(state).unsqueeze(0)  # convert state to tensor
        logits = self.model(s)

        actions, log_probs_for_step = [], []
        for logit in logits:
            # create a probability distribution from the network's output logits
            prob_dist = torch.distributions.Categorical(logits=logit)
            # Sample an action from the distribution
            action = prob_dist.sample()
            actions.append(action.item())
            # store the log probability of the sampled action
            log_probs_for_step.append(prob_dist.log_prob(action))
        
        # Store the sum of log probabilities for this timestep
        self.log_probs.append(torch.stack(log_probs_for_step).sum())
        return actions

    def store_reward(self, r):
        #Stores the reward received at each step of an episode.
        self.rewards.append(r)

    def finish_episode(self):

        R, returns = 0, []
        #  calculate the discounted returns for each step in the episode
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)

        # Normalize returns which makes training more stable
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # calculate the loss
        loss = sum(-lp * G for lp, G in zip(self.log_probs, returns))

        # update the network weights
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # clear memory for the next episode
        self.log_probs.clear()
        self.rewards.clear()
