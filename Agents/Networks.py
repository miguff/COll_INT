import torch.nn as nn
import torch as T
import torch.nn.functional as F
device = T.device("cuda" if T.cuda.is_available() else "cpu")
from torch.distributions import Beta
import math


class CriticNetwork(nn.Module):
    def __init__(self, input_dims,
                 fc1_dims=64, 
                 fc2_dims=64, ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.critic = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc2_dims, 1)

        )

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.critic(state)
        x = x.squeeze(-1)
        return x
    

class ActorNetwork(nn.Module):
    def __init__(self, input_dims,
                 fc1_dims=64, 
                 fc2_dims=64, 
                 n_actions=2):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LeakyReLU(),
        )

        self.alpha_head = nn.Linear(self.fc2_dims, self.n_actions)
        self.beta_head = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = device
        self.to(self.device)


    def forward(self, state):
        x = self.actor(state)

        alpha = F.softplus(self.alpha_head(x)) + 1e-5
        beta = F.softplus(self.beta_head(x)) + 1e-5

        return alpha, beta
    
    def get_dist(self, state):
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        alpha, beta = self.forward(state)
        return Beta(alpha, beta)
    
    def act(self, state):
        dist = self.get_dist(state)
        action = dist.sample()

        action = 2 * action - 1

        logprob = dist.log_prob((action + 1) / 2) - math.log(2.0)
        logprob = logprob.sum(-1)
        #return action.squeeze(0).detach().numpy(), logprob.item()
        return action, logprob
    
    def evaluate_actions(self, states, actions):
        """
        Used during PPO update to compute logprobs and entropy for given batch.
        states: (batch, state_dim)
        actions: (batch, action_dim)
        """
        if not isinstance(states, T.Tensor):
            states = T.tensor(states, dtype=T.float32, device=device)
        if not isinstance(actions, T.Tensor):
            actions = T.tensor(actions, dtype=T.float32, device=device)

        dist = self.get_dist(states)  # Beta over each action dim

        y = (actions + 1) / 2

        logprobs = dist.log_prob(y).sum(-1) - math.log(2.0) * actions.shape[-1]  # (batch,)
        entropy = dist.entropy().sum(-1)           # (batch,)
        return logprobs, entropy



class CarEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, cars: T.tensor):
        """
        cars: (batch, num_cars, 6)
        returns: (batch, embed_dim)
        """
        B, N, Feature = cars.shape  # batch, num_cars, features(=6)
        x = cars.view(B * N, Feature)           # (B*N, 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))           # (B*N, embed_dim)
        x = x.view(B, N, -1)              # (B, N, embed_dim)

        # simple permutation-invariant pooling: mean over cars
        state_embedding = x.mean(dim=1)   # (B, embed_dim)
        return state_embedding


