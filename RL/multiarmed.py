import torch

# There are n levers which will give reward when pulled.
# Each lever has its own (normal w/ rand mu and std) distribution of reward
# Task is to minimize regret
class MultiArmedBandit:
    def __init__(self, device="cpu", n=5):
        self.n = n # Number of levers
        self.device = device

    def initialize(self, batch_size):
        # self.mus = torch.arange(self.n).unsqueeze(0).repeat(batch_size, 1)
        # self.mus = self.mus[:, torch.randperm(2)]
        self.mus = torch.stack([torch.randperm(self.n) for _ in range(batch_size)], dim=0).to(self.device)
        # self.mus = torch.randn(batch_size, self.n).to(self.device)
        # self.stds = torch.randn(batch_size, self.n).abs().to(self.device)
        self.batch_size = batch_size
        # Why does torch insist on returning both max and argmax
        maxes, argmaxes = self.mus.max(-1)
        return maxes, argmaxes # Best expected reward per step

    def step(self, action):
        # action is int between 0 and n-1 to which lever is being pulled
        mus = self.mus[torch.arange(self.batch_size).to(self.device), action]
        # stds = self.stds[torch.arange(self.batch_size).to(self.device), action]
        
        return mus#  + stds * torch.randn(self.batch_size).to(self.device)