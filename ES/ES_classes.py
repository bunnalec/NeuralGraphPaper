import torch
import torch.nn as nn

class ES_Linear(nn.Linear):
    def __init__(self, n_inps, n_outs, sigma=.1, **kwargs):
        super().__init__(n_inps, n_outs, **kwargs)
        self.sigma = sigma
        self.train_mode = True
        
    def generate_epsilons(self, batch_size):
        assert batch_size % 2 == 0, "In train mode batch size must be even so every +ep can have a -ep"
        epsilon_w = torch.randn(batch_size//2, *self.weight.shape)
        self.epsilon_w = torch.cat([epsilon_w, -epsilon_w], axis=0).to(self.weight.device)

        epsilon_b = torch.randn(batch_size//2, *self.bias.shape)
        self.epsilon_b = torch.cat([epsilon_b, -epsilon_b], axis=0).to(self.weight.device)

    def forward(self, x):
        if self.train_mode:
            assert x.shape[0] == self.epsilon_w.shape[0], "Batch size was different than epsilon batch size"
            # Factoring:
            # x @ (w.T + ep1) + (b + ep2)
            # To:
            # x @ w.T + x @ ep1 + b + ep2
            # So I don't have to repeat weight which is sorta wack
            if x.dim() == self.epsilon_w.dim():
                noise_w = x @ torch.swapaxes(self.sigma * self.epsilon_w, -1, -2)
            elif x.dim() == self.epsilon_w.dim() - 1:
                noise_w = (x.unsqueeze(-2) @ torch.swapaxes(self.sigma * self.epsilon_w, -1, -2)).squeeze(-2)
            else:
                raise Exception("Not implemented more than 2 batch dims")

            out = x @ self.weight.T + noise_w + self.bias + (self.sigma * self.epsilon_b).reshape(self.epsilon_b.shape[0], *((x.dim()-2)*[1]), self.epsilon_b.shape[-1])
            return out
        else:
            return x @ self.weight.T + self.bias

    def estimate_grads(self, losses):
        # Estimate gradients
        self.weight.grad = (self.epsilon_w * losses.reshape(-1, 1, 1)).mean(0) / self.sigma
        self.bias.grad = (self.epsilon_b * losses.reshape(-1, 1)).mean(0) / self.sigma


class ES_MLP(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential

    # Turn on train mode or eval mode
    def train_mode(self, flag):
        for layer in self.sequential:
            if isinstance(layer, ES_Linear):
                layer.train_mode = flag

    def generate_epsilons(self, batch_size):
        for layer in self.sequential:
            if isinstance(layer, ES_Linear):
                layer.generate_epsilons(batch_size)

    def forward(self, x):
        # Whatever we do its without explicit grad calcs
        with torch.no_grad():
            return self.sequential(x)
        
    def estimate_grads(self, losses):
        for layer in self.sequential:
            if isinstance(layer, ES_Linear):
                layer.estimate_grads(losses)