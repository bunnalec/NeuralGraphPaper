import torch
import torch.nn as nn

# Copied from Mr. GPT himself :)
def rank_normalize(tensor):
    # Step 1: Sort the tensor and get ranks
    _, indices = torch.sort(tensor)
    ranks = torch.arange(1, len(tensor) + 1, dtype=torch.float32).to(tensor.device)

    # Step 2: Normalize the ranks
    normalized_ranks = ranks / len(tensor)

    # Use the indices to create a rank-normalized tensor
    rank_normalized_tensor = torch.zeros_like(tensor, dtype=torch.float32).to(tensor.device)

    rank_normalized_tensor[indices] = normalized_ranks

    return rank_normalized_tensor

# Also copied from Mr. GPT (and modified)
def expand_repeat(tensor, repeats, dim):
    if dim < 0:
        dim = tensor.dim() + dim + 1
    return tensor.unsqueeze(dim).repeat(*(1 for _ in range(dim)), repeats, *(1 for _ in range(tensor.dim() - dim)))

class ES_Linear(nn.Linear):
    def __init__(self, n_inps, n_outs, **kwargs):
        super().__init__(n_inps, n_outs, **kwargs)
        self.train_mode = True
        
    def generate_epsilons(self, batch_size, sigma=.1):
        assert batch_size % 2 == 0, "In train mode batch size must be even so every +ep can have a -ep"
        epsilon_w = sigma * torch.randn(batch_size//2, *self.weight.shape)
        self.epsilon_w = torch.cat([epsilon_w, -epsilon_w], axis=0).to(self.weight.device)
        
        # Generate new weights here so don't have to do it every evaluation
        self.new_w = expand_repeat(self.weight, batch_size, 0) + self.epsilon_w

        if self.bias is not None:
            epsilon_b = sigma * torch.randn(batch_size//2, *self.bias.shape)
            self.epsilon_b = torch.cat([epsilon_b, -epsilon_b], axis=0).to(self.weight.device)
            self.new_b = expand_repeat(self.bias, batch_size, 0) + self.epsilon_b

    def forward(self, x):
        if self.train_mode:
            assert hasattr(self, "epsilon_w"), "Never initialized epsilons but being used in train mode."
            assert x.shape[0] == self.epsilon_w.shape[0], f"Batch size was different than epsilon batch size: x shape was {x.shape} and epsilon shape was {self.epsilon_w.shape}"
            assert x.dim() >= 2, "Need at least 2 dimensions on input (batch, inps)"
            og_shape = x.shape

            # In case there's extra batch dims (which there are in NeuralGraphs):
            # Need to check if x dim or w dim is bigger
            if x.dim() == 2:
                x = x.reshape(x.shape[0], 1, x.shape[1])
                w = self.new_w
            else:
                w = self.new_w.reshape(self.new_w.shape[0], *(1 for _ in range(x.dim() - 3)), *self.new_w.shape[1:])
            
            # This one always works
            b = self.new_b.reshape(self.new_b.shape[0], *(1 for _ in range(x.dim() - 2)), *self.new_b.shape[1:]) if self.bias is not None else 0
            
            # Honestly no idea why torch insists on having weight transposed like this
            out = x @ torch.swapaxes(w, -1, -2) + b
            out = out.reshape(*og_shape[:-1], w.shape[-2])
            return out
        else:
            b = self.bias if self.bias is not None else 0
            return x @ self.weight.T + b

    def estimate_grads(self, losses, sigma=.1, normalize=False):
        # Just found out that they rank normalize.  Gonna try it
        if normalize:
            losses = rank_normalize(losses)

        # Estimate gradients
        self.weight.grad = (self.epsilon_w * losses.reshape(-1, 1, 1)).mean(0) / sigma**2
        if self.bias is not None:
            self.bias.grad = (self.epsilon_b * losses.reshape(-1, 1)).mean(0) / sigma**2


class ES_MLP(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential

    # Turn on train mode or eval mode
    def train_mode(self, flag):
        for layer in self.sequential:
            if hasattr(layer, "train_mode"):
                layer.train_mode = flag

    def generate_epsilons(self, batch_size, sigma=.1):
        for layer in self.sequential:
            if hasattr(layer, "generate_epsilons"):
                layer.generate_epsilons(batch_size, sigma=sigma)

    def forward(self, x):
        # Whatever we do its without explicit grad calcs
        with torch.no_grad():
            return self.sequential(x)
        
    def estimate_grads(self, losses, sigma=.1, normalize=False):
        for layer in self.sequential:
            if hasattr(layer, "estimate_grads"):
                layer.estimate_grads(losses, sigma=sigma, normalize=normalize)