import torch

def sensing_matrix(m,n, seed=1):
    torch.manual_seed(seed)
    return torch.bernoulli(torch.full((m, n), 0.5))

# print(sensing_matrix(4,4))

