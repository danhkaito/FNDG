import torch
from torch import nn
from typing import *

class SOM(nn.Module):
    def __init__(self, size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.size = size
        self.alpha = torch.tensor(0.3)
        self.sigma = torch.tensor(max(size[:-1]) / 2.0)
        self.weights = torch.randn(self.size)
        self.map_indices = torch.ones(size[0], size[1]).nonzero().view(-1, 2)

    def pdist(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
        """
        Calculates distance of order `p` between `a` and `b`.
        """
        return (a-b).abs().pow(p).sum(-1).pow(1/p)

    def find_bmus(self, distances: torch.Tensor) -> torch.Tensor:
        """Find BMU for each batch in `distances`."""
        min_idxs = distances.argmin(-1)
        # Distances are flattened, so we need to transform 1d indices into 2d map locations
        return torch.stack([min_idxs / self.size[1], min_idxs % self.size[1]], dim=1)
    
    

