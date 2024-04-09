from typing import Optional

import torch
from torch_runstats.scatter import scatter
from ..embedding._graph_mixin import GraphModuleMixin
# from ._linear import Linear
from e3nn import o3
from e3nn.o3 import Linear

class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_out,
        )
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        node_features = self.linear(node_features)
        return node_features
    
class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float
    
    def __init__(
        self, 
        reduce="sum",
    ):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, atomic_energy:  torch.Tensor) -> torch.Tensor:
        index = torch.zeros(len(atomic_energy), dtype=torch.long, device=atomic_energy.device)
        total_energy = scatter(
            atomic_energy, index, dim=0, reduce=self.reduce
        )
        return total_energy