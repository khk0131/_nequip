from typing import Optional

import torch
from torch_runstats.scatter import scatter
from ..embedding._graph_mixin import GraphModuleMixin
# from ._linear import Linear
from nequip.data import AtomicDataDict

from e3nn.o3 import Linear

class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str="node_features",
        out_field: Optional[str]=None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]
        
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field],
        )
        
    def forward(self, data: dict[str, torch.Tensor]) ->  dict[str, torch.Tensor]:
        data[self.out_field] = self.linear(data[self.field])
        return data
    

class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float
    
    def __init__(
        self, 
        field: str="atomic_energy",
        out_field: str="total_energy",
        reduce="sum",
        irreps_in=None,
    ):
        super().__init__()
        self.reduce = reduce
        self.field = field
        self.out_field = out_field
        
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )
        
    def forward(self, data:  dict[str, torch.Tensor]) ->  dict[str, torch.Tensor]:
        data = AtomicDataDict.with_batch(data)
        data[self.out_field] = scatter(
            data[self.field], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        
        return data