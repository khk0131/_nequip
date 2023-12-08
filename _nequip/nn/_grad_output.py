# from ..embedding._graph_mixin import GraphModuleMixin
from nequip.nn import GraphModuleMixin
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict


@compile_mode("unsupported")
class CalculateForce(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        pos = data[AtomicDataDict.POSITIONS_KEY]
        total_energy = data[AtomicDataDict.TOTAL_ENERGY_KEY]
        
        partial_forces = torch.autograd.grad(
            [
                -1 * total_energy,
            ],
            [
                pos,
            ],
            retain_graph=self.training,
            create_graph=self.training,
        )[0]

        partial_forces = partial_forces.negative() # -1をかける
        
        data[AtomicDataDict.PARTIAL_FORCE_KEY] = partial_forces
        
        return data
        
        