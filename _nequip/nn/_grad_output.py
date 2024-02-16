from ..embedding._graph_mixin import GraphModuleMixin
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from typing import Dict

@compile_mode("unsupported")
class CalculateForce(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        pos = data["pos"]
        total_energy = data["total_energy"]
        
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
        
        data["partial_forces"] = partial_forces
        
        return data
    

@compile_mode("unsupported")
class PartialForceOutput(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        func: GraphModuleMixin,
    ):
        super().__init__()
        self.func = func
        
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={"partial_forces": Irreps("1o")},
            irreps_out=func.irreps_out,
        )
        self.irreps_out["partial_forces"] = Irreps("1o")
        self.irreps_out["forces"] = Irreps("1o")
        
        
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = data.copy()
        out_data = {}

        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            """Wrapper from pos to atomic energy"""
            nonlocal data, out_data
            data["pos"] = pos
            out_data = self.func(data)
            return out_data["atomic_energy"].squeeze(-1)

        pos = data["pos"]

        partial_forces = torch.autograd.functional.jacobian(
            func=wrapper,
            inputs=pos,
            create_graph=self.training,  # needed to allow gradients of this output during training
            vectorize=self.vectorize,
        )
        partial_forces = partial_forces.negative()
        # output is [n_at, n_at, 3]

        out_data["partial_forces"] = partial_forces
        out_data["forces"] = partial_forces.sum(dim=0)

        return out_data
        
        