import torch
import torch.nn.functional

from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from ._graph_mixin import GraphModuleMixin

from typing import Dict, Any


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """原子のタイプのonehotを作るクラス
    """  
    num_types: int
    
    def __init__(
        self,
        num_types: int,
        irreps_in=None,
    ):
        """
        Parameters
        ----------
            num_types : int
                原子のタイプの種類数
            set_features: bool
                node_featuresを定めるか
        """
        super().__init__()
        self.num_types = num_types
        
        irreps_out = {"node_attrs": Irreps([(self.num_types, (0, 1))])}
        irreps_out["node_features"] = irreps_out["node_attrs"]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        
    def forward(self, pos: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            data: Dict[str, torch.Tensor]
                atom_types, posなどの情報を格納
            atom_types: torch.Tensor, shape:[num_atoms,]
                原子のタイプ
            pos: torch.Tensor, shape:[num_atoms, 3]
                原子の座標
        """
        type_numbers = atom_types.squeeze(-1)
        node_attrs = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=pos.dtype)
        # data["node_attrs"] = one_hot
        # if self.set_features:
        #     data["node_features"] = one_hot
        # node_features = node_attrs
        return node_attrs



