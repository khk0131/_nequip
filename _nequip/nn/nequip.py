from ..embedding._graph_mixin import GraphModuleMixin
from ..embedding._edge import RadialBasisEdgeEncoding
from ..embedding._edge import SphericalHarmonicsEdgeAttrs
from ..embedding._one_hot import OneHotAtomEncoding
from nequip.data import AtomicDataDict

from ._convnetlayer import ConvNet
from ._atomwise import AtomwiseLinear
from ._atomwise import AtomwiseReduce

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

@compile_mode('script')
class Nequip(GraphModuleMixin, torch.nn.Module):
    use_sc: bool
    
    def __init__(
        self,
        num_atom_types: int,
        r_max: float,
        lmax: int,
        num_layers: int,
        vectorize: bool = False,
        irreps_in=None,
        irreps_out=None,
    ) -> None:
        super().__init__()
        
        self.num_types = num_atom_types
        self.r_max=r_max
        self.lmax = lmax
        self.num_layers = num_layers
        self.vectorize = vectorize
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        
        
        self.one_hot_encoding = OneHotAtomEncoding(
            num_types = self.num_types,
            irreps_in = self.irreps_in,
        )
        
        self.radial_encoding = RadialBasisEdgeEncoding(
            r_max=self.r_max,
            irreps_in = self.irreps_in,
        )
        
        self.spherical_encoding = SphericalHarmonicsEdgeAttrs(
            irreps_edge_sh = self.lmax,
            irreps_in = self.irreps_in,
        )

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=self.one_hot_encoding.irreps_out,
        )

        self._init_irreps(
            irreps_in=self.irreps_out,
            irreps_out=self.radial_encoding.irreps_out,
        )
        
        self._init_irreps(
            irreps_in=self.irreps_out,
            irreps_out=self.spherical_encoding.irreps_out,
        )
        
        self.convnetlayer = ConvNet(
            irreps_in=self.irreps_out,
            feature_irreps_hidden=self.irreps_out["edge_attrs"],
            num_layers=self.num_layers,
            resnet=True,
        )
        
        self.output_layer = AtomwiseLinear(irreps_in={"node_features": o3.Irreps.spherical_harmonics(self.lmax)})
        self.total_energy = AtomwiseReduce(irreps_in={"atomic_energy": o3.Irreps("1x0e")})
    
        
    def forward(self, data: AtomicDataDict.Type):
        pos = data["pos"]
        edge_index = data["edge_index"]
        cut_off = data["cut_off"]
        cell= data["cell"]
        edge_cell_shift = torch.zeros(
            edge_index.shape[1], 3, device=pos.device, dtype=pos.dtype
        )
        vector_i_to_j = pos[edge_index[1]] - pos[edge_index[0]]
        edge_cell_shift -= (vector_i_to_j > cut_off) * cell
        edge_cell_shift += (vector_i_to_j < -cut_off) * cell
        
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        edge_cell_shift = torch.cat((edge_cell_shift, -edge_cell_shift), dim=0)
        pos.requires_grad_(True)
        
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]] + edge_cell_shift
        
        data["edge_cell_shift"] = edge_cell_shift
        data["edge_vectors"] = edge_vectors
        data["edge_index"] = edge_index
        
        data = self.one_hot_encoding(data)
        data = self.radial_encoding(data)
        data = self.spherical_encoding(data)
        data = self.convnetlayer(data)
        
        data = self.output_layer(data)
        data = self.total_energy(data)
        
        total_energy = data["total_energy"]
        pos = data["pos"]

        force = torch.autograd.grad(
            [
                -1 * total_energy,
            ],
            [
                pos,
            ],
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
    
        return total_energy, force
    
    def count_parameters(self):
        """Nequipモデルのパラメータ数をカウントする
        """
        return sum(p.numel() for p in self.parameters())
        
        
