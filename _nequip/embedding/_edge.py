from typing import Union, Dict, Optional

import torch
import math

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin

Type = Dict[str, torch.Tensor]

def with_edge_vectors(data: Type, with_lengths: bool = True) -> Type:
    """
    edge_vectors, edge_lengthsを求める
    edge_cell_shiftを考慮したedge_vector
    """
    if with_lengths and "edge_lengths" not in data:
        data["edge_lengths"] = torch.linalg.norm(
                data["edge_vectors"], dim=-1
        )
    return data

@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor
    
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    
    return out * (x < 1.0)

class PolynominalCutoff(torch.nn.Module):
    _factor: float
    p: float
    
    def __init__(self, r_max: float, p: float = 6):
        super().__init__()
        assert p >= 2.0
        self.p = p
        self._factor = 1.0 / float(r_max)
        
    def forward(self, x):
        return _poly_cutoff(x, self._factor, self.p)

@compile_mode("script")
class SphericalHarmonicsEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    out_field: str
    
    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in = None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        ):
        super().__init__()
        self.out_field = out_field
        
        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
    
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh
        return data
    
class BesselBasis(torch.nn.Module):
    r_max: float
    prefactor: float
    
    def __init__(
        self,
        r_max,
        num_basis=8,
        trainable=True,
    ):
        super(BesselBasis, self).__init__()
        
        self.trainable = trainable
        self.num_basis = num_basis
        
        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max
        
        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))
    
@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str
    
    def __init__(
        self,
        basis=BesselBasis,
        cutoff = PolynominalCutoff,
        r_max: float = 4.0,
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in = None,
    ):
        super().__init__()
        self.basis = basis(r_max=r_max)
        self.cutoff = cutoff(r_max=r_max)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        edge_lengths = data[AtomicDataDict.EDGE_LENGTH_KEY]

        edge_lengths_embedded = (
            self.basis(edge_lengths) * self.cutoff(edge_lengths).unsqueeze(-1)
        )

        data[self.out_field] = edge_lengths_embedded
        return data