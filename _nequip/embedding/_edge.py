from typing import Union, Dict

import torch
import math

from e3nn import o3
from e3nn.util.jit import compile_mode

from ._graph_mixin import GraphModuleMixin

def with_edge_lengths(edge_vectors: torch.Tensor) -> torch.Tensor:
    """
        edge_vectors, edge_lengthsを求める
        edge_cell_shiftを考慮したedge_vector
    """
    edge_lengths = torch.linalg.norm(
        edge_vectors, dim=-1
    )
    return edge_lengths

@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    """ 距離に依存したRBF(radial basis function)がcutoffで2回微分可能でないので包絡関数を定義して2回微分を可能にする
        Parameters
        ----------
            x: torch.Tensor, shape: [num_edges]
                各エッジの長さの情報
            factor: float
                cutoffで正規化する
            p: float
                RBFに対して用いる包絡関数で使用するパラメータ
    """
    x = x * factor
    
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    
    return out * (x < 1.0)

class PolynominalCutoff(torch.nn.Module):
    """距離に依存したRBFを定義するために必要な包絡関数を定義
    """
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
    """ edge vectorsを球面調和関数に投影することによってedge attrsを求める
        この投影により、SO(3): 回転に対して同変性を保ちながらMLPなどの処理が可能になる
    """
    out_field: str
    
    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in = None,
        out_field: str = "edge_attrs",
        ):
        """
        Parameters
        ----------
            irreps_edge_sh: int or str or o3.Irreps
                球面調和関数の最大のl(角振動数), 
                data[edge_attrs]のshapeは[num_edges, 0+3+5+...+2*lmax+1]となる
            edge_sh_normalization: str
                component: |Y(x)|.pow(2) = 2*l + 1 -> edge_attrs.shap[-1] = 2*l + 1にする
            edge_sh_normalize: bool
                球面調和関数にedge_vectorsを投影する前に正規化を行うかどうか
        """
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
    
    def forward(self, edge_vectors: torch.Tensor) -> torch.Tensor:
        """球面調和関数により、edge_vectorsを投影する
        球面調和関数の計算結果はedge_attrsに入る
        Parameters
        ----------
            data: Dict[str, torch.Tensor]
            edge_vectorsをdataの中に含まれる必要性がある.
            edge_vectors : torch.Tensor
                原子iから原子jへの方向ベクトル, shape: [num_edges, 3]
        """
        edge_attrs = self.sh(edge_vectors)
        return edge_attrs
    
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
        bessel_num_basis: int = 8,
        bessel_basis_trainable: bool = True,
        polynomial_p: float = 6.0,
        out_field: str = "edge_embedding",
        irreps_in = None,
    ):
        super().__init__()
        self.basis = basis(r_max=r_max, num_basis=bessel_num_basis, trainable=bessel_basis_trainable)
        self.cutoff = cutoff(r_max=r_max, p=polynomial_p)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )
        
    def forward(self, edge_vectors) -> torch.Tensor:
        """edgeの情報(edge_lengths)をベッセル関数に入れた結果edge_embeddingに入れる
        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            edge_lengths : torch.Tensor, shape:[num_edges]
                中心の原子からneighborの原子へのベクトルの大きさ
        """
        edge_lengths = with_edge_lengths(edge_vectors)
        edge_embedding = (
            self.basis(edge_lengths) * self.cutoff(edge_lengths).unsqueeze(-1)
        )
        return edge_embedding