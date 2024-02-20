from ..embedding._graph_mixin import GraphModuleMixin
from ..embedding._edge import RadialBasisEdgeEncoding
from ..embedding._edge import SphericalHarmonicsEdgeAttrs
from ..embedding._one_hot import OneHotAtomEncoding
from ..embedding._get_edge_cell_shift import get_edge_cell_shift

from ._convnetlayer import ConvNet
from ._atomwise import AtomwiseLinear
from ._atomwise import AtomwiseReduce

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from typing import Dict

@compile_mode('script')
class Nequip(torch.nn.Module, GraphModuleMixin):
    """
        Nequip (https://www.nature.com/articles/s41467-022-29939-5)
        E(3)-同変グラフニューラルネットワークを用いた機械学習ポテンシャルを構築Allegroと比較して精度が良い場合もあるが、スケーラビリティに劣る
        
        Parameters
        ----------
            num_atom_types: int
                元素の種類数
            r_max: double
                cutoffの大きさ
            lmax: int
                球面調和関数でedge_vectorを投影する際のlの最大値
            num_layers: int
                convnet layerの数
            invariant_layers: int
                interaction layer内のfully connected layerの数を構築
            invariant_neurons: int
                interaction layer内のfully connecte layerの数を構築
            activation_function: str
                Interaction layer内のFully connected layerで使用する活性化関数
            bessel_num_basis: int
                原子iと原子jの距離をベッセル関数を用いてベクトルに変換する
                ときの値の個数
                大きいほど精度は良いが処理は遅い
                default: 8
            bessel_basis_trainable: bool
                ベッセル関数の重みを学習可能にさせるのかどうか
                default: True
            polynominal_p: float
                DimeNet(https://arxiv.org/abs/2003.03123)で提案された
                cutoff関数のハイパーパラメーター
                default: 6.0
            edge_sh_normalization: str
                球面調和関数の値のnormalize方法
                default: 'component'
            edge_sh_normalize: bool
                球面調和関数の値をnormalizeするかどうか
                default: True
            resnet: bool
                この論文(https://arxiv.org/abs/1512.03385)で提案された
                残差ブロックを飛ばしたものと残差ブロックを通したものの差で重みの更新を行うことによって、従来よりも精度の高い予測が可能になった
                default: True
            irreps_in: Dict[str, o3.Irreps]
                各処理前に持つirrepsを表す
            irreps_out: Dict[str, o3.Irreps]
                各処理後に持つirrepsを表す
    """
    
    def __init__(
        self,
        num_atom_types: int,
        r_max: float,
        lmax: int,
        num_layers: int,
        invariant_layers: int,
        invariant_neurons: int,
        activation_function: str = "silu",
        bessel_num_basis: int = 8,
        bessel_basis_trainable: bool = True,
        polynomial_p: float = 6.0,
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        resnet: bool = True,
        irreps_in = None,
        irreps_out = None,
    ) -> None:
        super().__init__()
        
        self.num_types = num_atom_types
        self.r_max=r_max
        self.lmax = lmax
        self.num_layers = num_layers
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.activation_function = activation_function
        self.bessel_num_basis = bessel_num_basis
        self.bessel_basis_trainable = bessel_basis_trainable
        self.polynomial_p = polynomial_p
        self.edge_sh_normalization = edge_sh_normalization
        self.edge_sh_normalize = edge_sh_normalize
        self.resnet = resnet
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        
        self.one_hot_encoding = OneHotAtomEncoding(
            num_types = self.num_types,
            irreps_in = self.irreps_in,
        )
        
        self.radial_encoding = RadialBasisEdgeEncoding(
            r_max=self.r_max,
            bessel_num_basis=self.bessel_num_basis,
            bessel_basis_trainable=self.bessel_basis_trainable,
            polynomial_p=self.polynomial_p,
            irreps_in = self.irreps_in,
        )
        
        self.spherical_encoding = SphericalHarmonicsEdgeAttrs(
            irreps_edge_sh = self.lmax,
            edge_sh_normalization = self.edge_sh_normalization,
            edge_sh_normalize = self.edge_sh_normalize,
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
            invariant_layers=self.invariant_layers,
            invariant_neurons=self.invariant_neurons,
            activation_function=self.activation_function,
            resnet=self.resnet,
        )
        
        self.conv_to_output_hidden = AtomwiseLinear(
            irreps_in={"node_features": o3.Irreps.spherical_harmonics(self.lmax)}
        )
        
        self.atomic_energy = AtomwiseLinear(
            irreps_in={"node_features": o3.Irreps.spherical_harmonics(self.lmax)}, 
            out_field="atomic_energy", 
            irreps_out=o3.Irreps("1x0e")
        )
        
        self.total_energy = AtomwiseReduce(
            irreps_in={"atomic_energy": o3.Irreps("1x0e")}, out_field="total_energy"
        )
    
    def forward(self, data: Dict[str, torch.Tensor]):
        """ポテンシャルエネルギーと力を予測する
        Parameters
        ----------
            data: dict{str: torch.Tensor}
                入力値としてpos, edge_index, cut_off, cellの情報を持つ
                ConvNetを通った後でのtotal_energy, atomic_forceなどの情報を格納していく
        Returns
        -------
            total_energy: torch.Tensor, shape: []
            atomic_force: torch.Tensor, shape: [num_atoms, 3]
        """
        pos = data["pos"]
        edge_index = data["edge_index"]
        cut_off = data["cut_off"]
        cell= data["cell"]
        pos.requires_grad_(True)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        edge_cell_shift = get_edge_cell_shift(pos, edge_index, cut_off, cell)
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]] + edge_cell_shift
        
        data["edge_index"] = edge_index
        data["edge_cell_shift"] = edge_cell_shift
        data["edge_vectors"] = edge_vectors

        data = self.one_hot_encoding(data)
        data = self.radial_encoding(data)
        data = self.spherical_encoding(data) # ここまでがembedding
        data = self.convnetlayer(data) # Message Passingでnode featuresをnum_layers分だけ更新
        data = self.conv_to_output_hidden(data)
        data = self.atomic_energy(data)
        data = self.total_energy(data) # 全体のエネルギーを求めます
        
        total_energy = data["total_energy"]
        pos = data["pos"]
        
        atomic_force = torch.autograd.grad(
            [
                -1 * total_energy,
            ],
            [
                pos,
            ],
            retain_graph=self.training,
            create_graph=self.training,
        )[0] # 各原子に働く力を求めます
    
        return total_energy, atomic_force
    
    def count_parameters(self):
        """Nequipモデルのパラメータ数をカウントする
        """
        return sum(p.numel() for p in self.parameters())
        
        
