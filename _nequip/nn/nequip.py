from ..embedding._graph_mixin import GraphModuleMixin
from ..embedding._edge import RadialBasisEdgeEncoding
from ..embedding._edge import SphericalHarmonicsEdgeAttrs
from ..embedding._one_hot import OneHotAtomEncoding
from ..embedding._get_edge_cell_shift import get_edge_cell_shift

from ._convnetlayer import ConvNet
from ._atomwise import AtomwiseLinear
from ._atomwise import AtomwiseReduce

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from typing import Dict, List
import numpy as np

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
            parity: int
                parityは反転操作を示す
                p = ±1 (+1: even parity, -1: odd parity)
                p = 1では関数の反転はない, p=-1のとき反転を表す
            num_layers: int
                convnet layerの数
            num_features: int
                feature_irreps_hiddenの各irrepがnum_features個になる.
                allegroでいうenv_embed_multiplicity
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
            feature_irreps_hidden: 
                ConvNet内でのnode featuresを表すirreps
    """
    
    def __init__(
        self,
        num_atom_types: int,
        r_max: float,
        lmax: int,
        parity: int,
        num_layers: int,
        num_features: int,
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
        self.parity = parity
        self.num_layers = num_layers
        self.num_features = num_features
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
        self.feature_irreps_hidden = o3.Irreps(
            [
                (self.num_features, (l, p))
                for p in ((1, -1) if self.parity == -1 else (1,))
                for l in range(self.lmax + 1)
            ]
        )
        self.conv_to_output_hidden_irreps = o3.Irreps([(max(1, num_features//2), (0, 1))])
        
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
            feature_irreps_hidden=self.feature_irreps_hidden,
            num_layers=self.num_layers,
            invariant_layers=self.invariant_layers,
            invariant_neurons=self.invariant_neurons,
            activation_function=self.activation_function,
            resnet=self.resnet,
        )
        self.irreps_after_convnet = self.convnetlayer.irreps_out["node_features"]
        self.conv_to_output_hidden = AtomwiseLinear(
            irreps_in=self.irreps_after_convnet,
            irreps_out=self.conv_to_output_hidden_irreps,
        )
        
        self.atomic_energy = AtomwiseLinear(
            irreps_in=self.conv_to_output_hidden_irreps,
            irreps_out=o3.Irreps("1x0e"),
        )
        
        self.total_energy = AtomwiseReduce()
    
    def forward(self,
                input_data: List[Dict[str, torch.Tensor]],
    ):
        """ポテンシャルエネルギーと力を予測する
        Parameters
        ----------
            pos: 原子の位置情報
            atom_types: 原子がどのtypeを持つか
            edge_index: 辺ベクトルの中心と行き先の情報を持つ
            cut_off: cutoffの大きさ
            cell: cellの情報
        Returns
        -------
            total_energy: torch.Tensor, shape: []
            atomic_force: torch.Tensor, shape: [num_atoms, 3]
        """
        num_data = len(input_data)
        assert num_data > 0
        pos_list = []
        edge_index_list = []
        edge_cell_shift_list = []
        atom_type_list = []
        atom_max_num = 0
        edge_index_mid_tensor = torch.tensor([])
        for data_idx in range(len(input_data)):
            num_atoms = input_data[data_idx]['pos'].shape[0]
            edge_cell_shift = get_edge_cell_shift(input_data[data_idx]['pos'],
                                                  input_data[data_idx]['edge_index'],
                                                  input_data[data_idx]['cut_off'],
                                                  input_data[data_idx]['cell'])
            edge_index_mid_tensor  = input_data[data_idx]['edge_index'] + atom_max_num
            pos_list.append(input_data[data_idx]['pos'])
            edge_index_list.append(edge_index_mid_tensor)
            edge_cell_shift_list.append(edge_cell_shift)
            atom_type_list.append(input_data[data_idx]['atom_types'])
            atom_max_num += num_atoms
            
        pos = torch.cat(pos_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_cell_shift = torch.cat(edge_cell_shift_list, dim=0)
        atom_types = torch.cat(atom_type_list, dim=0)
        
        pos.requires_grad_(True)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        edge_cell_shift = torch.cat((edge_cell_shift, -edge_cell_shift), dim=0)
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]] + edge_cell_shift

        node_attrs = self.one_hot_encoding(pos, atom_types)
        node_features = node_attrs
        edge_embedding = self.radial_encoding(edge_vectors)
        edge_attrs = self.spherical_encoding(edge_vectors) # ここまでがembedding
        node_features = self.convnetlayer(node_features, node_attrs, edge_index, edge_embedding, edge_attrs) # Message Passingでnode featuresをnum_layers分だけ更新
        node_features = self.conv_to_output_hidden(node_features)
        atomic_energy = self.atomic_energy(node_features)
        total_energy = self.total_energy(atomic_energy) # 全体のエネルギーを求めます
        
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
        
        data_slice_list = []
        for data_idx in range(num_data):
            x = torch.tensor(data_idx, device=pos.device)
            x = x.repeat(1, input_data[data_idx]['pos'].shape[0]).reshape(-1)
            data_slice_list.append(x)
        data_slice = torch.cat(data_slice_list, dim=0).reshape(-1)
        total_energy_per_data = scatter(atomic_energy, data_slice, dim=0).reshape(-1)
        outputs = {
            "total_energy": total_energy_per_data,
            "atomic_force": atomic_force,
        }

        return outputs
    
    def count_parameters(self):
        """Nequipモデルのパラメータ数をカウントする
        """
        return sum(p.numel() for p in self.parameters())
        
        
