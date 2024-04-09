from typing import Dict

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from _nequip.embedding._graph_mixin import GraphModuleMixin

from ._non_linear import ShiftedSoftPlus
# from ._linear import Linear
from e3nn.o3 import Linear

class InteractionBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self, 
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        activation_function: str="silu",
    ) -> None:
        super().__init__()
        
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={"node_features": irreps_out}
        )
        """
            あらかじめembeddingしておく必要がある
            FullyConnectedNetで活性化関数をsiluかsspで選べる
        """
        feature_irreps_in = self.irreps_in["node_features"]
        feature_irreps_out = self.irreps_out["node_features"]
        irreps_edge_attr = self.irreps_in["edge_attrs"]
        
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )
        
        irreps_after_tp = []
        instructions = []
        
        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_after_tp)
                        irreps_after_tp.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
                        
        irreps_after_tp = o3.Irreps(irreps_after_tp) # tensor productで出力後のirreps
        irreps_after_tp, p, _ = irreps_after_tp.sort()
        
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_after_tp,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc = FullyConnectedNet(
            [self.irreps_in["edge_embedding"].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[activation_function],
        )

        self.tp = tp

        self.linear_2 = Linear(
            irreps_in=irreps_after_tp.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.sc = FullyConnectedTensorProduct(
            feature_irreps_in,
            self.irreps_in["node_attrs"],
            feature_irreps_out,
        )

    def forward(
        self, 
        edge_embedding: torch.Tensor,
        node_attrs: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction Block with ResNet (self-connection).
        ResNetスタイルを使ってnode featuresを更新

        node_input:
            node_attr: 各原子のone_hot
            edge_neighbor: エッジの先のノード
            edge_center: エッジの元のノード
            edge_attr: edge_vectorを球面調和関数に入れて投影したもの
            edge_embedded: rotation equivariantを達成するために、edge_lengthをbessel関数に入れてGNNにおける距離の情報を保つ
        return:
            node_features: 各nodeに持つ特徴量
            
        """
        weight = self.fc(edge_embedding)

        edge_neighbor = edge_index[1]
        edge_center = edge_index[0]

        sc = self.sc(node_features, node_attrs) # self-interaction

        node_features = self.linear_1(node_features) # self-interaction1 
        
        edge_features = self.tp(
            node_features[edge_neighbor], edge_attrs, weight
        ) # convolution flter
        node_features = scatter(edge_features, edge_center, dim=0, dim_size=len(node_features)) # node_featuresがSO(3)を保つ

        node_features = self.linear_2(node_features) # self-interaction2 

        node_features = node_features + sc 
        
        return node_features