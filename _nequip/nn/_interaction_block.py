from typing import Dict

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from _nequip.embedding._graph_mixin import GraphModuleMixin

from ._non_linear import ShiftedSoftPlus

from ._linear import Linear

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
        
        irreps_mid = []
        instructions = []
        
        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
                        
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
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
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.sc = FullyConnectedTensorProduct(
            feature_irreps_in,
            self.irreps_in["node_attrs"],
            feature_irreps_out,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate interaction Block with ResNet (self-connection).

        :param node_input:
        :param node_attr: 各原子のone_hot
        :param edge_src: edgeの中心原子
        :param edge_dst: edgeを結ぶ隣接原子
        :param edge_attr: edge_vectorを球面調和関数に入れて投影したもの
        :param edge_embedded: rotation equivariantを達成するために、edge_lengthをbessel関数に入れてGNNにおける距離の情報を保つ
        :return:
        """
        weight = self.fc(data["edge_embedding"])

        x = data["node_features"]
        edge_src = data["edge_index"][1]
        edge_dst = data["edge_index"][0]

        if self.sc is not None:
            sc = self.sc(x, data["node_attrs"])

        x = self.linear_1(x) # self-interaction1 ?
        
        edge_features = self.tp(
            x[edge_src], data["edge_attrs"], weight
        )
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        x = self.linear_2(x) # self-interaction2 ?

        if self.sc is not None:
            x = x + sc # concatnation ?
        
        data["node_features"] = x
        return data