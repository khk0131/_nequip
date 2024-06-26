from typing import Dict, Callable
import torch
import logging

from e3nn import o3
from e3nn.nn import Gate

from ._interaction_block import InteractionBlock
from _nequip.embedding._graph_mixin import GraphModuleMixin
from ._non_linear import ShiftedSoftPlus
from .exist_path import tp_path_exists

class ConvNet(GraphModuleMixin, torch.nn.Module):
    """


    """
    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=InteractionBlock,
        num_layers: int = 2,
        invariant_layers: int = 1,
        invariant_neurons: int = 8,
        activation_function: str = "silu",
        resnet: bool = False,
    ):
        """
            irreps_layer_out_prev: o3.Irreps
                convnet layerに入る前のirreps, node_featuresのirrepsを含む必要性がある.
            feature_irreps_hidden: o3.Irreps
                convet layer内でのnode_featuresの特徴量. 
                同変性NNPでl={0,1,2}, p={-1,1}のとき32x0o+3201+32x1o+32x1e+32x2o+32x2e+32x3o+32x3eとなる
                Nequipの論文内ではl={0,1}, p={1}のとき32x0e+32x1e
                
        """
        super().__init__()
        
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.num_layers = num_layers
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.activation_function = activation_function    
        self.convs = torch.nn.ModuleList([])
        self.equivariant_nonlins = torch.nn.ModuleList([])
        self.resnets = []
        act_scalar: Dict[int, Callable] = {
            1: ShiftedSoftPlus,
            -1: torch.tanh,
        }
        act_gate: Dict[int, Callable] = {
            1: ShiftedSoftPlus,
            -1: torch.abs,
        }
        SCALAR = o3.Irrep("0e")

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=["node_features"],
        )
        
        for num_layer in range(self.num_layers):
            if num_layer > 0:
                self.irreps_in = self.irreps_out
            edge_attr_irreps = self.irreps_in["edge_attrs"]
            irreps_layer_out_prev = self.irreps_in["node_features"] 
            
            irreps_scalars, irreps_gated = [], []
            for mul, ir in self.feature_irreps_hidden:
                if ir.l == 0 and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir):
                    irreps_scalars.append((mul, ir))
                if ir.l > 0 and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir):
                    irreps_gated.append((mul, ir))
                    
            irreps_scalars = o3.Irreps(irreps_scalars)
            irreps_gated = o3.Irreps(irreps_gated)
            irreps_layer_out = (irreps_scalars + irreps_gated).simplify()
            
            ir = o3.Irrep("0o")
            if tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, SCALAR):
                ir = SCALAR
                
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
            
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    act_scalar[ir.p] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[
                    act_gate[ir.p] for _, ir in irreps_gates
                ],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()
            self.equivariant_nonlin = equivariant_nonlin
            self.equivariant_nonlins.append(self.equivariant_nonlin)
            
            if irreps_layer_out == irreps_layer_out_prev and resnet:
                self.resnet = True
            else:
                self.resnet = False
            self.resnets.append(self.resnet) # resnetを使うか
            
            self.conv = convolution(
                irreps_in=self.irreps_in,
                irreps_out=conv_irreps_out,
                invariant_layers=self.invariant_layers,
                invariant_neurons=self.invariant_neurons,
                activation_function=self.activation_function,
            )
            self.convs.append(self.conv)
            self.irreps_out.update(self.conv.irreps_out)
            self.irreps_out["node_features"] = self.equivariant_nonlin.irreps_out # node_featuresを1つのconvnetで更新
            
    def forward(
        self,
        node_features: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_attrs: torch.Tensor,
    ) -> torch.Tensor: # node_features, conv(node_attrs, edge_index, edge_embedding, edge_attrs), resnet styleで更新
        for layer_num, (conv, equivariant_nonlin) in enumerate(zip(self.convs, self.equivariant_nonlins)):
            old_node_features = node_features # old_node_featuresを変更する
            node_features = conv(edge_embedding, node_attrs, node_features, edge_index, edge_attrs)
            node_features = equivariant_nonlin(node_features)
            
            if self.resnets[layer_num]:
                node_features = (
                    old_node_features + node_features
                )
                
        return node_features