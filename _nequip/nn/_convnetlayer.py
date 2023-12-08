from typing import Dict, Callable
import torch
import logging

from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._interaction_block import InteractionBlock
from _nequip.embedding._graph_mixin import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus
from nequip.utils.tp_utils import tp_path_exists
from .fc import MultiLayerPerceptron

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}


class ConvNet(GraphModuleMixin, torch.nn.Module):
    """
    Args:

    """

    resnet: bool
    
    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 2,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()

        assert nonlinearity_type in ("gate", "norm")

        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList([])
        self.equivariant_nonlins = torch.nn.ModuleList([])
        self.resnets = []

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )
        
        for num_layer in range(num_layers):
            if num_layer > 0:
                self.irreps_in = self.irreps_out
                
            edge_attr_irreps = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
            irreps_layer_out_prev = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
            
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.feature_irreps_hidden
                    if ir.l == 0
                    and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
                ]
            )
            
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.feature_irreps_hidden
                    if ir.l > 0
                    and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
                ]
            )
            
            irreps_layer_out = (irreps_scalars + irreps_gated).simplify()
            ir = (
                "0e"
                if tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
            
            self.equivariant_nonlin = MultiLayerPerceptron(
                irreps_scalars = irreps_scalars,
                irreps_gates = irreps_gates,
                irreps_gated = irreps_gated,
            )
            conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()
            
            # if nonlinearity_type == "gate":
            #     ir = (
            #         "0e"
            #         if tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, "0e")
            #         else "0o"
            #     )
            #     irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
                
            #     print(irreps_scalars, irreps_gates)
            #     equivariant_nonlin = Gate(
            #         irreps_scalars=irreps_scalars,
            #         act_scalars=[
            #             acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
            #         ],
            #         irreps_gates=irreps_gates,
            #         act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
            #         irreps_gated=irreps_gated,
            #     )
                
            #     conv_irreps_out = equivariant_nonlin.irreps_in.simplify()
            #     print(1, conv_irreps_out)
                
            # else:
            #     conv_irreps_out = irreps_layer_out.simplify()
                
            #     equivariant_nonlin = NormActivation(
            #         irreps_in=conv_irreps_out,
            #         scalar_nonlinearity=acts[nonlinearity_scalars[1]],
            #         normalize=True,
            #         epsilon=1e-8,
            #         bias=True,
            #     )
                
            self.equivariant_nonlins.append(self.equivariant_nonlin)
            
            if irreps_layer_out == irreps_layer_out_prev and resnet:
                self.resnet = True
            else:
                self.resnet = False
            self.resnets.append(self.resnet)
            
            convolution_kwargs.pop("irreps_in", None)
            convolution_kwargs.pop("irreps_out", None)
            self.conv = convolution(
                irreps_in=self.irreps_in,
                irreps_out=conv_irreps_out,
                **convolution_kwargs,
            )
            self.convs.append(self.conv)
            self.irreps_out.update(self.conv.irreps_out)
            self.irreps_out[
                AtomicDataDict.NODE_FEATURES_KEY
            ] = self.equivariant_nonlin.irreps_out
            
    def forward(
        self,
        data: AtomicDataDict.Type
    ) -> AtomicDataDict.Type:
        for layer_num, (conv, equivariant_nonlin) in enumerate(zip(self.convs, self.equivariant_nonlins)):
            old_x = data[AtomicDataDict.NODE_FEATURES_KEY]

            data = conv(data)
            
            data[AtomicDataDict.NODE_FEATURES_KEY] = equivariant_nonlin(data[AtomicDataDict.NODE_FEATURES_KEY])
            
            if self.resnets[layer_num]:
                data[AtomicDataDict.NODE_FEATURES_KEY] = (
                    old_x + data[AtomicDataDict.NODE_FEATURES_KEY]
                )
                
        return data