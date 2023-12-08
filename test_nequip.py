from typing import Dict, Callable
import torch
import sys
sys.path.append('/nfshome18/khosono/nequip/nequip/nn')
import e3nn
from e3nn import o3
from _convnetlayer import ConvNetLayer
from _interaction_block import InteractionBlock

irreps_in = {"pos": o3.Irreps("1x1o"), "node_features": o3.Irreps.spherical_harmonics(2), "edge_attrs": o3.Irreps.spherical_harmonics(2), "edge_embedding": o3.Irreps("3x0e"), "node_attrs": o3.Irreps.spherical_harmonics(2)}
nequi_model = ConvNetLayer(
    irreps_in = irreps_in,
    feature_irreps_hidden = o3.Irreps.spherical_harmonics(2),
)

inter_block = InteractionBlock(
    irreps_in = irreps_in,
    irreps_out = o3.Irreps.spherical_harmonics(2),
)


inter = inter_block()