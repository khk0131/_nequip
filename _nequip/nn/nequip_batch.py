from ..embedding._graph_mixin import GraphModuleMixin
from ..embedding._edge import RadialBasisEdgeEncoding
from ..embedding._edge import SphericalHarmonicsEdgeAttrs
from ..embedding._one_hot import OneHotAtomEncoding
from nequip.data import AtomicDataDict

from ._convnetlayer import ConvNet
from ._atomwise import AtomwiseLinear
from ._atomwise import AtomwiseReduce

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from typing import List, Dict

@compile_mode('script')
class NequipBatch(GraphModuleMixin, torch.nn.Module):
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
    
        
    def forward(self, frames: List[Dict[str, torch.Tensor]]):
        num_frames= len(frames)
        assert num_frames > 0
        num_atoms = frames[0]['pos'].shape[0]
        for frame_idx in range(num_frames):
            assert num_atoms == frames[frame_idx]['pos'].shape[0]
            
        pos_list = []
        edge_index_list = []
        edge_cell_shift_list = []
        atom_types_list = []
        for frame_idx in range(len(frames)):
            edge_cell_shift = torch.zeros(
                frames[frame_idx]['edge_index'].shape[1], 3,
                device=frames[frame_idx]['pos'].device,
                dtype=frames[frame_idx]['pos'].dtype,
            )
            vector_i_to_j = frames[frame_idx]['pos'][frames[frame_idx]['edge_index'][1]] - frames[frame_idx]['pos'][frames[frame_idx]['edge_index'][0]]
            edge_cell_shift -= (vector_i_to_j > frames[frame_idx]['cut_off']) * frames[frame_idx]['cell']
            edge_cell_shift += (-vector_i_to_j < -frames[frame_idx]['cut_off']) * frames[frame_idx]['cell']
            
            frames[frame_idx]['edge_index'] += frame_idx * num_atoms
            pos_list.append(frames[frame_idx]['pos'])
            edge_index_list.append(frames[frame_idx]['edge_index'])
            edge_cell_shift_list.append(edge_cell_shift)
            atom_types_list.append(frames[frame_idx]['atom_types'])
            
        pos = torch.cat(pos_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_cell_shift = torch.cat(edge_cell_shift_list, dim=0)
        atom_types = torch.cat(atom_types_list, dim=0)
        
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        edge_cell_shift = torch.cat((edge_cell_shift, -edge_cell_shift), dim=0)
        pos.requires_grad_(True)
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]] + edge_cell_shift
        
        data: AtomicDataDict.Type
        data['pos'] = pos
        data['egde_index'] = edge_index
        data["edge_cell_shift"] = edge_cell_shift
        data["edge_vectors"] = edge_vectors
        
        data = self.one_hot_encoding(data)
        data = self.radial_encoding(data)
        data = self.spherical_encoding(data)
        data = self.convnetlayer(data)
        
        data = self.output_layer(data)
        data = self.total_energy(data)
        
        atomic_energy = data["atomic_energy"]
        total_energy_all_batch = data["total_energy"]
        pos = data["pos"]

        force = torch.autograd.grad(
            [
                -1 * total_energy_all_batch,
            ],
            [
                pos,
            ],
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
        
        frames_slice = torch.arange(num_frames, device=pos.device).reshape(-1, 1).repeat(1, num_atoms).reshape(-1)
        total_energy = scatter(atomic_energy, frames_slice, dim=0, dim_size=num_frames).reshape(-1)
        
        force = force.reshape(num_frames, num_atoms, 3)
        atomic_energy = atomic_energy.reshape(num_frames, num_atoms)
    
        return total_energy, force