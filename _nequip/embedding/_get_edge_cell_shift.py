import torch

def get_edge_cell_shift(pos: torch.Tensor,
                        edge_index: torch.Tensor,
                        cut_off: torch.Tensor,
                        cell: torch.Tensor):
    
    vector_i_to_j = pos[edge_index[1]] - pos[edge_index[0]]
    edge_cell_shift = torch.zeros(
        edge_index.shape[1], 3, device=pos.device, dtype=pos.dtype
    )
    edge_cell_shift -= (vector_i_to_j > cut_off) * cell
    edge_cell_shift += (vector_i_to_j < -cut_off) * cell
    
    return edge_cell_shift