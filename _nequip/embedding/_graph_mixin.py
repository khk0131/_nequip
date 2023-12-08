from typing import Dict, Any, Sequence
from e3nn import o3
from nequip.data import AtomicDataDict

_SPECIAL_IRREPS = [None]

def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


class GraphModuleMixin:
    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        my_irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        irreps_out: Dict[str, Any] = {},
    ):
        """
        pos: o3.Irreps(1x1o)である必要性
        edge_index: [2, num_edge]
        """
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = _fix_irreps_dict(irreps_in)
        
        if AtomicDataDict.POSITIONS_KEY in irreps_in:
            if irreps_in[AtomicDataDict.POSITIONS_KEY] != o3.Irreps("1x1o"):
                raise ValueError(
                    f"Positions must have irreps 1o, got instead"
                )
        irreps_in[AtomicDataDict.POSITIONS_KEY] = o3.Irreps("1o")
        
        if AtomicDataDict.EDGE_INDEX_KEY in irreps_in:
            if irreps_in[AtomicDataDict.EDGE_INDEX_KEY] is not None:
                raise ValueError(
                    "Edge indexes must have irreps None"
                )
        irreps_in[AtomicDataDict.EDGE_INDEX_KEY] = None
        
        my_irreps_in = AtomicDataDict._fix_irreps_dict(my_irreps_in)
        
        irreps_out = AtomicDataDict._fix_irreps_dict(irreps_out)
        
        self.irreps_in = irreps_in
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out
        