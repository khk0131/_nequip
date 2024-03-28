import torch
from torch import fx
import math

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin

@torch.jit.script
def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

@compile_mode("script")
class Extract(CodeGenMixin, torch.nn.Module):
    def __init__(self, irreps_in, irreps_outs, instructions):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions
        
        assert len(irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)
        
        graph = fx.Graph()
        x = fx.Proxy(graph.placeholder)
        

@compile_mode("script")
class _Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([])) # self.irreps_outsを結合してirreps_inに変換
        
        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)
        
        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions] # irreps_inの順番にinstructionsを並べ替える
        exit()

@compile_mode("script")
class Gate(torch.nn.Module):
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, acts_gates, irreps_gated):
        """
            irreps_scalars: activation functions(act_scalars)を通るscalars
            irreps_gates: activation functins(act_gates)を通るscalars
        """
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)
        
        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError("Gate scalars must be scalars")
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError("Scalars must be scalars")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError("The number of gates irreps are the same as the one of gated irreps")
        
        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)