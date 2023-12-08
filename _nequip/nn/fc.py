import math
from typing import List, Dict
import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn import Extract
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom

@compile_mode("script")
class _Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([]))

        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)

        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions]

        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def forward(self, x):
        return self.cut(x)

@compile_mode("script")
class MultiLayerPerceptron(torch.nn.Module):
    """num_mlp_layers層のMLP"""

    def __init__(
        self,
        irreps_scalars=None,
        irreps_gates=None,
        irreps_gated=None,
        activation_function=torch.nn.functional.silu,
    ):
        """
        Parameters
        ----------
            input_dim : int
                inputの数
            output_dim : int
                outputの数
            hidden_dims : List[int]
                隠れ層のノードの数
            activation_function: function
                default: torch.nn.functional.silu
                活性化関数
        """
        super().__init__()
        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_in = self.sc.irreps_in
        self.irreps_out = irreps_scalars+ irreps_gated
        input_dim = self.irreps_in.dim
        output_dim = self.irreps_out.dim
        self.activation_function = activation_function
        mlp_nodes = [input_dim] +  [output_dim]
        self.linears = torch.nn.ModuleList()
        for in_dim, out_dim in zip(mlp_nodes, mlp_nodes[1:]):
            self.linears.append(torch.nn.Linear(in_dim, out_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            x : torch.Tensor
                テンソル
        """
        for linear_idx, linear in enumerate(self.linears):
            x = linear(x)
            if linear_idx != len(self.linears) - 1:
                # 最後でなければ活性化関数を通す
                x = self.activation_function(x)

        return x


class MultiLayerPerceptronTorchGraph(CodeGenMixin, torch.nn.Module):
    """num_mlp_layers層のMLP"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation_function=torch.nn.functional.silu,
    ):
        """
        Parameters
        ----------
            input_dim : int
                inputの数
            output_dim : int
                outputの数
            hidden_dims : List[int]
                隠れ層のノードの数
            activation_function: function
                default: torch.nn.functional.silu
                活性化関数
        """
        super().__init__()
        nonlin_const = normalize2mom(activation_function).cst

        dimensions = [input_dim] + hidden_dims + [output_dim]
        num_layers = len(dimensions) - 1

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            w = torch.empty(h_in, h_out)
            w.normal_()
            b = torch.empty(h_out)
            b.normal_()

            params[f"_weight_{layer}"] = w
            params[f"_bias_{layer}"] = b
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            b = Proxy(graph.get_attr(f"_bias_{layer}"))
            w = w * (norm_from_last / math.sqrt(float(h_in)))
            b = b * (norm_from_last / math.sqrt(float(h_in)))
            features = torch.matmul(features, w) + b

            if layer < num_layers - 1:
                features = activation_function(features)
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)
