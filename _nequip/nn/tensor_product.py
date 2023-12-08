from math import sqrt
from typing import List, Optional, Union, Any, Callable, NamedTuple

import torch 
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from opt_einsum_fx import optimize_einsums_full

class Instrcution(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple

@compile_mode("script")
class TensorProduct(CodeGenMixin, torch.nn.Module):
    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = None,
        path_normalization: str = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        _specialized_code: Optional[bool] = None,
        compile_left_right: bool = True,
    ):
        super().__init__()
        
        if irrep_normalization is None:
            irrep_normalization = "component"
        
        if path_normalization is None:
            path_normalization = "element"
            
        assert irrep_normalization in ["component", "norm", "none"]
        
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out
        
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instrcution(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                }[connection_mode]
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]
        
        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        
        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
            
        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
            
        def num_elements(ins):
            return {
                "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
                "uvu": self.irreps_in2[ins.i_in2].mul,
            }[ins.connection_mode]
            
        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert ins.connection_mode in ["uvw", "uvu"]
            
            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1
                
            if path_normalization == "element":
                x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i) for i in instructions if i.i_out == ins.i_out)
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
            if path_normalization == "none":
                x = 1
            
            if x > 0.0:
                alpha /= x
                
            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight
            
            normalization_coefficients += [sqrt(alpha)]
            
        self.instructions = [
            Instrcution(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]
        
        self.i_in1_dim = self.irreps_in1.dim
        self.i_in2_dim = self.irreps_in2.dim
        
        if shared_weights is False and internal_weights is None:
            internal_weights = False
        
        if shared_weights is None:
            shared_weights = True
            
        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)
            
        if _specialized_code is None:
            self._specialized_code = _specialized_code
        
        assert shared_weights or not internal_weights
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
            
        if compile_left_right:
            graphmod_left_right = codegen_tensor_product_left_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                self._specialized_code,
            )
            assert graphmod_left_right is not None
            
        self._codegen_register({"_compiled_main_left_right":graphmod_left_right})
        
        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)
        
        if self.internal_weights and self.weight_numel > 0:
            assert shared_weights, "if internal_weights is True, then shared_weights should be True."
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
            
    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-1] == self.i_in1_dim, "Incorect last dimension"
        assert y.shape[-1] == self.i_in2_dim, "Incorect last dimension"
        
        real_weight = self.weight
        
        return self._compiled_main_left_right(x, y, real_weight)

def prod(x):
    out = 1
    for a in x:
        out *= a
    return out

def codegen_tensor_product_left_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instrcution],
    shared_weights: bool = False,
    specialized_code: bool = True,
)-> fx.GraphModule:
    graph = fx.Graph()
    
    tracer = fx.proxy.GraphAppendingTracer(graph)
    
    x1s = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder("x2", torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)
    
    empty = fx.Proxy(graph.call_function(torch.empty, ((),), dict(device="cpu")), tracer=tracer)
    if shared_weights:
        output_shape = torch.broadcast_tensors(empty.extend(x1s.shape[:-1]), empty.extend(x2s.shape[:-1]))[0].shape
    else:
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]), empty.extend(x2s.shape[:-1]), empty.expand(weights.shape[:-1])
        )[0].shape
    del empty
    
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    
    if len(instructions) == 0:
        outputs = x1s.new_zeros(output_shape + (irreps_out.dim,))
        graph.output(outputs.node, torch.Tensor)
        return fx.GraphModule({}, graph, "tp_forward")
    
    if shared_weights:
        x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(output_shape + (-1,))
    else:
        x1s, x2s, weights = (
            x1s.broadcast_to(output_shape + (-1,)),
            x2s.broadcast_to(output_shape + (-1,)),
            weights.broadcast_to(output_shape + (-1,)),
        )
        
    output_shape = output_shape + (irreps_out.dim,)
    
    x1s = x1s.reshape(-1, irreps_in1.dim)
    x2s = x2s.reshape(-1, irreps_in2.dim)
    
    batch_numel = x1s.shape[0]
    
    def prod(x):
        out = 1
        for a in x:
            out *= a
        return out
    
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    
    if weight_numel > 0:
        weights = weights.reshape(-1, weight_numel)
    del weight_numel
    
    if len(irreps_in1) == 1:
        x1_list = [x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
    else:
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim) for i, mul_ir in zip(irreps_in1.slices, irreps_in1)
        ]
        
    x2_list = []
    
    if len(irreps_in2) == 1:
        x1_list.append(x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim))
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))
            
    z = "" if shared_weights else "z"
    
    xx_dict = dict()
    flat_weight_index = 0
    outputs = []
    
    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]
        
        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        
        if mul_ir_in1.dim == 0 or mul_ir_in2.ir.dim == 0 or mul_ir_out.ir.l == 0:
            continue
        
        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]
        
        assert ins.connection_mode in ["uvw", "uvu"]
        
        if ins.has_weight:
            w = weights[:, flat_weight_index : flat_weight_index + prod(ins.path_shape)].reshape(
                (() if shared_weights else (-1,) + tuple(ins.path_shape))
            )
            flat_weight_index += prod(ins.path_shape)
            
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == "uu":
                xx_dict[key] = torch.einsum("zui,zuj->zuij", x1, x2)
            else:
                xx_dict[key] = torch.einsum("zui,zvj->zuvij", x1, x2)
        xx = xx_dict[key]
        del key
        
        w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        
        l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if ins.connection_mode == "uvw":
            assert ins.has_weight
            if specialized_code and l1l2l3 == (0, 0, 0):
                result = torch.einsum(
                    f"{z}uvw,zu,zv->zw", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                )
            elif specialized_code and mul_ir_in1.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zu,zvj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                    mul_ir_out.ir.dim
                )
            elif specialized_code and mul_ir_in2.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zui,zv->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                    mul_ir_out.ir.dim
                )
            elif specialized_code and mul_ir_out.ir.l == 0:
                result = torch.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
            else:
                result = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
        if ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(
                        f"{z}uv,zu,zv->zu", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zu,zvj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zv->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
            else:
                # not so useful operation because v is summed
                result = torch.einsum("ijk,zuvij->zuk", w3j, xx)
        
        result = ins.path_weight * result
        
        outputs += [result.reshape(batch_numel, mul_ir_out.dim)]
                
    outputs = [
        _sum_tensors(
            [out for ins, out in zip(instructions, outputs) if ins.i_out == i_out],
            shape=(batch_numel, mul_ir_out.dim),
            like=x1s,
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(outputs) > 1:
        outputs = torch.cat(outputs, dim=1)
    else:
        outputs = outputs[0]
        
    outputs = outputs.reshape(output_shape)
    
    graph.output(outputs.node, torch.Tensor)
    
    graph.lint()
    
    constants_root = torch.nn.Module()

    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")

    return graphmod

def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)



