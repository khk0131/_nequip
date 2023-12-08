import torch
from torch import fx
from opt_einsum_fx import optimize_einsums_full

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin

from typing import NamedTuple, Optional, List, Tuple, Union

class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float
    
@compile_mode("script")
class Linear(CodeGenMixin, torch.nn.Module):
    weight_numel: int
    internal_weights: bool
    shared_weights: bool
    
    def __init__(
        self, 
        irreps_in,
        irreps_out,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
    ):
        """
        biases: irreps_outのどのirrepにbiasを与えるか、scalarでないとダメ
        internal_weights: weightsがlearnableかどうか
        """
        super().__init__()
        
        assert path_normalization in ["element", "path"]
        
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        
        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]
            
        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]
        
        """
        同じirrepのmulの和を求める
        """
        def alpha(ins):
            x = sum(
                irreps_in[i.i_in if path_normalization == "element" else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if f_in is not None:
                x *= f_in
            return 1.0 if x == 0 else x
        
        instructions = [
            Instruction(i_in=ins.i_in, i_out=ins.i_out, path_shape=ins.path_shape, path_weight=alpha(ins)**(-0.5))
            for ins in instructions
        ]
        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir):
                raise ValueError(f"{ins.i_in} and {ins.i_out} do not have the same irrep")
            
        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        instructions += [
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]
        
        if shared_weights is False and internal_weights is None:
            internal_weights = False
        
        if shared_weights is None:
            shared_weights = True
            
        if internal_weights is None:
            internal_weights = True
            
        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        
        graphmod, self.weight_numel, self.bias_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            f_in,
            f_out,
            shared_weights=shared_weights,
        )
        self._codegen_register({"_compiled_main": graphmod})
        
        """
        デフォルトではshared_weights = Falseなので、internal_weights = Falseになる.
        仮にinternal_weight = Trueだとしたら、weight, biasをそれぞれlearnableに変更する
        
        register_bufferで_codegen_registerに引数を追加
        """
        
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(*((f_in, f_out) if f_in is not None else ()), self.weight_numel))
        else:
            self.register_buffer("weight", torch.Tensor())
            
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = torch.nn.Parameter(
                torch.zeros(*((f_out,) if f_out is not None else ()), self.bias_numel)
            )
        else:
            self.register_buffer("bias", torch.Tensor())
            
        
    def forward(self, features, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weighs = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            bias = self.bias
        return self._compiled_main(features, weight, bias)
    
    
def _codegen_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    f_in: Optional[int] = None,
    f_out: Optional[int] = None,
    shared_weights: bool = False,
    optimize_einsums: bool = True,
) -> Tuple[fx.GraphModule, int, int]:
    graph_out = fx.Graph()
    tracer_out = fx.proxy.GraphAppendingTracer(graph_out)
    
    x = fx.Proxy(graph_out.placeholder("x", torch.Tensor), tracer_out)
    ws = fx.Proxy(graph_out.placeholder("w", torch.Tensor), tracer_out)
    bs = fx.Proxy(graph_out.placeholder("b", torch.Tensor), tracer_out)
    
    
    if f_in is None:
        size = x.shape[:-1]
        outsize = size + (irreps_out.dim,)
    else:
        size = x.shape[:-2]
        outsize = size + (
            f_out,
            irreps_out.dim,
        )
        
    bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)
    """
    scalarであるirrepのdimをかき集める
    ins.i_in = -1はスカラーの印
    
    bias_numel: linearでirreps_outのscalarになるpathの本数を表す
    """
    
    if bias_numel > 0:
        if f_out is None:
            bs = bs.reshape(-1, bias_numel)
        else:
            bs = bs.reshape(-1, f_out, bias_numel)
            
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    
    if len(instructions) == 0 and bias_numel == 0:
        out = x.new_zeros(outsize)
        
        graph_out.output(out.node, torch.Tensor)
        return fx.GraphModule({}, graph_out, "linear_forward"), 0, 0
    
    if f_in is None:
        x = x.reshape(-1, irreps_in.dim)
    else:
        x = x.reshape(-1, f_in, irreps_in.dim)
    batch_out = x.shape[0]
    
    def prod(x):
        out = 1
        for a in x:
            out *= a
        return out
    
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1) # ins.i_in = -1はただscalarが何個あるか数える役割, weight_numelは何個のweightを持つかを示す
    if weight_numel > 0:
        ws = ws.reshape(-1, weight_numel) if f_in is None else ws.reshape(-1, f_in, f_out, weight_numel)
        
    if len(irreps_in) == 1:
        x_list = [x.reshape(batch_out, *(() if f_in is None else (f_in,)), irreps_in[0].mul, irreps_in[0].ir.dim)]
    else:
        x_list = [
            x.narrow(-1, i.start, mul_ir.dim).reshape(batch_out, *(() if f_in is None else (f_in,)), mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in.slices(), irreps_in)
        ]
    
    z = "" if shared_weights else "z"
    
    flat_weight_index = 0
    flat_bias_index = 0
    
    out_list = []
    for ins in instructions:
        mul_ir_out = irreps_out[ins.i_out]
        
        if ins.i_in == -1:
            b = bs.narrow(-1, flat_bias_index, prod(ins.path_shape))
            flat_bias_index += prod(ins.path_shape)
            out_list += [(ins.path_weight * b).reshape(1, *(() if f_out is None else (f_out,)), mul_ir_out.dim)]
        else:
            mul_ir_in = irreps_in[ins.i_in]
            
            if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                continue  # どっちかscalarだったらcontinue(上のif ins.i_in==-1 で同じins.i_in, ins.i_outを持っている)
            
            path_nweight = prod(ins.path_shape)
            if len(instructions) == 1:
                w = ws
            else:
                w = ws.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape((() if shared_weights else (-1,)) + (() if f_in is None else (f_in, f_out)) + ins.path_shape)
            flat_weight_index += path_nweight
            
            if f_in is None:
                ein_out = torch.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])
            else:
                ein_out = torch.einsum(f"{z}xyuw,zxui->zywi", w, x_list[ins.i_in])
            
            ein_out = ins.path_weight * ein_out
            
            out_list += [ein_out.reshape(batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim)] # mul_ir_out.dim = mul_ir * irreps_in[ins.i_in].dim
    
    """
    out_listの中で、同じi_outになるものの和を取る
    """    
    out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
            shape=(batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim),
            like=x,
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out) > 1:
        out = torch.cat(out, dim=-1)
    else:
        out = out[0]
        
    out = out.reshape(outsize)
    graph_out.output(out.node, torch.Tensor)
    
    graph_out.lint()
    
    # graphmod_out = fx.GraphModule({}, graph_out, "linear_forward")
    graphmod_out = fx.GraphModule({}, graph_out, "linear_forward")
    
    if optimize_einsums:
        # See _tensor_product/_codegen.py for notes
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, *(() if f_in is None else (f_in,)), irreps_in.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                f_in or 1,
                f_out or 1,
                weight_numel,
            ),
            torch.zeros(
                1 if shared_weights else batchdim,
                f_out or 1,
                bias_numel,
            ),
        )
        graphmod_out = optimize_einsums_full(graphmod_out, example_inputs)
    return graphmod_out, weight_numel, bias_numel
            
def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)

            