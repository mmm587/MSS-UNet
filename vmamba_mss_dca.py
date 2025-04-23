import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

from .OSS import MamberBlock

import selective_scan_cuda_core as selective_scan_cuda

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
###########################################################################################################################################################
class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # input_t: float, fp16, bf16; weight_t: float;
        # u, B, C, delta: input_t
        # D, delta_bias: float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and (D.dtype != torch.float):
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and (delta_bias.dtype != torch.float):
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()
        
        assert u.shape[1] % (B.shape[1] * nrows) == 0 
        assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        
        _dD = None
        if D is not None:
            if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                _dD = dD.to(ctx._d_dtype)
            else:
                _dD = dD

        _ddelta_bias = None
        if delta_bias is not None:
            if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
            else:
                _ddelta_bias = ddelta_bias

        return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)


def selective_scan_fn_v1(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
###########################################################################################################################################################
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

# TODO 深度理解这个merging 的意义
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

# class PatchExpand2D(nn.Module):
#     def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim*2
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
#         self.norm = norm_layer(self.dim // dim_scale)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         x = self.expand(x)

#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
#         x= self.norm(x)

#         return x
    

# class Final_PatchExpand2D(nn.Module):
#     def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
#         self.norm = norm_layer(self.dim // dim_scale)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         x = self.expand(x)

#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
#         x= self.norm(x)

#         return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,#是模型的输入维度
        d_state=16,#是状态向量的维度（默认16）
        # d_state="auto", # 20240109
        d_conv=3,#是卷积核大小（默认3）
        expand=2,#控制了内部隐藏层的扩展维度
        dt_rank="auto",#是动态时间步长（dt）的秩，它决定了模型参数的数量
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}#存储了 device 和 dtype 用于参数初始化
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)#计算公式为 expand * d_model，这定义了模型内部的特征维度。
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        #接下来的部分定义了多个神经网络层，分别用于线性变换、卷积操作、激活函数等。
        #是一个线性变换，用于将输入 d_model 映射到 d_inner * 2 的维度。
        #d_model=(96,192,384,768)
        #d_inner=2*d_model
        #这里就是把d_model扩展到4倍
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        #是卷积层，in_channels 和 out_channels 都是 d_inner，
        #卷积操作的核大小是 d_conv，并且采用了 groups=self.d_inner 实现深度可分离卷积。
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # 是一个 SiLU 激活函数。
        self.act = nn.SiLU()
        #x_proj 是4个线性层的元组，用于对 d_inner 进行投影，
        # 并生成与 dt_rank 和 d_state 相关的矩阵。
        # 最后，self.x_proj_weight 将这些权重堆叠成一个张量，并删除了 x_proj 以节省内存。
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
        #dt_projs 是4个时间步长变换层的元组，使用了 self.dt_init 初始化。
        #self.dt_projs_weight 和 self.dt_projs_bias 分别保存了这些层的权重和偏置。
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        #A_logs 和 Ds 分别是 A 和 D 的初始化函数，用于设置模型中与动态时间步长相关的参数。
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0
        #标准化LayerNorm层
        self.out_norm = nn.LayerNorm(self.d_inner)
        #线性层Linear Layer
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        #如果 dropout > 0，则初始化 Dropout 层。
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    #初始化一个线形层，用于生成与dt_rank 和 d_inner 相关的时间步长变换矩阵
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        #接下来的部分是对初始化权重和偏置的处理，用于确保每个时间步长变换的偏置是非负的。
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj
    
    #初始化状态相关的A参数
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    #初始化时间跳跃相关的D参数
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
# 让我们逐步分析一个形状为 \( (B, H, W, C) \) 的张量 \( x \) 经过 `self.in_proj` 和 `self.out_proj` 这两个线性层后的形状变化。

# 1. **经过 `self.in_proj`**：
#    - 输入：\( x \) 的形状为 \( (B, H, W, C) \)。
#    - `self.in_proj` 是一个线性层，将 \( C \) 维的特征映射到 \( self.d_inner * 2 \) 维。
#    - 输出：\( xz \) 的形状变为 \( (B, H, W, self.d_inner * 2) \)。

# 2. **分割 `xz`**：
#    - `xz.chunk(2, dim=-1)` 将 \( xz \) 沿着最后一个维度（通道维度）分成两部分，每部分的形状为 \( (B, H, W, \frac{self.d_inner * 2}{2}) \) 或 \( (B, H, W, self.d_inner) \)。
#    - 因此，\( x \) 和 \( z \) 的形状都是 \( (B, H, W, self.d_inner) \)。

# 3. **对 \( x \) 进行维度变换**：
#    - `x.permute(0, 3, 1, 2)` 将 \( x \) 的形状从 \( (B, H, W, self.d_inner) \) 变换为 \( (B, self.d_inner, H, W) \)。
#    - 这个变换是为了适配卷积操作，将通道维度移到第二个位置。

# 4. **经过卷积和激活函数**：
#    - `x` 经过二维卷积和激活函数后，形状仍然是 \( (B, self.d_inner, H, W) \)。

# 5. **核心计算**：
#    - `y1, y2, y3, y4` 是 `forward_core` 方法的输出，假设它们的形状都是 \( (B, -1, L) \)，其中 \( L = H \times W \)。
#    - 将这四个输出相加，得到 \( y \) 的形状为 \( (B, -1, L) \)。

# 6. **维度变换和展平**：
#    - `torch.transpose(y, dim0=1, dim1=2)` 将 \( y \) 的形状从 \( (B, -1, L) \) 变换为 \( (B, L, -1) \)。
#    - `view(B, H, W, -1)` 将 \( y \) 的形状从 \( (B, L, -1) \) 展平为 \( (B, H, W, self.d_inner) \)。

# 7. **归一化和激活**：
#    - `self.out_norm(y)` 进行归一化，形状保持不变，为 \( (B, H, W, self.d_inner) \)。
#    - `y * F.silu(z)` 将 \( y \) 和 \( z \) 相乘，形状保持不变，为 \( (B, H, W, self.d_inner) \)。

# 8. **经过 `self.out_proj`**：
#    - `self.out_proj` 是一个线性层，将 \( self.d_inner \) 维的特征映射回 \( C \) 维。
#    - 输出：最终的输出形状为 \( (B, H, W, C) \)。

# 综上所述，经过 `self.in_proj` 和 `self.out_proj` 这两个线性层后，输入张量 \( x \) 的形状从 \( (B, H, W, C) \) 变为了 \( (B, H, W, C) \)，即最终输出的形状与输入形状相同。

    def forward(self, x: torch.Tensor, **kwargs):
        print(x.shape)
        B, H, W, C = x.shape# x=(B, H, W, C)
        #通过线性变换将输入x投影到新的表示空间  
        xz = self.in_proj(x)# xz=（B，H, W, 4C)
        #xz.chunk(2, dim=-1)：沿着最后一个维度（通道维度 C）将 xz 张量分成两部分
        #分别赋值给 x 和 z。x 和 z 都是形状为 (B, H, W, d) 的张量，其中 d = C // 2。
        x, z = xz.chunk(2, dim=-1) # (b, h, w, 2C)
        #x.permute(0, 3, 1, 2)：改变 x 的维度顺序，将其从 (B, H, W, d) 转换为 (B, d, H, W)，
        #也就是将通道维度移到第二个维度，适配卷积操作。
        x = x.permute(0, 3, 1, 2).contiguous()
        #对x进行二维卷积后再激活
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        #再对x经过核心计算
        y1, y2, y3, y4 = self.forward_core(x)#（B, -1, L)L=H*W
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)#归一化
        y = y * F.silu(z)#激活函数
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out#[B H W d]

#################################################################################################
class SS2D_1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        ssm_ratio=2,
        dt_rank="auto",        
        # ======================
        dropout=0.,
        conv_bias=True,
        bias=False,
        dtype=None,
        # ======================
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        shared_ssm=False,
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K * inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (K * D)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


        # channel scan settings
        self.KC = 2
        self.K2C = self.KC

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = nn.LayerNorm(self.d_inner)

        #cccc
        dc_inner = 4 
        self.dtc_rank = 6 #6
        self.dc_state = 16 #16
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)

        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) # (K, N, inner)
        del self.xc_proj
        #simple init
        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.K2C * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float() # (b, k, d, l)
        dts = dts.contiguous().float() # (b, k, d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1) # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1) #一个h,w展开，一个w,h展开，然后堆在一起
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
            return xs 
        
        #四个方向
        if self.K == 4:
            # K = 4
            xs = cross_scan_2d(x) # (b, k, d, l) #[batch_size, 4, channels, height * width]

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L) # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
            As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
            Ds = self.Ds # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            out_y = self.selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        # if this shows potential, we can raise the speed later by modifying selective_scan
        elif self.K == 1:
            x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
            # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
            dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
            dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
            x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1) # (b, -1, h, w)

            x_dt_BCs = cross_scan_2d(x_dt_BC) # (b, k, d, l)
            xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

            xs = xs.contiguous().view(B, -1, L) # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
            As = -torch.exp(self.A_logs.float()).repeat(4, 1) # (k * d, d_state)
            Ds = self.Ds.repeat(4) # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4) # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

            out_y = self.selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)
        
        return y

    forward_core = forward_corev1
    #forward_core = forward_corev0


    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn_v1

        xc = xc.permute(0, 3, 1, 2).contiguous()

        b,d,h,w = xc.shape
        
        #xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        #print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc) #b,d,1,1
        xc = xc.permute(0,2,1,3).contiguous() #b,1,d,1
        xc = self.conv_cin(xc) #b,4,d,1
        xc = xc.squeeze(-1) #b,4,d

        B, D, L = xc.shape #b,1,c
        D, N = self.Ac_logs.shape #2,16
        K, D, R = self.dtc_projs_weight.shape #2,1,6

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1) #input:b,d,l output:b,2,d,l
        #print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight) #8,2,1,96; 2,38,1 ->8,2,38,96
        
        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2) # 8,2,38,96-> 6,16,16
        #dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2 
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2

        out_y = self.selective_scanC(
            xsc, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()


        #y: b,4,d
        y = y.unsqueeze(-1) # b,4,d,1
        y = self.conv_cout(y) # b,1,d,1

        y = y.transpose(dim0=2, dim1=3).contiguous() # b,1,1,d
        y = self.channel_norm(y) 
        #before norm: input 222 shape torch.Size([8, 1, 1, 48])  ; output:333 shape torch.Size([8, 48, 1, 48]) 
        y = y.to(xc.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        #input x:b,c,h,w
        #x = x.permute(0, 2, 3, 1).contiguous() # b,h,w,c
        # print(x.shape)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, c)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z) # b,h,w,c

        #channel attention
        c = self.cforward_core(y)#x:b,h,w,d; output:b,1,1,d
        y2 = y * c
        y = y + y2

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out.permute(0, 3, 1, 2).contiguous()
        # print(out.shape)
        return out
#################################################################################################
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_1(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MamberBlock(
                dim=dim,
                num_heads=heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                #drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                #norm_layer=norm_layer,
                #attn_drop_rate=attn_drop,
                #d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


# class VSSLayer_up(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         depth (int): Number of blocks.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(
#         self, 
#         dim, 
#         depth, 
#         attn_drop=0.,
#         drop_path=0., 
#         norm_layer=nn.LayerNorm, 
#         upsample=None, 
#         use_checkpoint=False, 
#         d_state=16,
#         **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.use_checkpoint = use_checkpoint

#         self.blocks = nn.ModuleList([
#             VSSBlock(
#                 hidden_dim=dim,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer,
#                 attn_drop_rate=attn_drop,
#                 d_state=d_state,
#             )
#             for i in range(depth)])
        
#         if True: # is this really applied? Yes, but been overriden later in VSSM!
#             def _init_weights(module: nn.Module):
#                 for name, p in module.named_parameters():
#                     if name in ["out_proj.weight"]:
#                         p = p.clone().detach_() # fake init, just to keep the seed ....
#                         nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#             self.apply(_init_weights)

#         if upsample is not None:
#             self.upsample = upsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.upsample = None


#     def forward(self, x):
#         if self.upsample is not None:
#             x = self.upsample(x)
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         return x
    


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)#计算编码器的层数  num_layers=4
        if isinstance(dims, int):#如果dims是一个整数, 这里是一个数组所以忽略以下代码
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]#[96,192,384,768]
        self.embed_dim = dims[0]#嵌入维度=96
        self.num_features = dims[-1]#特征数=768
        self.dims = dims
        #创建一个patch_embed实例，将输入图像变成补丁嵌入
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding vit 里面有 ======================
        self.ape = False#设置绝对位置编码（APE）标志为False。
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:#如果启用APE，创建一个绝对位置编码参数，并使用截断正态分布进行初始化。这里不启用
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # 采用随机深度 drp rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule计算编码器的随即深度衰减率
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]#计算解码器的随即深度衰减率

        self.layers = nn.ModuleList()#创建一个模块列表，用于存储编码器层。
        for i_layer in range(self.num_layers): # 4 层
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,  # 0
                attn_drop=attn_drop_rate, # 0
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # self.layers_up = nn.ModuleList()原本VMamba中利用VSSBlock构建解码器
        # for i_layer in range(self.num_layers):
        #     layer = VSSLayer_up(
        #         dim=dims_decoder[i_layer],
        #         depth=depths_decoder[i_layer],
        #         d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
        #         drop=drop_rate, 
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         upsample=PatchExpand2D if (i_layer != 0) else None,
        #         use_checkpoint=use_checkpoint,
        #     )
        #     self.layers_up.append(layer)

        # self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        # self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x): # x [1, 3, 256, 256]
        skip_list = []
        x = self.patch_embed(x) # x [1, 64, 64, 96]  , dims=[96, 192, 384, 768]
        if self.ape: # absolute position embedding vit
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # 这里的drop_rate = 0
        # 返回最下面的 x feature 和 中间的四步的 feature
        for layer in self.layers: 
            skip_list.append(x) # x [1, 96, 64, 64]
            x = layer(x) # x encoder 的最终输出
        return x, skip_list # x [1, 8, 8, 768] , len(skip_list) = 4 , skip_list[0] [1, 64, 64, 96]
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up): # x [1, 8, 8, 768]
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])

        return x # [1, 64, 64, 96]
    
    def forward_final(self, x): # x [1, 64, 64, 96]
        x = self.final_up(x) # x [1, 256, 256, 24]
        x = x.permute(0,3,1,2) # x [1, 24, 256, 256]
        x = self.final_conv(x) # x [1, 1, 256, 256]
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward_bak(self, x):
        x, skip_list = self.forward_features(x)  # skip_list[0] [2, 64, 64, 96]   [3]  [2, 8, 8, 768]
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x
    # modified by sim  to sdi module
    def forward(self, x):
        x, skip_list = self.forward_features(x)  # skip_list[0] [2, 64, 64, 96]   [3]  [2, 8, 8, 768]        
        return skip_list[0], skip_list[1], skip_list[2], skip_list[3]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vmunet = VSSM()
    model = vmunet.to(device)
    x =torch.randn(1, 3, 256, 256)#B C H W
    x = x.to(device)
    out1 , out2 ,out3 ,out4 = model(x)  # 现在 model 和 x 都在 GPU 上
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)


    


