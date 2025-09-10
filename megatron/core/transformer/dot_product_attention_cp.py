# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Optional

import einops
import torch
import xformers
from torch import Tensor
from xformers.ops.fmha import memory_efficient_attention_forward_requires_grad, memory_efficient_attention_backward
from xformers.ops.fmha.attn_bias import AttentionBias
from xformers.ops.fmha.common import AttentionOp

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.core.transformer.dot_product_attention import DotProductAttention
import megatron.core.parallel_state as mpu


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []


def get_bwd_op(fwd_op):
    if fwd_op is None:
        return None
    op_map = {
        xformers.ops.fmha.cutlass.FwOp: xformers.ops.fmha.cutlass.BwOp,
        xformers.ops.fmha.flash.FwOp: xformers.ops.fmha.flash.BwOp,
        xformers.ops.fmha.flash3.FwOp: xformers.ops.fmha.flash3.BwOp,
    }
    assert fwd_op in op_map
    return op_map[fwd_op]


def to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride):
    if cp_size == 1:
        zz_mask = attention_mask
    else:
        chunked = attention_mask.chunk(dim=3, chunks=cp_size * 2)
        zz_mask = [_x for _p in zip(chunked[:cp_size], reversed(chunked[cp_size:])) for _x in _p]
        zz_mask = torch.cat(zz_mask, dim=3)
    attn_bias = torch.zeros(zz_mask.shape, device=q.device, dtype=q.dtype)
    attn_bias.masked_fill_(zz_mask, float('-inf'))
    attn_bias = attn_bias.expand(-1, heads_k_stride * (nheads // nheads_k), -1, -1)
    return attn_bias


class AttnFuncWithCp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attention_mask,
        attention_dropout,
        softmax_scale,
        pg,
    ):
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        nheads = q.shape[2]
        nheads_k = k.shape[2]
        heads_k_stride = nheads_k
        assert nheads % nheads_k == 0 and nheads_k % heads_k_stride == 0
        outs = []
        lses = []

        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        attn_bias = to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride)

        for i in range(0, nheads_k, heads_k_stride):
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            q_i = q[:, :, i * nheads // nheads_k:(i + heads_k_stride) * nheads // nheads_k]
            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')

            out_i, lse_i = memory_efficient_attention_forward_requires_grad(
                q_i,
                k_i,
                v_i,
                attn_bias=attn_bias,
                p=attention_dropout,
                scale=softmax_scale,
                op=None,
                output_dtype=None,
            )
            outs.append(out_i)
            lses.append(lse_i)

        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        ctx.save_for_backward(q, k, v)
        ctx.outs = outs
        ctx.lses = lses
        ctx.attention_mask = attention_mask
        ctx.p = attention_dropout
        ctx.scale = softmax_scale
        ctx.op = None
        ctx.output_dtype = None
        ctx.heads_k_stride = heads_k_stride
        ctx.pg = pg

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        outs = ctx.outs
        lses = ctx.lses
        attention_mask = ctx.attention_mask
        p = ctx.p
        scale = ctx.scale
        op = get_bwd_op(ctx.op)
        output_dtype = ctx.output_dtype
        heads_k_stride = ctx.heads_k_stride
        pg = ctx.pg

        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        nheads = q.shape[2]
        nheads_k = k.shape[2]

        kv_buffer = torch.empty(
            (2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        dq = []
        dk = []
        dv = []
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        attn_bias = to_zz_mask_attn_bias(attention_mask, cp_size, q, nheads, nheads_k, heads_k_stride)

        for i in range(0, nheads_k, heads_k_stride):
            q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            if i < nheads_k - heads_k_stride:
                kvsl = i + heads_k_stride
                kvsr = kvsl + heads_k_stride
                send_k = k[:, :, kvsl:kvsr].contiguous()
                send_v = v[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            k_i = kv_buffer[0]
            v_i = kv_buffer[1]

            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            k_i = einops.rearrange(k_i, 's b h d -> b s h d')
            v_i = einops.rearrange(v_i, 's b h d -> b s h d')
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d')

            dq_i, _dk_i, _dv_i = memory_efficient_attention_backward(
                dout_i,
                outs[i],
                lses[i],
                q_i,
                k_i,
                v_i,
                attn_bias=attn_bias,
                p=p,
                scale=scale,
                op=op,
            )

            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dk_i = einops.rearrange(_dk_i, 'b s h d -> s b h d')
            _dv_i = einops.rearrange(_dv_i, 'b s h d -> s b h d')
            if pg is None:
                dk_i = _dk_i
                dv_i = _dv_i
            else:
                dk_i = torch.zeros(
                    (k_i.shape[1] // cp_size, k_i.shape[0], k_i.shape[2], k_i.shape[3]),
                    device=k_i.device,
                    dtype=k_i.dtype,
                )
                dv_i = torch.zeros(
                    (v_i.shape[1] // cp_size, v_i.shape[0], v_i.shape[2], v_i.shape[3]),
                    device=v_i.device,
                    dtype=v_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dk_i, _dk_i, group=pg)
                torch.distributed.reduce_scatter_tensor(dv_i, _dv_i, group=pg)

            dq.append(dq_i)
            dk.append(dk_i)
            dv.append(dv_i)

        dq = torch.cat(dq, dim=2)
        dk = torch.cat(dk, dim=2)
        dv = torch.cat(dv, dim=2)
        return dq, dk, dv, None, None, None, None


class DotProductAttentionCp(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        if model_comm_pgs is None:
            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                model_comm_pgs, 'tp'
            ), "DotProductAttention model_comm_pgs must have tp process group"

        world_size = model_comm_pgs.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        pg = None
        if self.config.context_parallel_size > 1:
            pg = mpu.get_context_parallel_group()
        output = AttnFuncWithCp.apply(
            query,
            key,
            value,
            attention_mask,
            self.config.attention_dropout,
            self.softmax_scale,
            pg,
        )

        output_size = (query.shape[0], query.shape[1], self.hidden_size_per_partition)
        output = output.view(*output_size)
        return output
