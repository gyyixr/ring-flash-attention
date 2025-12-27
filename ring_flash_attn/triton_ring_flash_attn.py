"""
Triton implementation of Ring Flash Attention.
Uses Triton kernels for attention computation instead of FlashAttention CUDA kernels.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Optional, Tuple
from .utils import RingComm, update_out_and_lse, get_default_args


@triton.jit
def _fwd_kernel(
    Q, K, V, Out, L,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    stride_lb, stride_lh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    seqlen_q: tl.constexpr,
    seqlen_k: tl.constexpr,
):
    """Triton kernel for forward pass of attention computation.
    
    Implements online softmax algorithm for numerical stability.
    Each program instance processes one block of Q rows.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = tl.program_id(2)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers for Q, K, V
    q_ptrs = Q + off_b * stride_qb + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_hz * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    
    # Initialize output accumulator and statistics
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    # Loop over K, V blocks
    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K, V blocks
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < seqlen_k, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < seqlen_k, other=0.0)
        
        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= softmax_scale
        
        # Apply causal mask if needed
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        
        # Update statistics
        l_ij = alpha * l_i + tl.sum(p, 1)
        
        # Update output
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update m_i and l_i
        m_i = m_ij
        l_i = l_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = Out + off_b * stride_ob + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)
    
    # Store LSE (log-sum-exp)
    l_ptrs = L + off_b * stride_lb + off_hz * stride_lh + offs_m
    lse = m_i + tl.log(l_i)
    tl.store(l_ptrs, lse, mask=offs_m < seqlen_q)


@triton.jit
def _bwd_kernel_dq(
    Q, K, V, Out, dOut, dQ, L,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    seqlen_q: tl.constexpr,
    seqlen_k: tl.constexpr,
):
    """Triton kernel for backward pass - computing dQ."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = tl.program_id(2)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q, O, dO, LSE
    q_ptrs = Q + off_b * stride_qb + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    o_ptrs = Out + off_b * stride_ob + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    o = tl.load(o_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    do_ptrs = dOut + off_b * stride_ob + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    do = tl.load(do_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    
    l_ptrs = L + off_b * stride_qb + off_hz * stride_qh + offs_m
    l = tl.load(l_ptrs, mask=offs_m < seqlen_q, other=0.0)
    
    # Compute D_i = rowsum(dO * O)
    Di = tl.sum(do * o, 1)
    
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over K, V blocks
    k_ptrs = K + off_b * stride_kb + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_hz * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    
    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K, V
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < seqlen_k, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < seqlen_k, other=0.0)
        
        # Recompute attention weights
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        
        # Compute dP
        dp = tl.dot(do, tl.trans(v))
        
        # Compute dS = P * (dP - D_i)
        ds = p * (dp - Di[:, None])
        
        # Accumulate dQ
        dq_acc += tl.dot(ds.to(k.dtype), k) * softmax_scale
    
    # Store dQ
    dq_ptrs = dQ + off_b * stride_qb + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _bwd_kernel_dkv(
    Q, K, V, Out, dOut, dK, dV, L,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    seqlen_q: tl.constexpr,
    seqlen_k: tl.constexpr,
):
    """Triton kernel for backward pass - computing dK and dV."""
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = tl.program_id(2)
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load K, V for this block
    k_ptrs = K + off_b * stride_kb + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_hz * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    
    k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    
    # Initialize dK, dV accumulators
    dk_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over Q blocks
    q_ptrs = Q + off_b * stride_qb + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    do_ptrs = dOut + off_b * stride_ob + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    o_ptrs = Out + off_b * stride_ob + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    l_ptrs = L + off_b * stride_qb + off_hz * stride_qh + offs_m
    
    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        
        # Load Q, dO, O, LSE
        q = tl.load(q_ptrs + start_m * stride_qm, mask=(start_m + offs_m[:, None]) < seqlen_q, other=0.0)
        do = tl.load(do_ptrs + start_m * stride_om, mask=(start_m + offs_m[:, None]) < seqlen_q, other=0.0)
        o = tl.load(o_ptrs + start_m * stride_om, mask=(start_m + offs_m[:, None]) < seqlen_q, other=0.0)
        l = tl.load(l_ptrs + start_m, mask=(start_m + offs_m) < seqlen_q, other=0.0)
        
        # Recompute attention weights
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        
        if CAUSAL:
            causal_mask = ((start_m + offs_m[:, None]) >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        
        # Compute D_i
        Di = tl.sum(do * o, 1)
        
        # Compute dP
        dp = tl.dot(do, tl.trans(v))
        
        # Compute dS = P * (dP - D_i)
        ds = p * (dp - Di[:, None])
        
        # Accumulate dK and dV
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * softmax_scale
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
    
    # Store dK, dV
    dk_ptrs = dK + off_b * stride_kb + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    dv_ptrs = dV + off_b * stride_vb + off_hz * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)


def triton_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of FlashAttention using Triton.
    
    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim]
        k: Key tensor [batch, seqlen_k, num_heads, head_dim]
        v: Value tensor [batch, seqlen_k, num_heads, head_dim]
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        
    Returns:
        out: Output tensor [batch, seqlen_q, num_heads, head_dim]
        lse: Log-sum-exp tensor [batch, num_heads, seqlen_q]
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    
    # Allocate output
    out = torch.empty_like(q)
    lse = torch.empty((batch, num_heads, seqlen_q), dtype=torch.float32, device=q.device)
    
    # Kernel launch parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    
    grid = (triton.cdiv(seqlen_q, BLOCK_M), num_heads, batch)
    
    _fwd_kernel[grid](
        q, k, v, out, lse,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        out.stride(0), out.stride(2), out.stride(1),
        lse.stride(0), lse.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=causal,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_warps=4,
        num_stages=2,
    )
    
    return out, lse


def triton_flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass of FlashAttention using Triton.
    
    Args:
        dout: Gradient of output [batch, seqlen_q, num_heads, head_dim]
        q, k, v: Input tensors from forward pass
        out: Output from forward pass
        lse: LSE from forward pass [batch, num_heads, seqlen_q]
        softmax_scale: Scaling factor
        causal: Whether causal masking was applied
        
    Returns:
        dq, dk, dv: Gradients of inputs
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    
    # Allocate gradients
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    
    # Compute dQ
    grid_dq = (triton.cdiv(seqlen_q, BLOCK_M), num_heads, batch)
    _bwd_kernel_dq[grid_dq](
        q, k, v, out, dout, dq, lse,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        out.stride(0), out.stride(2), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=causal,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_warps=4,
        num_stages=2,
    )
    
    # Compute dK and dV
    grid_dkv = (triton.cdiv(seqlen_k, BLOCK_N), num_heads, batch)
    _bwd_kernel_dkv[grid_dkv](
        q, k, v, out, dout, dk, dv, lse,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        out.stride(0), out.stride(2), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=causal,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_warps=4,
        num_stages=2,
    )
    
    return dq, dk, dv


def triton_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ring attention forward pass using Triton kernels.
    
    Similar to ring_flash_attn_forward but uses Triton instead of FlashAttention.
    """
    comm = RingComm(process_group)
    
    out = None
    lse = None
    
    next_k, next_v = None, None
    
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        
        if not causal or step <= comm.rank:
            # Use Triton kernel for computation
            block_out, block_lse = triton_flash_attn_forward(
                q, k, v,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
            )
            
            # Convert lse to expected format [batch, seqlen, num_heads]
            block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
            
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def triton_ring_flash_attn_backward(
    process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ring attention backward pass using Triton kernels.
    """
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    
    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    
    next_k, next_v = None, None
    
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        
        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            
            # Use Triton kernel for backward computation
            block_dq, block_dk, block_dv = triton_flash_attn_backward(
                dout, q, k, v, out, softmax_lse,
                softmax_scale=softmax_scale,
                causal=bwd_causal,
            )
            
            block_dq_buffer.copy_(block_dq)
            block_dk_buffer.copy_(block_dk)
            block_dv_buffer.copy_(block_dv)
            
            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv
        
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        
        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
    
    d_kv_comm.wait()
    
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class TritonRingFlashAttnFunc(torch.autograd.Function):
    """Autograd function wrapper for Triton Ring Flash Attention."""
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_softmax: bool,
        group: Optional[dist.ProcessGroup],
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        assert alibi_slopes is None, "alibi_slopes not supported in Triton implementation"
        assert dropout_p == 0, "dropout not supported in Triton implementation"
        
        k = k.contiguous()
        v = v.contiguous()
        
        out, softmax_lse = triton_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        
        dq, dk, dv = triton_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        
        return dq, dk, dv, None, None, None, None, None, None, None, None


def triton_ring_flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
):
    """Triton Ring Flash Attention with packed QKV."""
    return TritonRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def triton_ring_flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
):
    """Triton Ring Flash Attention with packed KV."""
    return TritonRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def triton_ring_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
):
    """Triton Ring Flash Attention with separate Q, K, V tensors."""
    return TritonRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
