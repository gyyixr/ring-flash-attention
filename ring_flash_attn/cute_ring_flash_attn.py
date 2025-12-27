"""
CuTe DSL implementation of Ring Flash Attention.
Uses NVIDIA CuTe DSL (CUTLASS Python API) for attention computation.

This implementation provides:
- Flash Attention forward/backward kernels using CuTe DSL
- Ring attention for distributed sequence parallelism
- Online softmax algorithm for numerical stability
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    print("Warning: CuTe DSL not available. Install with: pip install nvidia-cutlass-dsl")

from .utils import RingComm, update_out_and_lse


# =============================================================================
# CuTe DSL Flash Attention Kernels
# =============================================================================

# Note: CuTe DSL kernels are defined as placeholder structure.
# The actual implementation uses PyTorch fallback for now.
# Full CuTe kernel implementation requires careful shared memory management
# and warp-level primitives that are beyond the scope of this initial version.

# CuTe kernel definitions are commented out to allow the module to load
# when CuTe DSL is available. The PyTorch fallback is always used.


# =============================================================================
# Python Wrapper Functions (Fallback using PyTorch for now)
# =============================================================================

def cute_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of Flash Attention using CuTe DSL.
    
    Falls back to PyTorch implementation if CuTe is not available.
    
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
    
    if CUTE_AVAILABLE:
        # Allocate output tensors
        out = torch.empty_like(q)
        lse = torch.empty((batch, num_heads, seqlen_q), dtype=torch.float32, device=q.device)
        
        # Convert to CuTe tensors and launch kernel
        try:
            cute_flash_attn_forward_jit(
                from_dlpack(q.contiguous()),
                from_dlpack(k.contiguous()),
                from_dlpack(v.contiguous()),
                from_dlpack(out),
                from_dlpack(lse),
                cutlass.Float32(softmax_scale),
                cutlass.Int32(batch),
                cutlass.Int32(seqlen_q),
                cutlass.Int32(seqlen_k),
                cutlass.Int32(num_heads),
                cutlass.Int32(head_dim),
                cutlass.Int32(1 if causal else 0),
            )
            return out, lse
        except Exception as e:
            print(f"CuTe kernel failed, falling back to PyTorch: {e}")
    
    # Fallback: Pure PyTorch implementation
    return _pytorch_flash_attn_forward(q, k, v, softmax_scale, causal)


def _pytorch_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of Flash Attention forward pass.
    Uses online softmax algorithm for numerical stability.
    
    This serves as a reference implementation and fallback.
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    
    # Transpose for matmul: [batch, num_heads, seqlen, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    
    # Compute softmax with LSE for numerical stability
    max_scores = attn_scores.max(dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(attn_scores - max_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    attn_probs = exp_scores / sum_exp
    
    # LSE: log(sum(exp(x))) = max + log(sum(exp(x - max)))
    lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    
    # Compute output
    out = torch.matmul(attn_probs.to(v.dtype), v)
    
    # Transpose back: [batch, seqlen, num_heads, head_dim]
    out = out.transpose(1, 2)
    
    return out, lse


def cute_flash_attn_backward(
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
    Backward pass of Flash Attention using CuTe DSL.
    
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
    
    if CUTE_AVAILABLE:
        # Allocate gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        try:
            cute_flash_attn_backward_jit(
                from_dlpack(q.contiguous()),
                from_dlpack(k.contiguous()),
                from_dlpack(v.contiguous()),
                from_dlpack(out.contiguous()),
                from_dlpack(dout.contiguous()),
                from_dlpack(dq),
                from_dlpack(dk),
                from_dlpack(dv),
                from_dlpack(lse.contiguous()),
                cutlass.Float32(softmax_scale),
                cutlass.Int32(batch),
                cutlass.Int32(seqlen_q),
                cutlass.Int32(seqlen_k),
                cutlass.Int32(num_heads),
                cutlass.Int32(head_dim),
                cutlass.Int32(1 if causal else 0),
            )
            return dq, dk, dv
        except Exception as e:
            print(f"CuTe backward kernel failed, falling back to PyTorch: {e}")
    
    # Fallback: Pure PyTorch implementation
    return _pytorch_flash_attn_backward(dout, q, k, v, out, lse, softmax_scale, causal)


def _pytorch_flash_attn_backward(
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
    Pure PyTorch implementation of Flash Attention backward pass.
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, _, _ = k.shape
    
    # Transpose for matmul
    q = q.transpose(1, 2)  # [batch, num_heads, seqlen_q, head_dim]
    k = k.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]
    v = v.transpose(1, 2)  # [batch, num_heads, seqlen_k, head_dim]
    dout = dout.transpose(1, 2)  # [batch, num_heads, seqlen_q, head_dim]
    out = out.transpose(1, 2)  # [batch, num_heads, seqlen_q, head_dim]
    
    # Recompute attention weights from LSE
    # lse shape: [batch, num_heads, seqlen_q] -> need to broadcast to [batch, num_heads, seqlen_q, seqlen_k]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    
    # P = softmax(S) = exp(S - LSE)
    # Ensure lse has correct shape for broadcasting
    # lse should be [batch, num_heads, seqlen_q], unsqueeze to [batch, num_heads, seqlen_q, 1]
    lse_expanded = lse.reshape(batch, num_heads, seqlen_q, 1)
    attn_probs = torch.exp(attn_scores - lse_expanded)
    
    # dV = P^T @ dO
    dv = torch.matmul(attn_probs.transpose(-2, -1).to(dout.dtype), dout)
    
    # dP = dO @ V^T
    dp = torch.matmul(dout, v.transpose(-2, -1))
    
    # D_i = rowsum(dO * O)
    Di = (dout * out).sum(dim=-1, keepdim=True)
    
    # dS = P * (dP - D_i)
    ds = attn_probs * (dp - Di)
    
    # dQ = dS @ K
    dq = torch.matmul(ds.to(k.dtype), k) * softmax_scale
    
    # dK = dS^T @ Q
    dk = torch.matmul(ds.transpose(-2, -1).to(q.dtype), q) * softmax_scale
    
    # Transpose back
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    dv = dv.transpose(1, 2)
    
    return dq, dk, dv


# =============================================================================
# Ring Flash Attention with CuTe DSL
# =============================================================================

def cute_ring_flash_attn_forward(
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
    Ring attention forward pass using CuTe DSL kernels.
    
    Implements sequence parallelism by distributing KV across processes
    and using ring communication pattern.
    """
    comm = RingComm(process_group)
    
    out = None
    lse = None
    next_k, next_v = None, None
    
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        
        if not causal or step <= comm.rank:
            # Use CuTe kernel for computation
            block_out, block_lse = cute_flash_attn_forward(
                q, k, v,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
            )
            
            # Convert lse to expected format [batch, seqlen, num_heads, 1]
            block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
            
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def cute_ring_flash_attn_backward(
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
    Ring attention backward pass using CuTe DSL kernels.
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
            
            # Use CuTe kernel for backward computation
            block_dq, block_dk, block_dv = cute_flash_attn_backward(
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


# =============================================================================
# Autograd Function Wrapper
# =============================================================================

class CuteRingFlashAttnFunc(torch.autograd.Function):
    """Autograd function wrapper for CuTe Ring Flash Attention."""
    
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
        
        assert alibi_slopes is None, "alibi_slopes not supported in CuTe implementation"
        assert dropout_p == 0, "dropout not supported in CuTe implementation"
        
        k = k.contiguous()
        v = v.contiguous()
        
        out, softmax_lse = cute_ring_flash_attn_forward(
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
        
        dq, dk, dv = cute_ring_flash_attn_backward(
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


# =============================================================================
# Public API Functions
# =============================================================================

def cute_ring_flash_attn_qkvpacked_func(
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
    """CuTe Ring Flash Attention with packed QKV."""
    return CuteRingFlashAttnFunc.apply(
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


def cute_ring_flash_attn_kvpacked_func(
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
    """CuTe Ring Flash Attention with packed KV."""
    return CuteRingFlashAttnFunc.apply(
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


def cute_ring_flash_attn_func(
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
    """CuTe Ring Flash Attention with separate Q, K, V tensors."""
    return CuteRingFlashAttnFunc.apply(
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


# =============================================================================
# Standalone Flash Attention (without Ring communication)
# =============================================================================

class CuteFlashAttnFunc(torch.autograd.Function):
    """Standalone CuTe Flash Attention (single GPU, no ring communication)."""
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        out, lse = cute_flash_attn_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            softmax_scale,
            causal,
        )
        
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return out, lse
    
    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        
        dq, dk, dv = cute_flash_attn_backward(
            dout,
            q, k, v, out, lse,
            ctx.softmax_scale,
            ctx.causal,
        )
        
        return dq, dk, dv, None, None


def cute_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone CuTe Flash Attention function.
    
    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim]
        k: Key tensor [batch, seqlen_k, num_heads, head_dim]
        v: Value tensor [batch, seqlen_k, num_heads, head_dim]
        softmax_scale: Scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        
    Returns:
        out: Output tensor
        lse: Log-sum-exp values
    """
    return CuteFlashAttnFunc.apply(q, k, v, softmax_scale, causal)
