"""Quick test script to verify Triton implementation works."""

import torch
import torch.distributed as dist
from ring_flash_attn.triton_ring_flash_attn import triton_flash_attn_forward, triton_flash_attn_backward


def test_triton_kernel():
    """Test Triton kernels standalone (no distributed)."""
    print("Testing Triton FlashAttention kernels...")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    batch = 2
    seqlen = 512
    num_heads = 4
    head_dim = 64
    
    q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    
    softmax_scale = head_dim ** (-0.5)
    
    # Test forward
    try:
        out, lse = triton_flash_attn_forward(q, k, v, softmax_scale, causal=True)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {out.shape}")
        print(f"  LSE shape: {lse.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward
    try:
        dout = torch.randn_like(out)
        dq, dk, dv = triton_flash_attn_backward(
            dout, q, k, v, out, lse, softmax_scale, causal=True
        )
        print(f"✓ Backward pass successful")
        print(f"  dQ shape: {dq.shape}")
        print(f"  dK shape: {dk.shape}")
        print(f"  dV shape: {dv.shape}")
        print(f"  dQ range: [{dq.min():.4f}, {dq.max():.4f}]")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All Triton kernel tests passed!")
    return True


def test_triton_ring_attention():
    """Test Triton ring attention with distributed."""
    print("\nTesting Triton Ring Attention (requires distributed)...")
    
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        
        from ring_flash_attn.triton_ring_flash_attn import triton_ring_flash_attn_qkvpacked_func
        
        batch = 1
        seqlen = 1024
        nheads = 4
        head_dim = 64
        
        qkv = torch.randn(batch, seqlen, 3, nheads, head_dim, device=device, dtype=torch.bfloat16)
        
        out = triton_ring_flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            causal=True,
        )
        
        if rank == 0:
            print(f"✓ Triton ring attention forward successful")
            print(f"  Output shape: {out.shape}")
        
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"✗ Triton ring attention failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except:
            pass
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Triton Implementation Verification")
    print("=" * 80)
    
    # Test standalone kernels
    success = test_triton_kernel()
    
    # Test distributed version if available
    if success and dist.is_available():
        try:
            # This will only work if run with torchrun
            test_triton_ring_attention()
        except RuntimeError as e:
            print(f"\nNote: Distributed test skipped (not running with torchrun)")
            print(f"  To test distributed: torchrun --nproc_per_node=2 {__file__}")
    
    print("=" * 80)
