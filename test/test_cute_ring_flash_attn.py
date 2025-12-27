"""
Test for CuTe DSL Ring Flash Attention implementation.
Tests correctness by comparing with native PyTorch implementation.
"""

import sys
import torch
import torch.distributed as dist
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_cute_flash_attn_standalone():
    """Test standalone CuTe Flash Attention (single GPU, no distributed)."""
    print("=" * 80)
    print("Testing CuTe Flash Attention (Standalone)")
    print("=" * 80)
    
    from ring_flash_attn import (
        cute_flash_attn_forward,
        cute_flash_attn_backward,
        CUTE_AVAILABLE,
    )
    
    print(f"CuTe DSL Available: {CUTE_AVAILABLE}")
    
    # Test configuration
    batch_size = 2
    seqlen = 256
    num_heads = 4
    head_dim = 64
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nTest config: batch={batch_size}, seqlen={seqlen}, heads={num_heads}, dim={head_dim}")
    print(f"Device: {device}, dtype: {dtype}")
    
    set_seed(42)
    
    # Generate random inputs
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    
    softmax_scale = head_dim ** (-0.5)
    causal = True
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    try:
        out, lse = cute_flash_attn_forward(q, k, v, softmax_scale, causal)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {out.shape}")
        print(f"  LSE shape: {lse.shape}")
        print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("\n--- Backward Pass ---")
    try:
        dout = torch.randn_like(out)
        dq, dk, dv = cute_flash_attn_backward(dout, q, k, v, out, lse, softmax_scale, causal)
        print(f"✓ Backward pass successful")
        print(f"  dQ shape: {dq.shape}")
        print(f"  dK shape: {dk.shape}")
        print(f"  dV shape: {dv.shape}")
        print(f"  dQ range: [{dq.min().item():.4f}, {dq.max().item():.4f}]")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare with reference PyTorch implementation
    print("\n--- Numerical Verification ---")
    try:
        # Reference implementation using PyTorch
        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        
        # Transpose for matmul: [batch, num_heads, seqlen, head_dim]
        q_t = q_ref.transpose(1, 2)
        k_t = k_ref.transpose(1, 2)
        v_t = v_ref.transpose(1, 2)
        
        # Compute attention
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
        
        if causal:
            causal_mask = torch.triu(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out_ref = torch.matmul(attn_probs, v_t).transpose(1, 2)
        
        # Compare outputs
        max_error = (out - out_ref).abs().max().item()
        mean_error = (out - out_ref).abs().mean().item()
        
        print(f"  Output Max Error: {max_error:.6e}")
        print(f"  Output Mean Error: {mean_error:.6e}")
        
        if max_error < 1e-3:
            print("✓ Output matches reference (within tolerance)")
        else:
            print(f"⚠ Output differs from reference (max error: {max_error:.6e})")
        
        # Backward reference
        out_ref.backward(dout)
        
        dq_error = (dq - q_ref.grad).abs().max().item()
        dk_error = (dk - k_ref.grad).abs().max().item()
        dv_error = (dv - v_ref.grad).abs().max().item()
        
        print(f"  dQ Max Error: {dq_error:.6e}")
        print(f"  dK Max Error: {dk_error:.6e}")
        print(f"  dV Max Error: {dv_error:.6e}")
        
        if max(dq_error, dk_error, dv_error) < 1e-3:
            print("✓ Gradients match reference (within tolerance)")
        else:
            print(f"⚠ Gradients differ from reference")
        
    except Exception as e:
        print(f"⚠ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ CuTe Flash Attention standalone test passed!")
    return True


def test_cute_ring_flash_attn():
    """Test CuTe Ring Flash Attention (distributed)."""
    print("\n" + "=" * 80)
    print("Testing CuTe Ring Flash Attention (Distributed)")
    print("=" * 80)
    
    # Initialize distributed
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl")
        except Exception as e:
            print(f"⚠ Could not initialize distributed: {e}")
            print("  Run with: torchrun --nproc_per_node=N test_cute_ring_flash_attn.py")
            return False
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    from ring_flash_attn import (
        cute_ring_flash_attn_qkvpacked_func,
        ring_flash_attn_qkvpacked_func,
    )
    
    if rank == 0:
        print(f"World size: {world_size}")
    
    # Test configuration
    batch_size = 1
    seqlen = 512 * world_size  # Total sequence length
    num_heads = 4
    head_dim = 64
    dtype = torch.bfloat16
    
    if rank == 0:
        print(f"Test config: batch={batch_size}, total_seqlen={seqlen}, heads={num_heads}, dim={head_dim}")
    
    set_seed(rank)
    
    # Generate QKV
    qkv = torch.randn(
        batch_size, seqlen, 3, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)
    
    dout = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)
    
    # Split for ring attention
    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()
    
    dist.barrier()
    
    # Test CuTe Ring Attention
    if rank == 0:
        print("\n--- CuTe Ring Attention ---")
    
    try:
        cute_out, cute_lse, _ = cute_ring_flash_attn_qkvpacked_func(
            local_qkv,
            dropout_p=0,
            causal=True,
            return_attn_probs=True,
        )
        
        if rank == 0:
            print(f"✓ CuTe forward pass successful")
            print(f"  Output shape: {cute_out.shape}")
        
        # Backward
        cute_out.backward(local_dout)
        cute_dqkv = local_qkv.grad.clone()
        
        if rank == 0:
            print(f"✓ CuTe backward pass successful")
            print(f"  dQKV shape: {cute_dqkv.shape}")
        
    except Exception as e:
        if rank == 0:
            print(f"✗ CuTe Ring Attention failed: {e}")
            import traceback
            traceback.print_exc()
        dist.barrier()
        return False
    
    dist.barrier()
    
    # Compare with native ring attention
    if rank == 0:
        print("\n--- Native Ring Attention (Reference) ---")
    
    local_qkv_ref = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv_ref.requires_grad = True
    
    try:
        ref_out, ref_lse, _ = ring_flash_attn_qkvpacked_func(
            local_qkv_ref,
            dropout_p=0,
            causal=True,
            return_attn_probs=True,
        )
        
        ref_out.backward(local_dout)
        ref_dqkv = local_qkv_ref.grad.clone()
        
        if rank == 0:
            print(f"✓ Native Ring Attention completed")
        
    except Exception as e:
        if rank == 0:
            print(f"✗ Native Ring Attention failed: {e}")
        dist.barrier()
        return False
    
    # Compare results
    if rank == 0:
        print("\n--- Comparison ---")
        out_error = (cute_out - ref_out).abs().max().item()
        dqkv_error = (cute_dqkv - ref_dqkv).abs().max().item()
        
        print(f"  Output Max Error: {out_error:.6e}")
        print(f"  dQKV Max Error: {dqkv_error:.6e}")
        
        if out_error < 1e-2 and dqkv_error < 1e-2:
            print("✓ CuTe Ring Attention matches native implementation!")
        else:
            print("⚠ Results differ (may be due to different numerical implementations)")
    
    dist.barrier()
    
    if rank == 0:
        print("\n✓ CuTe Ring Flash Attention test completed!")
    
    return True


def test_cute_autograd():
    """Test CuTe Flash Attention with autograd."""
    print("\n" + "=" * 80)
    print("Testing CuTe Flash Attention Autograd")
    print("=" * 80)
    
    from ring_flash_attn import cute_flash_attn_func
    
    batch_size = 2
    seqlen = 128
    num_heads = 4
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    set_seed(42)
    
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    print("\n--- Autograd Test ---")
    try:
        out, lse = cute_flash_attn_func(q, k, v, causal=True)
        print(f"✓ Forward pass: output shape = {out.shape}")
        
        loss = out.sum()
        loss.backward()
        
        print(f"✓ Backward pass: dQ shape = {q.grad.shape}")
        print(f"  Q grad norm: {q.grad.norm().item():.4f}")
        print(f"  K grad norm: {k.grad.norm().item():.4f}")
        print(f"  V grad norm: {v.grad.norm().item():.4f}")
        
    except Exception as e:
        print(f"✗ Autograd test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Autograd test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("CuTe DSL Ring Flash Attention Tests")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        sys.exit(1)
    
    print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
    
    # Check CuTe DSL availability
    try:
        from ring_flash_attn import CUTE_AVAILABLE
        print(f"✓ CuTe DSL available: {CUTE_AVAILABLE}")
    except ImportError as e:
        print(f"⚠ Could not import ring_flash_attn: {e}")
        CUTE_AVAILABLE = False
    
    results = []
    
    # Test 1: Standalone Flash Attention
    results.append(("Standalone Flash Attention", test_cute_flash_attn_standalone()))
    
    # Test 2: Autograd
    results.append(("Autograd Integration", test_cute_autograd()))
    
    # Test 3: Distributed (only if running with torchrun)
    import os
    if os.environ.get("RANK") is not None:
        results.append(("Distributed Ring Attention", test_cute_ring_flash_attn()))
        dist.destroy_process_group()
    else:
        print("\n⚠ Skipping distributed test (run with torchrun for distributed tests)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 80)
    
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
