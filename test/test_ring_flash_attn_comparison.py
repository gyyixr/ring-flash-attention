"""
Comparison test for Ring Flash Attention implementations.
Tests correctness and performance of native FlashAttention vs Triton implementations.
"""

import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from ring_flash_attn.triton_ring_flash_attn import triton_ring_flash_attn_qkvpacked_func
from utils import log, set_seed
import time


def compute_error_metrics(tensor1, tensor2, name=""):
    """Compute error metrics between two tensors."""
    abs_diff = (tensor1 - tensor2).abs()
    max_error = abs_diff.max().item()
    mean_error = abs_diff.mean().item()
    
    # Relative error
    rel_error = (abs_diff / (tensor1.abs() + 1e-8)).mean().item()
    
    # RMSE
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    
    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "rel_error": rel_error,
        "rmse": rmse,
    }


def run_single_test(
    func,
    qkv,
    dout,
    dropout_p,
    causal,
    deterministic,
    forward_only=False,
    warmup_iters=5,
    num_iters=10,
):
    """
    Run a single implementation test.
    
    Returns:
        out: Forward output
        lse: Log-sum-exp statistics
        dqkv: Gradient (if not forward_only)
        elapsed_time: Total time in seconds
        memory_allocated: Peak memory in bytes
    """
    local_qkv = qkv.detach().clone()
    local_qkv.requires_grad = not forward_only
    
    # Warmup
    for _ in range(warmup_iters):
        if forward_only:
            with torch.no_grad():
                _ = func(
                    local_qkv,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=True,
                )
        else:
            local_qkv.grad = None
            out_warmup = func(
                local_qkv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=True,
            )
            if isinstance(out_warmup, tuple):
                out_warmup = out_warmup[0]
            out_warmup.backward(dout)
    
    # Clear memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Actual measurement
    start_time = time.time()
    
    for _ in range(num_iters):
        if forward_only:
            with torch.no_grad():
                result = func(
                    local_qkv,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=True,
                )
        else:
            local_qkv.grad = None
            result = func(
                local_qkv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=True,
            )
            out_iter, lse_iter, _ = result
            out_iter.backward(dout)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    # Get results
    out, lse, _ = result if isinstance(result, tuple) else (result, None, None)
    dqkv = local_qkv.grad if not forward_only else None
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    
    return out, lse, dqkv, elapsed_time, memory_allocated


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    
    # Test configuration
    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False
    
    # Performance test settings
    forward_only = False
    warmup_iters = 5
    num_iters = 20
    
    assert seqlen % world_size == 0
    assert d % 8 == 0
    
    # Generate test data
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)
    
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)
    
    # Prepare local data for ring attention
    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()
    
    dist.barrier()
    
    # ========== Test 1: Native FlashAttention (Reference) ==========
    if rank == 0:
        print("=" * 80)
        print("Testing Native FlashAttention (Reference)")
        print("=" * 80)
    
    out_ref, lse_ref, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    
    local_out_ref = out_ref.chunk(world_size, dim=1)[rank]
    local_lse_ref = lse_ref.chunk(world_size, dim=-1)[rank]
    
    out_ref.backward(dout)
    dqkv_ref = qkv.grad
    local_dqkv_ref = dqkv_ref.chunk(world_size, dim=1)[rank]
    
    if rank == 0:
        print(f"✓ Native FlashAttention forward/backward completed")
    
    dist.barrier()
    
    # ========== Test 2: Ring Flash Attention (Native) ==========
    if rank == 0:
        print("\n" + "=" * 80)
        print("Testing Ring Flash Attention (Native Implementation)")
        print("=" * 80)
    
    ring_out, ring_lse, ring_dqkv, ring_time, ring_mem = run_single_test(
        ring_flash_attn_qkvpacked_func,
        local_qkv,
        local_dout,
        dropout_p,
        causal,
        deterministic,
        forward_only=forward_only,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
    )
    
    # Compute errors vs reference
    ring_out_error = compute_error_metrics(local_out_ref, ring_out, "Ring Output")
    ring_lse_error = compute_error_metrics(local_lse_ref, ring_lse, "Ring LSE")
    
    if not forward_only:
        ring_dq_error = compute_error_metrics(local_dqkv_ref[:, :, 0], ring_dqkv[:, :, 0], "Ring dQ")
        ring_dk_error = compute_error_metrics(local_dqkv_ref[:, :, 1], ring_dqkv[:, :, 1], "Ring dK")
        ring_dv_error = compute_error_metrics(local_dqkv_ref[:, :, 2], ring_dqkv[:, :, 2], "Ring dV")
    
    if rank == 0:
        print(f"✓ Ring Flash Attention completed")
        print(f"  Output Max Error: {ring_out_error['max_error']:.6e}")
        print(f"  Output RMSE: {ring_out_error['rmse']:.6e}")
        print(f"  LSE Max Error: {ring_lse_error['max_error']:.6e}")
        if not forward_only:
            print(f"  dQ Max Error: {ring_dq_error['max_error']:.6e}")
            print(f"  dK Max Error: {ring_dk_error['max_error']:.6e}")
            print(f"  dV Max Error: {ring_dv_error['max_error']:.6e}")
        print(f"  Time: {ring_time:.4f}s ({num_iters/ring_time:.2f} iter/s)")
        print(f"  Peak Memory: {ring_mem:.2f} GB")
    
    dist.barrier()
    
    # ========== Test 3: Triton Ring Flash Attention ==========
    if rank == 0:
        print("\n" + "=" * 80)
        print("Testing Triton Ring Flash Attention")
        print("=" * 80)
    
    try:
        triton_out, triton_lse, triton_dqkv, triton_time, triton_mem = run_single_test(
            triton_ring_flash_attn_qkvpacked_func,
            local_qkv,
            local_dout,
            dropout_p,
            causal,
            deterministic,
            forward_only=forward_only,
            warmup_iters=warmup_iters,
            num_iters=num_iters,
        )
        
        # Compute errors vs reference
        triton_out_error = compute_error_metrics(local_out_ref, triton_out, "Triton Output")
        triton_lse_error = compute_error_metrics(local_lse_ref, triton_lse, "Triton LSE")
        
        if not forward_only:
            triton_dq_error = compute_error_metrics(local_dqkv_ref[:, :, 0], triton_dqkv[:, :, 0], "Triton dQ")
            triton_dk_error = compute_error_metrics(local_dqkv_ref[:, :, 1], triton_dqkv[:, :, 1], "Triton dK")
            triton_dv_error = compute_error_metrics(local_dqkv_ref[:, :, 2], triton_dqkv[:, :, 2], "Triton dV")
        
        if rank == 0:
            print(f"✓ Triton Ring Flash Attention completed")
            print(f"  Output Max Error: {triton_out_error['max_error']:.6e}")
            print(f"  Output RMSE: {triton_out_error['rmse']:.6e}")
            print(f"  LSE Max Error: {triton_lse_error['max_error']:.6e}")
            if not forward_only:
                print(f"  dQ Max Error: {triton_dq_error['max_error']:.6e}")
                print(f"  dK Max Error: {triton_dk_error['max_error']:.6e}")
                print(f"  dV Max Error: {triton_dv_error['max_error']:.6e}")
            print(f"  Time: {triton_time:.4f}s ({num_iters/triton_time:.2f} iter/s)")
            print(f"  Peak Memory: {triton_mem:.2f} GB")
            print(f"  Relative Efficiency: {(ring_time/triton_time)*100:.1f}%")
    
    except Exception as e:
        if rank == 0:
            print(f"✗ Triton Ring Flash Attention failed: {e}")
            import traceback
            traceback.print_exc()
    
    dist.barrier()
    
    # ========== Summary Report ==========
    if rank == 0:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print("\nCorrectness (vs Native FlashAttention):")
        print("-" * 80)
        print(f"{'Implementation':<25} {'Output Max Err':<15} {'Output RMSE':<15} {'LSE Max Err':<15}")
        print("-" * 80)
        print(f"{'Ring (Native)':<25} {ring_out_error['max_error']:<15.6e} {ring_out_error['rmse']:<15.6e} {ring_lse_error['max_error']:<15.6e}")
        try:
            print(f"{'Triton Ring':<25} {triton_out_error['max_error']:<15.6e} {triton_out_error['rmse']:<15.6e} {triton_lse_error['max_error']:<15.6e}")
        except:
            print(f"{'Triton Ring':<25} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        if not forward_only:
            print("\nGradient Errors:")
            print("-" * 80)
            print(f"{'Implementation':<25} {'dQ Max Err':<15} {'dK Max Err':<15} {'dV Max Err':<15}")
            print("-" * 80)
            print(f"{'Ring (Native)':<25} {ring_dq_error['max_error']:<15.6e} {ring_dk_error['max_error']:<15.6e} {ring_dv_error['max_error']:<15.6e}")
            try:
                print(f"{'Triton Ring':<25} {triton_dq_error['max_error']:<15.6e} {triton_dk_error['max_error']:<15.6e} {triton_dv_error['max_error']:<15.6e}")
            except:
                print(f"{'Triton Ring':<25} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        print("\nPerformance:")
        print("-" * 80)
        print(f"{'Implementation':<25} {'Throughput (iter/s)':<20} {'Time (s)':<15} {'Memory (GB)':<15}")
        print("-" * 80)
        ring_throughput = num_iters / ring_time
        print(f"{'Ring (Native)':<25} {ring_throughput:<20.2f} {ring_time:<15.4f} {ring_mem:<15.2f}")
        try:
            triton_throughput = num_iters / triton_time
            efficiency = (triton_throughput / ring_throughput) * 100
            print(f"{'Triton Ring':<25} {triton_throughput:<20.2f} {triton_time:<15.4f} {triton_mem:<15.2f}")
            print(f"\nTriton Relative Efficiency: {efficiency:.1f}%")
        except:
            print(f"{'Triton Ring':<25} {'N/A':<20} {'N/A':<15} {'N/A':<15}")
        
        print("=" * 80)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
