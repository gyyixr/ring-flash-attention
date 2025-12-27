"""
Performance benchmark comparison for Ring Flash Attention implementations.
Compares native FlashAttention, Ring (native), and Triton implementations.
"""

from flash_attn import flash_attn_kvpacked_func
import os
import sys
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_kvpacked_func
from ring_flash_attn.triton_ring_flash_attn import triton_ring_flash_attn_kvpacked_func


def benchmark_single(
    f, 
    q, 
    kv, 
    causal, 
    num_iter=100, 
    forward_only=True, 
    log=True, 
    profile=False,
    dout=None,
):
    """
    Benchmark a single implementation.
    
    Args:
        f: Function to benchmark
        q: Query tensor
        kv: Key-value packed tensor
        causal: Whether to use causal attention
        num_iter: Number of iterations
        forward_only: Whether to only benchmark forward pass
        log: Whether to log results
        profile: Whether to enable profiling
        dout: Gradient output tensor for backward pass
        
    Returns:
        throughput: Iterations per second
        time: Total time in seconds
        memory: Peak memory in GB
    """
    rank = dist.get_rank()
    device = q.device
    dtype = q.dtype
    
    deterministic = False
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{rank}"
                )
            ),
        )
        profiler.start()
    
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    
    if forward_only:
        with torch.no_grad():
            for i in range(num_iter):
                _ = f(
                    q,
                    kv,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )
                if profile:
                    profiler.step()
    else:
        for i in range(num_iter):
            q.grad = None
            kv.grad = None
            out = f(
                q,
                kv,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
            if profile:
                profiler.step()
    
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time_elapsed = begin.elapsed_time(end) / 1000.0
    
    if profile:
        profiler.stop()
    
    # Get memory usage
    memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    
    throughput = num_iter / time_elapsed
    
    if rank == 0 and log:
        print(f"{throughput:.3f} iter/s, {time_elapsed:.3f} sec, {memory_gb:.2f} GB")
    
    return throughput, time_elapsed, memory_gb


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    forward_only = False
    profile = False
    num_iter = 500 if forward_only else 100
    compile_func = False
    
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        compile_func = True
        torch._dynamo.config.capture_scalar_outputs = True
    
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    batch_size = 1
    deterministic = False
    # Config of llama3 8B
    seqlen = 1024 * 8
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    causal = True
    
    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0
    
    # Prepare data
    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    kv = torch.randn(
        batch_size,
        seqlen,
        2,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )
    
    if rank == 0:
        print("=" * 80)
        print("Ring Flash Attention Performance Comparison")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seqlen}")
        print(f"  Num heads: {num_heads}")
        print(f"  Num KV heads: {num_kv_heads}")
        print(f"  Head dim: {head_dim}")
        print(f"  Causal: {causal}")
        print(f"  Forward only: {forward_only}")
        print(f"  Num iterations: {num_iter}")
        print(f"  World size: {world_size}")
        print("=" * 80)
    
    results = {}
    
    # Test 1: Flash Attention (baseline, single GPU equivalent)
    if rank == 0:
        print("\n# flash_attn_kvpacked_func (baseline)")
    
    flash_func = flash_attn_kvpacked_func
    if compile_func:
        flash_func = torch.compile(flash_func)
    
    torch.cuda.empty_cache()
    # Warmup
    benchmark_single(flash_func, q, kv, causal, num_iter=10, forward_only=forward_only, log=False, dout=dout)
    # Actual run
    flash_throughput, flash_time, flash_mem = benchmark_single(
        flash_func, q, kv, causal, num_iter=num_iter, forward_only=forward_only, log=True, profile=profile, dout=dout
    )
    results['flash_attn'] = {
        'throughput': flash_throughput,
        'time': flash_time,
        'memory': flash_mem,
    }
    
    # Test 2: Ring Flash Attention (native)
    if rank == 0:
        print("\n# ring_flash_attn_kvpacked_func (native)")
    
    ring_func = ring_flash_attn_kvpacked_func
    if compile_func:
        ring_func = torch.compile(ring_func)
    
    torch.cuda.empty_cache()
    # Warmup
    benchmark_single(ring_func, q, kv, causal, num_iter=10, forward_only=forward_only, log=False, dout=dout)
    # Actual run
    ring_throughput, ring_time, ring_mem = benchmark_single(
        ring_func, q, kv, causal, num_iter=num_iter, forward_only=forward_only, log=True, profile=profile, dout=dout
    )
    results['ring_attn'] = {
        'throughput': ring_throughput,
        'time': ring_time,
        'memory': ring_mem,
    }
    
    # Test 3: Triton Ring Flash Attention
    if rank == 0:
        print("\n# triton_ring_flash_attn_kvpacked_func")
    
    triton_func = triton_ring_flash_attn_kvpacked_func
    if compile_func:
        triton_func = torch.compile(triton_func)
    
    try:
        torch.cuda.empty_cache()
        # Warmup
        benchmark_single(triton_func, q, kv, causal, num_iter=10, forward_only=forward_only, log=False, dout=dout)
        # Actual run
        triton_throughput, triton_time, triton_mem = benchmark_single(
            triton_func, q, kv, causal, num_iter=num_iter, forward_only=forward_only, log=True, profile=profile, dout=dout
        )
        results['triton_ring'] = {
            'throughput': triton_throughput,
            'time': triton_time,
            'memory': triton_mem,
        }
    except Exception as e:
        if rank == 0:
            print(f"Triton implementation failed: {e}")
            import traceback
            traceback.print_exc()
        results['triton_ring'] = None
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        mode = "Forward Only" if forward_only else "Forward + Backward"
        print(f"\nMode: {mode}")
        print(f"Iterations: {num_iter}")
        print("-" * 80)
        print(f"{'Implementation':<30} {'Throughput (iter/s)':<20} {'Time (s)':<15} {'Memory (GB)':<15}")
        print("-" * 80)
        
        # Flash attention (theoretic baseline for ring)
        theoretic_throughput = flash_throughput / world_size
        print(f"{'FlashAttention (theory)':<30} {theoretic_throughput:<20.2f} {'-':<15} {'-':<15}")
        
        # Ring attention (native)
        ring_efficiency = (ring_throughput / theoretic_throughput) * 100
        print(f"{'Ring Attention (native)':<30} {ring_throughput:<20.2f} {ring_time:<15.4f} {ring_mem:<15.2f}")
        print(f"  Relative efficiency: {ring_efficiency:.1f}%")
        
        # Triton ring attention
        if results['triton_ring'] is not None:
            triton_throughput = results['triton_ring']['throughput']
            triton_time = results['triton_ring']['time']
            triton_mem = results['triton_ring']['memory']
            triton_efficiency = (triton_throughput / theoretic_throughput) * 100
            triton_vs_ring = (triton_throughput / ring_throughput) * 100
            
            print(f"{'Triton Ring Attention':<30} {triton_throughput:<20.2f} {triton_time:<15.4f} {triton_mem:<15.2f}")
            print(f"  Relative efficiency: {triton_efficiency:.1f}%")
            print(f"  vs Ring (native): {triton_vs_ring:.1f}%")
        else:
            print(f"{'Triton Ring Attention':<30} {'FAILED':<20} {'-':<15} {'-':<15}")
        
        print("=" * 80)
        
        # Detailed comparison table
        print("\nDetailed Comparison:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Ring (native)':<20} {'Triton Ring':<20}")
        print("-" * 80)
        
        if results['triton_ring'] is not None:
            speedup = triton_throughput / ring_throughput
            mem_overhead = ((triton_mem - ring_mem) / ring_mem) * 100
            
            print(f"{'Throughput (iter/s)':<30} {ring_throughput:<20.2f} {triton_throughput:<20.2f}")
            print(f"{'Speedup':<30} {'1.00x':<20} {speedup:<20.2f}x")
            print(f"{'Memory (GB)':<30} {ring_mem:<20.2f} {triton_mem:<20.2f}")
            print(f"{'Memory Overhead (%)':<30} {'0.0%':<20} {mem_overhead:<20.1f}%")
        else:
            print(f"{'Triton implementation failed':<30}")
        
        print("=" * 80)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
