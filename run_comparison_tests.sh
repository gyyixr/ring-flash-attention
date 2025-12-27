#!/bin/bash

# Ring-Flash-Attention 对比测试快速运行脚本

echo "========================================="
echo "Ring-Flash-Attention 实现对比测试"
echo "========================================="
echo ""

# 检查CUDA可用性
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ CUDA available, {torch.cuda.device_count()} GPUs detected')" || exit 1

# 检查Triton
python -c "import triton; print('✓ Triton installed')" || { echo "✗ Triton not installed. Install with: pip install triton"; exit 1; }

# 检查FlashAttention
python -c "import flash_attn; print('✓ FlashAttention installed')" || { echo "✗ FlashAttention not installed"; exit 1; }

echo ""
echo "========================================="
echo "Step 1: 快速验证测试（单GPU）"
echo "========================================="
python test/quick_test_triton.py

if [ $? -ne 0 ]; then
    echo "✗ 快速验证失败"
    exit 1
fi

echo ""
echo "========================================="
echo "Step 2: 正确性对比测试（需要至少2个GPU）"
echo "========================================="

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "使用 $GPU_COUNT 个GPU进行测试..."
    
    # 使用2个GPU进行测试
    NPROC=2
    if [ "$GPU_COUNT" -ge 4 ]; then
        NPROC=4
    fi
    if [ "$GPU_COUNT" -ge 8 ]; then
        NPROC=8
    fi
    
    echo "运行正确性测试（$NPROC GPU）..."
    torchrun --nproc_per_node=$NPROC test/test_ring_flash_attn_comparison.py
    
    if [ $? -ne 0 ]; then
        echo "✗ 正确性测试失败"
        exit 1
    fi
    
    echo ""
    echo "========================================="
    echo "Step 3: 性能基准测试"
    echo "========================================="
    echo "运行性能测试（$NPROC GPU）..."
    torchrun --nproc_per_node=$NPROC benchmark/benchmark_comparison.py
    
    if [ $? -ne 0 ]; then
        echo "⚠ 性能测试失败（可能是正常的）"
    fi
else
    echo "⚠ 只有 $GPU_COUNT 个GPU，跳过分布式测试"
    echo "  分布式测试需要至少2个GPU"
fi

echo ""
echo "========================================="
echo "测试完成！"
echo "========================================="
echo ""
echo "查看详细文档："
echo "  - 使用指南: COMPARISON_GUIDE.md"
echo "  - 实施总结: IMPLEMENTATION_SUMMARY.md"
echo "  - 设计文档: .qoder/quests/ring-flash-attention-comparison.md"
echo ""
