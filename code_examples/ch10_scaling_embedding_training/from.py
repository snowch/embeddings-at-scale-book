# Code from Chapter 10
# Book: Embeddings at Scale

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GPUInstance:
    """Cloud GPU instance configuration"""
    name: str
    gpu_type: str
    num_gpus: int
    memory_gb: int
    cost_per_hour: float
    tflops_fp16: float  # Peak FP16 throughput

# Cloud GPU pricing (approximate, as of 2024)
CLOUD_INSTANCES = {
    'aws_p4d_24xlarge': GPUInstance(
        name='p4d.24xlarge',
        gpu_type='A100',
        num_gpus=8,
        memory_gb=640,
        cost_per_hour=32.77,
        tflops_fp16=1248  # 8× A100 × 156 TFLOPS
    ),
    'aws_p3_16xlarge': GPUInstance(
        name='p3.16xlarge',
        gpu_type='V100',
        num_gpus=8,
        memory_gb=488,
        cost_per_hour=24.48,
        tflops_fp16=1000  # 8× V100 × 125 TFLOPS
    ),
    'gcp_a2_ultra': GPUInstance(
        name='a2-ultragpu-8g',
        gpu_type='A100',
        num_gpus=8,
        memory_gb=680,
        cost_per_hour=30.00,
        tflops_fp16=1248
    ),
    'azure_nd96amsr_v4': GPUInstance(
        name='Standard_ND96amsr_A100_v4',
        gpu_type='A100',
        num_gpus=8,
        memory_gb=900,
        cost_per_hour=27.20,
        tflops_fp16=1248
    )
}

class CostOptimizer:
    """
    Optimize training cost across different strategies

    Levers:
    1. Instance type (A100 vs V100 vs H100)
    2. Spot vs on-demand pricing (50-90% savings)
    3. Training duration (optimize throughput)
    4. Batch size and accumulation (memory efficiency)
    5. Mixed precision (2-3× speedup)
    6. Early stopping (halt when converged)
    """

    @staticmethod
    def estimate_training_time(
        dataset_size: int,
        batch_size: int,
        epochs: int,
        samples_per_second: float
    ) -> float:
        """
        Estimate training time in hours

        Args:
            dataset_size: Number of training samples
            batch_size: Effective batch size
            epochs: Number of epochs
            samples_per_second: Throughput (depends on hardware)

        Returns:
            hours: Estimated training time
        """
        total_samples = dataset_size * epochs
        total_seconds = total_samples / samples_per_second
        return total_seconds / 3600

    @staticmethod
    def estimate_cost(
        instance: GPUInstance,
        training_hours: float,
        spot_instance: bool = False
    ) -> float:
        """
        Estimate training cost

        Args:
            instance: GPU instance configuration
            training_hours: Training duration
            spot_instance: Use spot/preemptible pricing

        Returns:
            cost: Total cost in USD
        """
        cost_per_hour = instance.cost_per_hour

        if spot_instance:
            # Spot instances typically 50-70% cheaper
            cost_per_hour *= 0.4  # 60% discount

        return cost_per_hour * training_hours

    @staticmethod
    def compare_strategies(
        dataset_size: int = 1_000_000_000,
        batch_size: int = 32768,
        epochs: int = 10
    ):
        """
        Compare cost across different instance types and strategies

        Scenario: Train embedding model on 1B samples, 10 epochs

        Args:
            dataset_size: Training samples
            batch_size: Effective batch size
            epochs: Number of epochs
        """

        print("=== Training Cost Comparison ===")
        print(f"Dataset: {dataset_size / 1e9:.1f}B samples")
        print(f"Batch size: {batch_size:,}")
        print(f"Epochs: {epochs}")
        print()

        strategies = [
            # (instance_key, throughput_samples_per_sec, use_spot)
            ('aws_p4d_24xlarge', 50000, False),  # A100, on-demand
            ('aws_p4d_24xlarge', 50000, True),   # A100, spot
            ('aws_p3_16xlarge', 30000, False),   # V100, on-demand
            ('aws_p3_16xlarge', 30000, True),    # V100, spot
        ]

        results = []

        for instance_key, throughput, use_spot in strategies:
            instance = CLOUD_INSTANCES[instance_key]

            # Estimate training time
            hours = CostOptimizer.estimate_training_time(
                dataset_size, batch_size, epochs, throughput
            )

            # Estimate cost
            cost = CostOptimizer.estimate_cost(instance, hours, use_spot)

            results.append({
                'instance': instance.name,
                'gpu': instance.gpu_type,
                'pricing': 'Spot' if use_spot else 'On-Demand',
                'hours': hours,
                'cost': cost,
                'cost_per_hour': instance.cost_per_hour if not use_spot else instance.cost_per_hour * 0.4
            })

        # Print results
        for r in results:
            print(f"{r['instance']} ({r['gpu']}, {r['pricing']}):")
            print(f"  Training time: {r['hours']:.1f} hours")
            print(f"  Cost per hour: ${r['cost_per_hour']:.2f}")
            print(f"  Total cost: ${r['cost']:.2f}")
            print()

        # Find best option
        best = min(results, key=lambda x: x['cost'])
        print(f"💰 Best option: {best['instance']} ({best['pricing']}) - ${best['cost']:.2f}")

# Uncomment to run:
# CostOptimizer.compare_strategies()
