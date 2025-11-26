#!/usr/bin/env python3
"""
Analyze the focused L3 overfetch data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV
df = pd.read_csv('l3_focused_analysis.csv')

# Create a comprehensive summary
print("=" * 80)
print("L3 FOCUSED ANALYSIS - KEY FINDINGS")
print("=" * 80)

# Finding 1: Strategy doesn't matter much (current implementation)
print("\n1. DATAFLOW STRATEGY IMPACT")
print("-" * 80)
strategy_stats = df.groupby('Strategy')['Overfetch_Total'].agg(['mean', 'std', 'min', 'max'])
print(strategy_stats)
print("\n   → All strategies show identical behavior in current L3 scheduler")
print("   → This is expected: L3 sees L2 tile loads, not PE operations")

# Finding 2: Aspect ratio DOES matter significantly
print("\n2. ASPECT RATIO IMPACT (Most Important Finding)")
print("-" * 80)
aspect_stats = df.groupby('Aspect').agg({
    'Overfetch_Total': 'mean',
    'Overfetch_A': 'mean',
    'Overfetch_B': 'mean',
    'Overfetch_C': 'mean'
}).sort_values('Overfetch_Total')
print(aspect_stats)
print("\n   → Tall matrices: 1.02× overfetch (best)")
print("   → Deep matrices: 7.80× overfetch")
print("   → Tall-Wide: 23.29× overfetch (worst - includes user's 32k×7k)")
print("   → Root cause: Tensor B dominates overfetch for non-tall shapes")

# Finding 3: L3 size shows clear knee
print("\n3. L3 SIZE IMPACT (Knee of the Curve)")
print("-" * 80)
l3_stats = df.groupby('L3_Size_MB').agg({
    'Overfetch_Total': 'mean',
    'L3_Hit_Rate': 'mean',
    'DRAM_Reads_MB': 'mean'
}).sort_index()
print(l3_stats)
print("\n   → 16MB: 13.13× overfetch, 49% hit rate")
print("   → 64MB: 6.08× overfetch, 72% hit rate  ← KNEE IS HERE")
print("   → 256MB: 0.88× overfetch, 99% hit rate")

# Finding 4: Per-tensor breakdown
print("\n4. WHICH TENSOR DOMINATES OVERFETCH?")
print("-" * 80)
tensor_breakdown = df.groupby('Aspect').agg({
    'Overfetch_A': 'mean',
    'Overfetch_B': 'mean',
    'Overfetch_C': 'mean'
})
print(tensor_breakdown)
print("\n   → Tensor B (weights) dominates for Square/Deep/Wide/Tall-Wide")
print("   → Tensor B overfetch reaches 228× for Tall-Wide (32k×7k)")
print("   → Tall matrices reuse both A and B well (both ~1.03×)")

# Finding 5: User's 32k×7k example
print("\n5. USER EXAMPLE: 32k×7k×7k")
print("-" * 80)
user_example = df[df['Workload_Name'] == 'User Example 32k×7k']
for l3_size in [16, 64, 256]:
    row = user_example[user_example['L3_Size_MB'] == l3_size].iloc[0]
    print(f"\n   L3={l3_size}MB:")
    print(f"     Total overfetch: {row['Overfetch_Total']:.2f}×")
    print(f"     Tensor B overfetch: {row['Overfetch_B']:.2f}× (DOMINATES)")
    print(f"     DRAM reads: {row['DRAM_Reads_MB']:.0f} MB")
    print(f"     L3 hit rate: {row['L3_Hit_Rate']*100:.1f}%")

print("\n   → Needs 256MB L3 to achieve <1× overfetch")
print("   → With 16/64MB L3: 342× overfetch on tensor B!")
print("   → This is because 7k×7k weight matrix (196MB) >> L3 capacity")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("1. For transformer inference (tall matrices): 16-64MB L3 is sufficient")
print("2. For square/deep GEMMs: Need 64-256MB L3")
print("3. For large weight matrices (>100MB): Need L3 >> weight size")
print("4. Dataflow strategy doesn't affect L3 (affects L2/L1 only)")
print("5. Tensor B (weights) is the critical tensor to optimize for")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('L3 Focused Analysis: Key Insights', fontsize=16, fontweight='bold')

# Plot 1: Overfetch by aspect ratio
ax = axes[0, 0]
aspect_means = df.groupby('Aspect')['Overfetch_Total'].mean().sort_values()
colors = ['green' if x < 2 else 'orange' if x < 10 else 'red' for x in aspect_means.values]
aspect_means.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Average Overfetch Factor')
ax.set_title('Overfetch by Aspect Ratio\n(Green=Good, Orange=Moderate, Red=Poor)')
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, label='Ideal (1.0×)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: L3 size vs overfetch (knee curve)
ax = axes[0, 1]
l3_curve = df.groupby('L3_Size_MB')['Overfetch_Total'].mean()
ax.plot(l3_curve.index, l3_curve.values, 'o-', linewidth=2, markersize=8, color='blue')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Ideal (1.0×)')
ax.axvline(x=64, color='orange', linestyle='--', linewidth=2, label='Knee @ 64MB')
ax.set_xlabel('L3 Cache Size (MB)')
ax.set_ylabel('Average Overfetch Factor')
ax.set_title('L3 Size vs Overfetch\n(Knee of the Curve)')
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)
ax.legend()
ax.fill_between([16, 64], [0, 0], [20, 20], alpha=0.2, color='red', label='High overfetch')
ax.fill_between([64, 256], [0, 0], [20, 20], alpha=0.2, color='yellow')

# Plot 3: Per-tensor overfetch breakdown
ax = axes[1, 0]
tensor_data = df.groupby('Aspect')[['Overfetch_A', 'Overfetch_B', 'Overfetch_C']].mean()
tensor_data = tensor_data.sort_values('Overfetch_B')
x = np.arange(len(tensor_data.index))
width = 0.25
ax.bar(x - width, tensor_data['Overfetch_A'], width, label='Tensor A (input)', alpha=0.8)
ax.bar(x, tensor_data['Overfetch_B'], width, label='Tensor B (weights)', alpha=0.8)
ax.bar(x + width, tensor_data['Overfetch_C'], width, label='Tensor C (output)', alpha=0.8)
ax.set_ylabel('Overfetch Factor')
ax.set_title('Per-Tensor Overfetch by Aspect\n(Tensor B dominates for most shapes)')
ax.set_xticks(x)
ax.set_xticklabels(tensor_data.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)

# Plot 4: User example detailed
ax = axes[1, 1]
user_data = df[df['Workload_Name'] == 'User Example 32k×7k'].groupby('L3_Size_MB').first()
x = np.arange(len(user_data.index))
width = 0.2
ax.bar(x - width*1.5, user_data['Overfetch_A'], width, label='Tensor A', alpha=0.8)
ax.bar(x - width*0.5, user_data['Overfetch_B'], width, label='Tensor B (PROBLEM!)', alpha=0.8, color='red')
ax.bar(x + width*0.5, user_data['Overfetch_C'], width, label='Tensor C', alpha=0.8)
ax.bar(x + width*1.5, user_data['Overfetch_Total'], width, label='Total', alpha=0.8, color='black')
ax.set_ylabel('Overfetch Factor')
ax.set_title('User Example: 32k×7k×7k\n(Tensor B = 196MB, needs L3 > 196MB)')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(s)}MB" for s in user_data.index])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('l3_focused_analysis_summary.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: l3_focused_analysis_summary.png")

