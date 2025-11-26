#!/usr/bin/env python3
"""
Analyze the comprehensive L3 overfetch data
Adapted from analyze_l3_focused.py for the larger comprehensive experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Determine CSV path
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = 'l3_comprehensive_analysis.csv'

if not os.path.exists(csv_path):
    # Try build directory
    csv_path = os.path.join(os.path.dirname(__file__), '../../build/l3_comprehensive_analysis.csv')

print(f"Reading: {csv_path}")
df = pd.read_csv(csv_path)

# Create a comprehensive summary
print("=" * 80)
print("L3 COMPREHENSIVE ANALYSIS - KEY FINDINGS")
print("=" * 80)
print(f"\nTotal configurations analyzed: {len(df)}")
print(f"Unique workloads: {df['Workload_Name'].nunique()}")
print(f"L3 sizes tested: {sorted(df['L3_Size_MB'].unique())}")
print(f"Strategies tested: {sorted(df['Strategy'].unique())}")

# Finding 1: Dataflow strategy impact
print("\n1. DATAFLOW STRATEGY IMPACT")
print("-" * 80)
strategy_stats = df.groupby('Strategy')['Overfetch_Total'].agg(['mean', 'std', 'min', 'max'])
print(strategy_stats)
best_strategy = strategy_stats['mean'].idxmin()
worst_strategy = strategy_stats['mean'].idxmax()
print(f"\n   → Best strategy on average: {best_strategy} ({strategy_stats.loc[best_strategy, 'mean']:.2f}× avg)")
print(f"   → Worst strategy on average: {worst_strategy} ({strategy_stats.loc[worst_strategy, 'mean']:.2f}× avg)")

# Finding 2: Aspect ratio impact
print("\n2. ASPECT RATIO IMPACT")
print("-" * 80)
aspect_stats = df.groupby('Aspect').agg({
    'Overfetch_Total': ['mean', 'std', 'min', 'max'],
    'Workload_Name': 'nunique'
}).round(2)
aspect_stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Workloads']
aspect_stats = aspect_stats.sort_values('Mean')
print(aspect_stats)
print(f"\n   → Best aspect ratio: {aspect_stats.index[0]} ({aspect_stats.iloc[0]['Mean']:.2f}× avg)")
print(f"   → Worst aspect ratio: {aspect_stats.index[-1]} ({aspect_stats.iloc[-1]['Mean']:.2f}× avg)")

# Finding 3: Category impact
print("\n3. CATEGORY IMPACT")
print("-" * 80)
category_stats = df.groupby('Category').agg({
    'Overfetch_Total': ['mean', 'std', 'min', 'max'],
    'Workload_Name': 'nunique'
}).round(2)
category_stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Workloads']
category_stats = category_stats.sort_values('Mean')
print(category_stats)

# Finding 4: L3 size impact
print("\n4. L3 SIZE IMPACT (Knee of the Curve)")
print("-" * 80)
l3_stats = df.groupby('L3_Size_MB').agg({
    'Overfetch_Total': ['mean', 'std', 'min', 'max'],
    'L3_Hit_Rate': 'mean',
    'DRAM_Reads_MB': 'mean'
}).round(2)
print(l3_stats)

# Find the knee
l3_means = df.groupby('L3_Size_MB')['Overfetch_Total'].mean()
knee_candidates = []
for i, size in enumerate(l3_means.index[:-1]):
    improvement = l3_means.iloc[i] - l3_means.iloc[i+1]
    knee_candidates.append((size, improvement))
knee_candidates.sort(key=lambda x: -x[1])
print(f"\n   → Biggest improvement step: {knee_candidates[0][0]}MB → next size ({knee_candidates[0][1]:.2f}× reduction)")

# Finding 5: Strategy × L3 Size interaction
print("\n5. STRATEGY × L3 SIZE INTERACTION")
print("-" * 80)
pivot = df.pivot_table(values='Overfetch_Total', index='L3_Size_MB', columns='Strategy', aggfunc='mean').round(2)
print(pivot)

# Finding 6: Per-tensor breakdown
print("\n6. PER-TENSOR BREAKDOWN BY ASPECT")
print("-" * 80)
tensor_breakdown = df.groupby('Aspect').agg({
    'Overfetch_A': 'mean',
    'Overfetch_B': 'mean',
    'Overfetch_C': 'mean'
}).round(2)
tensor_breakdown = tensor_breakdown.sort_values('Overfetch_B')
print(tensor_breakdown)

# Finding 7: Best configurations
print("\n7. BEST CONFIGURATIONS (Overfetch ≤ 1.5×)")
print("-" * 80)
good_configs = df[df['Overfetch_Total'] <= 1.5].groupby(['Workload_Name', 'L3_Size_MB', 'Strategy']).first().reset_index()
if len(good_configs) > 0:
    print(f"Found {len(good_configs)} configurations with overfetch ≤ 1.5×")
    summary = good_configs.groupby('L3_Size_MB').agg({'Workload_Name': 'nunique'})
    print(f"\nWorkloads achieving ≤1.5× by L3 size:")
    print(summary)
else:
    print("No configurations achieved overfetch ≤ 1.5×")

# Finding 8: Worst cases
print("\n8. WORST CASES (Overfetch > 50×)")
print("-" * 80)
worst_cases = df[df['Overfetch_Total'] > 50][['Workload_Name', 'Strategy', 'L3_Size_MB', 'Overfetch_Total', 'Tensor_B_MB']].sort_values('Overfetch_Total', ascending=False)
if len(worst_cases) > 0:
    print(f"Found {len(worst_cases)} configurations with overfetch > 50×")
    print(worst_cases.head(20).to_string(index=False))
else:
    print("No configurations had overfetch > 50×")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print(f"1. Prefer {best_strategy} dataflow strategy (lowest average overfetch)")
print(f"2. Avoid {worst_strategy} for large workloads (highest average overfetch)")
print(f"3. Knee of L3 curve near {knee_candidates[0][0]}MB for most workloads")
print("4. Tall/Extreme Tall matrices are most L3-efficient")
print("5. Very Deep and large Square matrices need largest L3 or alternative strategies")
print("=" * 80)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle('L3 Comprehensive Analysis: Overview', fontsize=16, fontweight='bold')

# Plot 1: Overfetch by aspect ratio (box plot)
ax1 = fig.add_subplot(2, 3, 1)
aspect_order = df.groupby('Aspect')['Overfetch_Total'].mean().sort_values().index
df_sorted = df.copy()
df_sorted['Aspect'] = pd.Categorical(df_sorted['Aspect'], categories=aspect_order, ordered=True)
df_sorted.boxplot(column='Overfetch_Total', by='Aspect', ax=ax1, rot=45)
ax1.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Ideal (1.0×)')
ax1.set_xlabel('Aspect Ratio')
ax1.set_ylabel('Overfetch Factor')
ax1.set_title('Overfetch by Aspect Ratio')
plt.suptitle('')  # Remove automatic title
ax1.set_yscale('log')

# Plot 2: Overfetch by strategy (box plot)
ax2 = fig.add_subplot(2, 3, 2)
df.boxplot(column='Overfetch_Total', by='Strategy', ax=ax2)
ax2.axhline(y=1, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Dataflow Strategy')
ax2.set_ylabel('Overfetch Factor')
ax2.set_title('Overfetch by Strategy')
plt.suptitle('')
ax2.set_yscale('log')

# Plot 3: L3 size vs overfetch by strategy
ax3 = fig.add_subplot(2, 3, 3)
for strategy in df['Strategy'].unique():
    data = df[df['Strategy'] == strategy].groupby('L3_Size_MB')['Overfetch_Total'].mean()
    ax3.plot(data.index, data.values, 'o-', linewidth=2, markersize=6, label=strategy)
ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Ideal (1.0×)')
ax3.set_xlabel('L3 Cache Size (MB)')
ax3.set_ylabel('Average Overfetch Factor')
ax3.set_title('L3 Size vs Overfetch by Strategy')
ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Overfetch by category
ax4 = fig.add_subplot(2, 3, 4)
category_order = df.groupby('Category')['Overfetch_Total'].mean().sort_values().index
category_means = df.groupby('Category')['Overfetch_Total'].mean()[category_order]
colors = ['green' if x < 2 else 'orange' if x < 10 else 'red' for x in category_means.values]
category_means.plot(kind='barh', ax=ax4, color=colors)
ax4.axvline(x=1, color='gray', linestyle='--', linewidth=1)
ax4.set_xlabel('Average Overfetch Factor')
ax4.set_title('Overfetch by Workload Category')
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Per-tensor breakdown heatmap
ax5 = fig.add_subplot(2, 3, 5)
tensor_pivot = df.groupby('Aspect')[['Overfetch_A', 'Overfetch_B', 'Overfetch_C']].mean()
tensor_pivot = tensor_pivot.loc[aspect_order]  # Sort by total overfetch
im = ax5.imshow(np.log10(tensor_pivot.values + 0.1), aspect='auto', cmap='RdYlGn_r')
ax5.set_xticks(range(3))
ax5.set_xticklabels(['A (input)', 'B (weights)', 'C (output)'])
ax5.set_yticks(range(len(tensor_pivot.index)))
ax5.set_yticklabels(tensor_pivot.index)
ax5.set_title('Per-Tensor Overfetch (log scale)')
plt.colorbar(im, ax=ax5, label='log10(Overfetch)')

# Plot 6: L3 hit rate vs overfetch
ax6 = fig.add_subplot(2, 3, 6)
for strategy, marker in [('WS', 'o'), ('IS', 's'), ('OS', '^')]:
    data = df[df['Strategy'] == strategy]
    ax6.scatter(data['L3_Hit_Rate'], data['Overfetch_Total'], alpha=0.5, marker=marker, label=strategy)
ax6.set_xlabel('L3 Hit Rate')
ax6.set_ylabel('Overfetch Factor')
ax6.set_title('L3 Hit Rate vs Overfetch')
ax6.set_yscale('log')
ax6.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
output_path = csv_path.replace('.csv', '_summary.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nOverview visualization saved to: {output_path}")

# ============================================================================
# DETAILED PLOTS
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('L3 Comprehensive Analysis: Detailed Views', fontsize=16, fontweight='bold')

# Plot 1: Strategy comparison by L3 size (heatmap)
ax = axes[0, 0]
pivot = df.pivot_table(values='Overfetch_Total', index='L3_Size_MB', columns='Strategy', aggfunc='mean')
im = ax.imshow(np.log10(pivot.values + 0.1), aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f'{int(x)}MB' for x in pivot.index])
ax.set_xlabel('Strategy')
ax.set_ylabel('L3 Size')
ax.set_title('Overfetch: Strategy × L3 Size (log scale)')
plt.colorbar(im, ax=ax, label='log10(Overfetch)')
# Annotate with values
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        color = 'white' if val > 10 else 'black'
        ax.text(j, i, f'{val:.1f}×', ha='center', va='center', color=color, fontsize=8)

# Plot 2: L3 curves by aspect ratio
ax = axes[0, 1]
for aspect in ['Tall', 'Square', 'Deep', 'Wide']:
    if aspect in df['Aspect'].values:
        data = df[df['Aspect'] == aspect].groupby('L3_Size_MB')['Overfetch_Total'].mean()
        ax.plot(data.index, data.values, 'o-', linewidth=2, markersize=6, label=aspect)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('L3 Cache Size (MB)')
ax.set_ylabel('Average Overfetch Factor')
ax.set_title('L3 Size vs Overfetch by Aspect (Core Types)')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Transformer workload comparison
ax = axes[1, 0]
transformer_workloads = df[df['Category'].isin(['Attention', 'MLP'])]
if len(transformer_workloads) > 0:
    for category in ['Attention', 'MLP']:
        cat_data = transformer_workloads[transformer_workloads['Category'] == category]
        l3_curve = cat_data.groupby('L3_Size_MB')['Overfetch_Total'].mean()
        ax.plot(l3_curve.index, l3_curve.values, 'o-', linewidth=2, markersize=8, label=category)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('L3 Cache Size (MB)')
    ax.set_ylabel('Average Overfetch Factor')
    ax.set_title('Transformer Workloads: Attention vs MLP')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 4: Per-workload overfetch at different L3 sizes
ax = axes[1, 1]
l3_sizes_to_show = [16, 64, 256]
available_sizes = [s for s in l3_sizes_to_show if s in df['L3_Size_MB'].values]
if len(available_sizes) >= 2:
    workload_means = df[df['L3_Size_MB'].isin(available_sizes)].pivot_table(
        values='Overfetch_Total',
        index='Workload_Name',
        columns='L3_Size_MB',
        aggfunc='mean'
    )
    workload_means = workload_means.sort_values(available_sizes[0], ascending=False).head(15)
    x = np.arange(len(workload_means.index))
    width = 0.25
    for i, size in enumerate(available_sizes):
        if size in workload_means.columns:
            ax.barh(x + i*width, workload_means[size], width, label=f'{size}MB', alpha=0.8)
    ax.set_yticks(x + width)
    ax.set_yticklabels(workload_means.index, fontsize=8)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Overfetch Factor')
    ax.set_title('Top 15 Workloads by Overfetch @ Different L3 Sizes')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_path2 = csv_path.replace('.csv', '_detailed.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Detailed visualization saved to: {output_path2}")

print("\nAnalysis complete!")
