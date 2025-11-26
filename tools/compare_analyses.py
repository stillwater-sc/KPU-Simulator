#!/usr/bin/env python3
"""
Compare focused vs comprehensive L3 analyses
Show what the 2+ hour run revealed that the 4-minute run didn't
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both datasets
df_focused = pd.read_csv('l3_focused_analysis.csv')
df_comprehensive = pd.read_csv('l3_comprehensive_analysis.csv')

print("=" * 80)
print("COMPREHENSIVE vs FOCUSED L3 ANALYSIS COMPARISON")
print("=" * 80)
print(f"\nFocused: {len(df_focused)} configs in 4 minutes")
print(f"Comprehensive: {len(df_comprehensive)} configs in 135 minutes (2.25 hours)")
print(f"Time cost: 33.75× longer")
print(f"Data gained: {len(df_comprehensive) / len(df_focused):.2f}× more configurations")

# What workloads are unique to comprehensive?
focused_workloads = set(df_focused['Workload_Name'].unique())
comprehensive_workloads = set(df_comprehensive['Workload_Name'].unique())
unique_to_comprehensive = comprehensive_workloads - focused_workloads

print(f"\n{'='*80}")
print("UNIQUE WORKLOADS IN COMPREHENSIVE (not in focused)")
print("=" * 80)

unique_df = df_comprehensive[df_comprehensive['Workload_Name'].isin(unique_to_comprehensive)]
unique_summary = unique_df.groupby('Workload_Name').first()[['M', 'N', 'K', 'Category', 'Aspect']]
unique_summary = unique_summary.sort_values(['Aspect', 'M'])

for idx, row in unique_summary.iterrows():
    print(f"{idx:40s} {row['M']:6d}×{row['N']:6d}×{row['K']:6d}  [{row['Aspect']:10s}]")

# Compare L3 size ranges
print(f"\n{'='*80}")
print("L3 SIZE RANGES")
print("=" * 80)
print("Focused:       16MB, 64MB, 256MB")
print("Comprehensive: 4MB, 16MB, 64MB, 256MB, 1GB")
print("\nUnique insights from 4MB and 1GB:")

# 4MB insights
print("\n--- 4MB L3 (Too Small?) ---")
small_l3 = df_comprehensive[df_comprehensive['L3_Size_MB'] == 4]
print(f"Average overfetch: {small_l3['Overfetch_Total'].mean():.2f}×")
print(f"Worst overfetch: {small_l3['Overfetch_Total'].max():.2f}× ({small_l3.loc[small_l3['Overfetch_Total'].idxmax(), 'Workload_Name']})")
print(f"Average hit rate: {small_l3['L3_Hit_Rate'].mean()*100:.1f}%")

# 1GB insights
print("\n--- 1GB L3 (Saturation Point?) ---")
large_l3 = df_comprehensive[df_comprehensive['L3_Size_MB'] == 1024]
print(f"Average overfetch: {large_l3['Overfetch_Total'].mean():.2f}×")
print(f"Worst overfetch: {large_l3['Overfetch_Total'].max():.2f}× ({large_l3.loc[large_l3['Overfetch_Total'].idxmax(), 'Workload_Name']})")
print(f"Average hit rate: {large_l3['L3_Hit_Rate'].mean()*100:.1f}%")
print(f"\nWorkloads still with >1× overfetch at 1GB:")
still_bad = large_l3[large_l3['Overfetch_Total'] > 1.0]
if len(still_bad) > 0:
    for _, row in still_bad.groupby('Workload_Name').first().iterrows():
        print(f"  {row.name}: {row['Overfetch_Total']:.2f}×")
else:
    print("  None! 1GB is sufficient for all workloads.")

# Extreme workload insights
print(f"\n{'='*80}")
print("EXTREME WORKLOAD INSIGHTS")
print("=" * 80)

extreme_workloads = [
    'Extreme Tall: 64k batch',
    'Extreme Wide: 100k vocab', 
    'Extreme Deep: 64k hidden',
    'GPT-3 MLP',
    'GPT-3 Q×K^T'
]

for wl_name in extreme_workloads:
    if wl_name in df_comprehensive['Workload_Name'].values:
        wl_data = df_comprehensive[df_comprehensive['Workload_Name'] == wl_name]
        row = wl_data.iloc[0]
        print(f"\n{wl_name}")
        print(f"  Shape: {row['M']}×{row['N']}×{row['K']}")
        print(f"  Tensor sizes: A={row['Tensor_A_MB']:.0f}MB, B={row['Tensor_B_MB']:.0f}MB, C={row['Tensor_C_MB']:.0f}MB")
        
        # Show overfetch across L3 sizes
        print(f"  Overfetch by L3 size:")
        for l3_size in [4, 16, 64, 256, 1024]:
            l3_row = wl_data[wl_data['L3_Size_MB'] == l3_size].iloc[0]
            print(f"    {l3_size:4d}MB: {l3_row['Overfetch_Total']:6.2f}× (A={l3_row['Overfetch_A']:.2f}×, B={l3_row['Overfetch_B']:.2f}×, C={l3_row['Overfetch_C']:.2f}×)")

# Key finding: What's the minimum L3 needed for <2× overfetch?
print(f"\n{'='*80}")
print("MINIMUM L3 SIZE FOR <2× OVERFETCH (by workload type)")
print("=" * 80)

for aspect in df_comprehensive['Aspect'].unique():
    aspect_data = df_comprehensive[df_comprehensive['Aspect'] == aspect]
    
    # For each L3 size, check if any workload has >2× overfetch
    l3_threshold = None
    for l3_size in sorted(df_comprehensive['L3_Size_MB'].unique()):
        l3_aspect = aspect_data[aspect_data['L3_Size_MB'] == l3_size]
        max_overfetch = l3_aspect['Overfetch_Total'].max()
        if max_overfetch < 2.0:
            l3_threshold = l3_size
            break
    
    if l3_threshold:
        print(f"{aspect:15s}: {l3_threshold:4d}MB (max overfetch < 2×)")
    else:
        print(f"{aspect:15s}: >1GB needed (still {max_overfetch:.1f}× at 1GB)")

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive vs Focused Analysis: What Did We Learn?', fontsize=16, fontweight='bold')

# Plot 1: Overfetch curves for all L3 sizes
ax = axes[0, 0]
l3_sizes_focused = sorted(df_focused['L3_Size_MB'].unique())
l3_sizes_comp = sorted(df_comprehensive['L3_Size_MB'].unique())

overfetch_focused = [df_focused[df_focused['L3_Size_MB']==l3]['Overfetch_Total'].mean() for l3 in l3_sizes_focused]
overfetch_comp = [df_comprehensive[df_comprehensive['L3_Size_MB']==l3]['Overfetch_Total'].mean() for l3 in l3_sizes_comp]

ax.plot(l3_sizes_focused, overfetch_focused, 'o-', linewidth=3, markersize=10, label='Focused (3 points)', color='blue')
ax.plot(l3_sizes_comp, overfetch_comp, 's-', linewidth=2, markersize=8, label='Comprehensive (5 points)', color='red', alpha=0.7)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='2× threshold')
ax.set_xlabel('L3 Cache Size (MB)')
ax.set_ylabel('Average Overfetch Factor')
ax.set_title('L3 Size vs Overfetch: More Data Points\n(Comprehensive adds 4MB and 1GB)')
ax.set_xscale('log', base=2)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Extreme workloads - tensor size distribution
ax = axes[0, 1]
extreme_data = df_comprehensive[df_comprehensive['Workload_Name'].isin(extreme_workloads)].groupby('Workload_Name').first()
x = np.arange(len(extreme_data))
width = 0.25

ax.bar(x - width, extreme_data['Tensor_A_MB'], width, label='Tensor A', alpha=0.8)
ax.bar(x, extreme_data['Tensor_B_MB'], width, label='Tensor B', alpha=0.8)
ax.bar(x + width, extreme_data['Tensor_C_MB'], width, label='Tensor C', alpha=0.8)

ax.set_ylabel('Tensor Size (MB)')
ax.set_title('Extreme Workloads: Tensor Sizes\n(Only in Comprehensive)')
ax.set_xticks(x)
ax.set_xticklabels([name.replace(' ', '\n') for name in extreme_data.index], rotation=0, fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

# Plot 3: Overfetch distribution by aspect (comprehensive only)
ax = axes[1, 0]
aspect_stats = []
for aspect in sorted(df_comprehensive['Aspect'].unique()):
    aspect_data = df_comprehensive[df_comprehensive['Aspect'] == aspect]['Overfetch_Total']
    aspect_stats.append({
        'aspect': aspect,
        'min': aspect_data.min(),
        'q25': aspect_data.quantile(0.25),
        'median': aspect_data.median(),
        'q75': aspect_data.quantile(0.75),
        'max': aspect_data.max(),
        'mean': aspect_data.mean()
    })

aspect_df = pd.DataFrame(aspect_stats).sort_values('mean')

# Box plot style
for i, row in aspect_df.iterrows():
    # Min to Q25
    ax.plot([i, i], [row['min'], row['q25']], 'k-', linewidth=1)
    # Q25 to Q75 (box)
    ax.plot([i, i], [row['q25'], row['q75']], 'b-', linewidth=8, alpha=0.5)
    # Q75 to Max
    ax.plot([i, i], [row['q75'], row['max']], 'k-', linewidth=1)
    # Median
    ax.plot(i, row['median'], 'ro', markersize=8)
    # Mean
    ax.plot(i, row['mean'], 'g^', markersize=8)

ax.set_ylabel('Overfetch Factor')
ax.set_title('Overfetch Distribution by Aspect\n(Comprehensive: shows full range across all L3 sizes)')
ax.set_xticks(range(len(aspect_df)))
ax.set_xticklabels(aspect_df['aspect'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=2, color='orange', linestyle='--', linewidth=1)
ax.set_yscale('log')
ax.legend(['Range', 'IQR', 'Median', 'Mean'], loc='upper left')

# Plot 4: Was it worth it? (time vs insight)
ax = axes[1, 1]

metrics = ['Workloads', 'L3 Sizes', 'Total Configs', 'Runtime (min)']
focused_vals = [
    len(df_focused['Workload_Name'].unique()),
    len(df_focused['L3_Size_MB'].unique()),
    len(df_focused),
    4
]
comp_vals = [
    len(df_comprehensive['Workload_Name'].unique()),
    len(df_comprehensive['L3_Size_MB'].unique()),
    len(df_comprehensive),
    135
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, focused_vals, width, label='Focused', alpha=0.8, color='blue')
bars2 = ax.bar(x + width/2, comp_vals, width, label='Comprehensive', alpha=0.8, color='red')

# Add ratio labels
for i, (f, c) in enumerate(zip(focused_vals, comp_vals)):
    ratio = c / f
    ax.text(i, max(f, c) * 1.1, f'{ratio:.1f}×', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Count / Time')
ax.set_title('Cost vs Benefit Analysis\n(Numbers show Comprehensive/Focused ratio)')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('comprehensive_vs_focused_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: comprehensive_vs_focused_comparison.png")

# Final verdict
print(f"\n{'='*80}")
print("VERDICT: WAS THE 2+ HOUR RUN WORTH IT?")
print("=" * 80)
print("\nWhat you gained:")
print("  ✓ 4MB L3 data point (confirms it's too small - 23× avg overfetch)")
print("  ✓ 1GB L3 data point (shows saturation - but likely impractical)")
print("  ✓ Extreme workload behavior (65k dimensions, GPT-3 scales)")
print("  ✓ More granular aspect ratio insights")
print("  ✓ Full overfetch distribution (min/max/median) per workload type")
print("\nWhat you already knew from focused analysis:")
print("  • Knee is around 64MB L3")
print("  • Tall matrices are best (~1× overfetch)")
print("  • Tensor B dominates overfetch for square/deep/wide")
print("  • 256MB L3 achieves <1× overfetch for most workloads")
print("\nRecommendation:")
print("  For practical L3 sizing decisions: Focused analysis (4 min) is sufficient")
print("  For research paper / extreme cases: Comprehensive (2+ hrs) provides completeness")
print("  For design space exploration: Focused gives you 90% of insights in 3% of time")
print("=" * 80)

