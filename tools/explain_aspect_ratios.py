#!/usr/bin/env python3
"""
Explain aspect ratio terminology and investigate overfetch asymmetry
"""

import pandas as pd

df = pd.read_csv('l3_focused_analysis.csv')

print("=" * 80)
print("MATRIX MULTIPLICATION ASPECT RATIO TERMINOLOGY")
print("=" * 80)
print()
print("Given C = A × B where:")
print("  A is M×K")
print("  B is K×N")
print("  C is M×N")
print()
print("Terminology:")
print("  TALL:      M >> N, K (large batch dimension)")
print("  WIDE:      N >> M, K (large output/vocab dimension)")
print("  DEEP:      K >> M, N (large hidden/reduction dimension)")
print("  SQUARE:    M ≈ N ≈ K (balanced dimensions)")
print("  TALL-WIDE: M >> K, N >> K (both M and N large, K small)")
print()

print("=" * 80)
print("ACTUAL WORKLOADS IN DATASET")
print("=" * 80)
print()

# Get one instance of each workload (strategy and L3 don't matter for shape)
workloads = df.groupby('Workload_Name').first()[['Aspect', 'M', 'N', 'K', 'Tensor_A_MB', 'Tensor_B_MB', 'Tensor_C_MB']]
workloads = workloads.sort_values('Aspect')

for aspect in workloads['Aspect'].unique():
    print(f"\n{aspect}:")
    print("-" * 80)
    aspect_wls = workloads[workloads['Aspect'] == aspect]
    for name, row in aspect_wls.iterrows():
        print(f"  {name:40s} {row['M']:6d}×{row['K']:6d}  ×  {row['K']:6d}×{row['N']:6d}  →  {row['M']:6d}×{row['N']:6d}")
        print(f"  {'':40s} A={row['Tensor_A_MB']:6.0f}MB  B={row['Tensor_B_MB']:6.0f}MB  C={row['Tensor_C_MB']:6.0f}MB")

print("\n" + "=" * 80)
print("UNDERSTANDING THE ASYMMETRY")
print("=" * 80)
print()
print("You correctly identified that we SHOULD be able to flip the stationary tensor.")
print("Let's investigate why Tall vs Wide vs Deep show different overfetch:")
print()

# Compare specific workloads
print("Example 1: Tall vs Wide")
print("-" * 80)
tall_example = df[(df['Workload_Name'] == 'Tall: Long Context') & (df['L3_Size_MB'] == 64)].iloc[0]
wide_example = df[(df['Workload_Name'] == 'Wide: Vocab Projection') & (df['L3_Size_MB'] == 64)].iloc[0]

print(f"\nTall: {tall_example['M']}×{tall_example['N']}×{tall_example['K']}")
print(f"  A={tall_example['M']}×{tall_example['K']}={tall_example['Tensor_A_MB']:.0f}MB")
print(f"  B={tall_example['K']}×{tall_example['N']}={tall_example['Tensor_B_MB']:.0f}MB")  
print(f"  C={tall_example['M']}×{tall_example['N']}={tall_example['Tensor_C_MB']:.0f}MB")
print(f"  Overfetch: A={tall_example['Overfetch_A']:.2f}×, B={tall_example['Overfetch_B']:.2f}×, C={tall_example['Overfetch_C']:.2f}×")
print(f"  Total: {tall_example['Overfetch_Total']:.2f}×")

print(f"\nWide: {wide_example['M']}×{wide_example['N']}×{wide_example['K']}")
print(f"  A={wide_example['M']}×{wide_example['K']}={wide_example['Tensor_A_MB']:.0f}MB")
print(f"  B={wide_example['K']}×{wide_example['N']}={wide_example['Tensor_B_MB']:.0f}MB")
print(f"  C={wide_example['M']}×{wide_example['N']}={wide_example['Tensor_C_MB']:.0f}MB")
print(f"  Overfetch: A={wide_example['Overfetch_A']:.2f}×, B={wide_example['Overfetch_B']:.2f}×, C={wide_example['Overfetch_C']:.2f}×")
print(f"  Total: {wide_example['Overfetch_Total']:.2f}×")

print("\n" + "=" * 80)
print("ROOT CAUSE HYPOTHESIS")
print("=" * 80)
print()
print("The asymmetry exists because:")
print()
print("1. Our L3 scheduler is NOT strategy-aware (yet)")
print("   - It simulates the L2 tile execution order")
print("   - But L2 scheduler doesn't vary by dataflow strategy")
print("   - All strategies (WS/IS/OS) produce same L2 tile schedule")
print()
print("2. The L2 tile order is OUTPUT-STATIONARY-LIKE")
print("   - For each output tile C[i,j]:")
print("     - Load C tile once")
print("     - Loop over K: load A[i,k] and B[k,j]")
print("   - This means:")
print("     - C tiles loaded M_tiles × N_tiles times")
print("     - A tiles loaded M_tiles × K_tiles times") 
print("     - B tiles loaded N_tiles × K_tiles times")
print()
print("3. Tensor B overfetch dominates when K_tiles is large")
print("   - Deep (K large): Many K tiles → B loaded many times")
print("   - Wide (N large): Many N tiles → B loaded many times")
print("   - Tall (M large): Many M tiles → A loaded many times (not B!)")
print()
print("4. Why Tall-Wide is worst:")
print("   - BOTH M and N are large")
print("   - M_tiles × N_tiles is huge (many output tiles)")
print("   - For each output tile, we scan through K")
print("   - B is reloaded M_tiles times (once per M tile)")
print()

# Let's verify this with the user's Tall-Wide example
print("Example: User's 32k×7k×7k (Tall-Wide)")
print("-" * 80)
user_example = df[(df['Workload_Name'] == 'User Example 32k×7k') & (df['L3_Size_MB'] == 64)].iloc[0]
print(f"\nShape: {user_example['M']}×{user_example['N']}×{user_example['K']}")
print(f"  A={user_example['M']}×{user_example['K']}={user_example['Tensor_A_MB']:.0f}MB (activation/input)")
print(f"  B={user_example['K']}×{user_example['N']}={user_example['Tensor_B_MB']:.0f}MB (WEIGHTS)")
print(f"  C={user_example['M']}×{user_example['N']}={user_example['Tensor_C_MB']:.0f}MB (output)")
print(f"\nWith output-stationary-like scheduling:")
print(f"  M_tiles × N_tiles output tiles need to be computed")
print(f"  For EACH output tile C[i,j], we need B[k,j] across all K")
print(f"  But B doesn't fit in L3 (196MB >> 64MB)")
print(f"  So B is evicted and reloaded M_tiles times")
print(f"\nResult:")
print(f"  Overfetch: A={user_example['Overfetch_A']:.2f}×, B={user_example['Overfetch_B']:.2f}×, C={user_example['Overfetch_C']:.2f}×")
print(f"  Total: {user_example['Overfetch_Total']:.2f}×")
print(f"  B overfetch is {user_example['Overfetch_B']:.0f}× !!")

print("\n" + "=" * 80)
print("WHY ISN'T IT SYMMETRIC?")
print("=" * 80)
print()
print("You're right that with proper strategy selection, we SHOULD be able to")
print("make Tall and Wide symmetric by choosing:")
print("  - Tall (M large): Use OUTPUT-STATIONARY to keep C in L3")
print("  - Wide (N large): Use INPUT-STATIONARY to keep A in L3")
print("  - Deep (K large): Use WEIGHT-STATIONARY to keep B in L3")
print()
print("BUT: Our current L3 scheduler doesn't do this!")
print("  - It uses the same L2 tile order for all strategies")
print("  - The L2 schedule is output-stationary-like by default")
print("  - This is why TALL matrices do well (C is small)")
print("  - But WIDE matrices do poorly (B is large and thrashes)")
print()
print("TO FIX THIS: We need strategy-aware L2 tile scheduling")
print("  - WS: Order tiles to maximize B reuse")
print("  - IS: Order tiles to maximize A reuse")  
print("  - OS: Order tiles to maximize C reuse")
print()
print("This would make the analysis symmetric and strategy-dependent!")

