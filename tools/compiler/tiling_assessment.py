from kpu_tiling import analyze_execution

# Driver code
if __name__ == "__main__":
    M, K, N = 32768, 7168, 16384  # Matrix dimensions
    T = 32  # Tile size

    results = analyze_execution(M, K, N, T)

    print(f"{'Strategy':<20} {'Stationary':<12} {'Cached':<10} {'Streamed':<10} {'Memory (KB)':<14} {'Bandwidth (GB)':<16}")
    print("-" * 82)
    for r in results:
        print(f"{r['Strategy']:<20} {r['Stationary']:<12} {r['Cached']:<10} {r['Streamed']:<10} {r['Memory_KB']:<14} {r['Bandwidth_GB']:<16}")