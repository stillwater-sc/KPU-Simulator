
def analyze_execution(M, K, N, T):
    tile_bytes = T * T * 4  # 4 bytes per float
    num_tiles_A = (M // T) * (K // T)
    num_tiles_B = (K // T) * (N // T)
    num_tiles_C = (M // T) * (N // T)

    strategies = ['result-stationary', 'input-stationary', 'weight-stationary']
    results = []

    for strategy in strategies:
        if strategy == 'result-stationary':
            stationary = 'C'
            cached = 'A'
            streamed = 'B'
            # C tiles loaded once, updated 224 times, then unloaded
            mem_stationary = tile_bytes  # 1 tile per PE
            bw_stationary = num_tiles_C * tile_bytes * 2  # load + unload
            bw_cached = num_tiles_A * tile_bytes  # reused across N
            bw_streamed = num_tiles_B * tile_bytes * (M // T)  # fetched per output row
        elif strategy == 'input-stationary':
            stationary = 'A'
            cached = 'B'
            streamed = 'C'
            mem_stationary = (K // T) * tile_bytes  # full row of A tiles
            bw_stationary = num_tiles_A * tile_bytes  # load once
            bw_cached = num_tiles_B * tile_bytes  # reused across M
            bw_streamed = num_tiles_C * tile_bytes  # fetched per output tile
        elif strategy == 'weight-stationary':
            stationary = 'B'
            cached = 'A'
            streamed = 'C'
            mem_stationary = (K // T) * tile_bytes  # full column of B tiles
            bw_stationary = num_tiles_B * tile_bytes  # load once
            bw_cached = num_tiles_A * tile_bytes  # reused across N
            bw_streamed = num_tiles_C * tile_bytes  # fetched per output tile

        total_bw = bw_stationary + bw_cached + bw_streamed
        results.append({
            'Strategy': strategy,
            'Stationary': stationary,
            'Cached': cached,
            'Streamed': streamed,
            'Memory_KB': round(mem_stationary / 1024),
            'Bandwidth_GB': round(total_bw / (1024**3))
        })

    return results