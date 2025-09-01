def generate_loop_config(M, K, N, T, strategy):
    roles = {
        "result-stationary": ("C", "A", "B"),
        "input-stationary": ("A", "B", "C"),
        "weight-stationary": ("B", "A", "C")
    }
    stationary, cached, streamed = roles[strategy]
    return {
        "M": M, "K": K, "N": N, "T": T,
        "strategy": strategy,
        "stationary": stationary,
        "cached": cached,
        "streamed": streamed,
        "tile_grid": {
            "MT": M // T,
            "KT": K // T,
            "NT": N // T
        }
    }