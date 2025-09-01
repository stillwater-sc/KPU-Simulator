import numpy as np


def block_matmul(config):
    M, K, N, T = config["M"], config["K"], config["N"], config["T"]
    MT, KT, NT = config["tile_grid"]["MT"], config["tile_grid"]["KT"], config["tile_grid"]["NT"]

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    for i in range(MT):
        for j in range(NT):
            C_tile = np.zeros((T, T), dtype=np.float32)
            for k in range(KT):
                A_tile = A[i*T:(i+1)*T, k*T:(k+1)*T]
                B_tile = B[k*T:(k+1)*T, j*T:(j+1)*T]
                C_tile += A_tile @ B_tile
            C[i*T:(i+1)*T, j*T:(j+1)*T] = C_tile

    # Validate against full matmul
    C_ref = A @ B
    error = np.max(np.abs(C - C_ref))
    print(f"Max error vs reference: {error:.6f}")

class BandwidthTracker:
    def __init__(self):
        self.counters = {
            'DRAM_to_L3': 0,
            'L3_to_L2': 0,
            'L2_to_L1': 0,
            'C_evict': 0,
            'reuse_hits': 0
        }

    def log(self, path, bytes):
        self.counters[path] += bytes

    def reuse(self):
        self.counters['reuse_hits'] += 1

    def report(self):
        print("\n--- Bandwidth Report ---")
        for k, v in self.counters.items():
            gb = v / (1024**3)
            print(f"{k:<15}: {v/1024:.2f} KB ({gb:.2f} GB)")

def init_tile_tracker(rows, cols):
    return np.zeros((rows, cols), dtype=np.int32)

def record_access(tracker, i, j):
    tracker[i, j] += 1

def simulate_matmul(config):
    """
    Simulate tiled matrix multiplication with memory movement tracking.
    
    Args:
        config (dict): Loop configuration dictionary.
    """
    M, K, N, T = config["M"], config["K"], config["N"], config["T"]
    MT, KT, NT = config["tile_grid"]["MT"], config["tile_grid"]["KT"], config["tile_grid"]["NT"]
    strategy = config["strategy"]
    stationary, cached, streamed = config["stationary"], config["cached"], config["streamed"]

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    # set up tile access tracker
    A_tracker = init_tile_tracker(M // T, K // T)
    B_tracker = init_tile_tracker(K // T, N // T)
    C_tracker = init_tile_tracker(M // T, N // T)

    tracker = BandwidthTracker()
    tile_bytes = T * T * 4

    cached_tiles = {}

    for i in range(MT):
        for j in range(NT):
            C_tile = np.zeros((T, T), dtype=np.float32)

            for k in range(KT):
                a_tile = A[i*T:(i+1)*T, k*T:(k+1)*T]
                b_tile = B[k*T:(k+1)*T, j*T:(j+1)*T]

                record_access(A_tracker, i, k)
                record_access(B_tracker, k, j)


                # Simulate memory movement
                tracker.log('DRAM_to_L3', tile_bytes)
                tracker.log('L3_to_L2', tile_bytes)
                tracker.log('L2_to_L1', tile_bytes)

                # Simulate reuse
                a_key = (i, k)
                if a_key in cached_tiles:
                    tracker.reuse()
                else:
                    cached_tiles[a_key] = True

                C_tile += a_tile @ b_tile

            # Evict C tile if result-stationary
            if strategy == 'result-stationary':
                tracker.log('C_evict', tile_bytes)

            C[i*T:(i+1)*T, j*T:(j+1)*T] = C_tile
            record_access(C_tracker, i, j)

    tracker.report()

    # Validate result
    C_ref = A @ B
    error = np.max(np.abs(C - C_ref))
    print(f"\nMax error vs reference: {error:.6f}")

    return A_tracker, B_tracker, C_tracker, tracker  # reuse heatmaps + bandwidth tracker
