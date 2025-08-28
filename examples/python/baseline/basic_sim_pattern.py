import stillwater_kpu as kpu

# System info
kpu.core.print_system_info()

# Generate test matrices
A, B = kpu.core.generate_test_matrices(100, 200, 150, seed=42)

# Benchmark with timing
with kpu.core.Timer() as timer:
    C = A @ B
print(f"NumPy took {timer.elapsed_ms:.2f} ms")

# Memory planning
memory_info = kpu.core.estimate_memory_usage([A.shape, B.shape, C.shape])
print(f"Total memory needed: {memory_info['human_readable']}")

# Validate scratchpad capacity
validation = kpu.core.validate_scratchpad_capacity(
    [A.shape, B.shape, (100, 150)], 
    scratchpad_size=1024*1024  # 1MB
)
print(f"Matrices fit: {validation['fits']}")