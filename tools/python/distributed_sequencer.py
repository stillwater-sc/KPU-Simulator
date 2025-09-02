from collections import deque, defaultdict

# Constants
TILE_SIZE = 32
CACHE_LINE_SIZE = 32  # elements
TRANSFER_CYCLES = TILE_SIZE  # 32 cache lines per tile
MATRIX_DIM = 64
TILE_INDICES = [(i, j) for i in range(0, MATRIX_DIM, TILE_SIZE)
                        for j in range(0, MATRIX_DIM, TILE_SIZE)]

# Buffers and credits
l3_buffer = defaultdict(deque)
l2_buffer = defaultdict(deque)
l1_buffer = defaultdict(deque)
fabric_ready = defaultdict(bool)

credits = {
    'L3': 4,  # max tiles
    'L2': 4,
    'L1': 2
}

# Bus occupancy per cycle
bus_dram_l3 = deque()
bus_l3_l2 = deque()
bus_l2_l1 = deque()

# Transaction log
log = []

# Simulation state
cycle = 0
dma_queue = deque([('B', k, j) for k in range(0, MATRIX_DIM, TILE_SIZE)
                                for j in range(0, MATRIX_DIM, TILE_SIZE)])
block_queue = deque([('A', i, k) for i in range(0, MATRIX_DIM, TILE_SIZE)
                                  for k in range(0, MATRIX_DIM, TILE_SIZE)] +
                    [('B', k, j) for k in range(0, MATRIX_DIM, TILE_SIZE)
                                  for j in range(0, MATRIX_DIM, TILE_SIZE)])
stream_queue = deque([(i, j, k) for i in range(0, MATRIX_DIM, TILE_SIZE)
                                 for j in range(0, MATRIX_DIM, TILE_SIZE)
                                 for k in range(0, MATRIX_DIM, TILE_SIZE)])

# Simulation loop
while dma_queue or block_queue or stream_queue or bus_dram_l3 or bus_l3_l2 or bus_l2_l1:
    # DMA Engine: DRAM → L3
    if bus_dram_l3 and bus_dram_l3[0][1] + TRANSFER_CYCLES <= cycle:
        bus_dram_l3.popleft()

    if dma_queue and len(bus_dram_l3) < credits['L3']:
        mat, k, j = dma_queue.popleft()
        tile_id = f"{mat}[{k},{j}]"
        bus_dram_l3.append((tile_id, cycle))
        l3_buffer[tile_id].append(cycle + TRANSFER_CYCLES)
        log.append((cycle, 'DMA', f'DRAM→L3', tile_id))

    # Block Mover: L3 → L2
    if bus_l3_l2 and bus_l3_l2[0][1] + TRANSFER_CYCLES <= cycle:
        bus_l3_l2.popleft()

    if block_queue and len(bus_l3_l2) < credits['L2']:
        mat, i, k_or_j = block_queue.popleft()
        tile_id = f"{mat}[{i},{k_or_j}]"
        if tile_id in l3_buffer and l3_buffer[tile_id][0] <= cycle:
            l3_buffer[tile_id].popleft()
            bus_l3_l2.append((tile_id, cycle))
            l2_buffer[tile_id].append(cycle + TRANSFER_CYCLES)
            log.append((cycle, 'BlockMover', f'L3→L2', tile_id))

    # Streamer: L2 → L1 → Fabric
    if bus_l2_l1 and bus_l2_l1[0][1] + TRANSFER_CYCLES <= cycle:
        bus_l2_l1.popleft()

    if stream_queue and len(bus_l2_l1) < credits['L1']:
        i, j, k = stream_queue.popleft()
        a_id = f"A[{i},{k}]"
        b_id = f"B[{k},{j}]"
        c_id = f"C[{i},{j}]"
        if a_id in l2_buffer and b_id in l2_buffer:
            if l2_buffer[a_id][0] <= cycle and l2_buffer[b_id][0] <= cycle:
                l2_buffer[a_id].popleft()
                l2_buffer[b_id].popleft()
                bus_l2_l1.append((f"{a_id}+{b_id}", cycle))
                fabric_ready[c_id] = True
                log.append((cycle, 'Streamer', f'L2→L1→Fabric', f"{a_id}+{b_id}"))
                log.append((cycle + TRANSFER_CYCLES, 'Compute', f'Fabric', c_id))

    cycle += 1
