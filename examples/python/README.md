# KPU Python Examples

This directory contains Python examples demonstrating the Stillwater KPU simulator.

## Basic Examples (`basic/`)
- `simple_kpu.py` - First steps with KPU
- `core_features_demo.py` - Core module utilities
- `system_info_demo.py` - System diagnostics
- `timing_demo.py` - Performance measurement

## Educational Examples (`educational/`)
- `neural_network_demo.py` - Neural network layer computation
- `performance_analysis.py` - Scaling analysis
- `blocked_matmul.py` - Memory-efficient algorithms

## Running Examples

```bash
cd examples/python/basic
python simple_kpu.py
python core_features_demo.py

## Recommended location

Recommended directory structure for demos, tools, and educational examples.

```txt

examples/
├── python/
│   ├── basic/                    # ← Your simple usage examples
│   │   ├── simple_kpu.py        # Your current example
│   │   ├── system_info_demo.py  # System diagnostics example
│   │   ├── matrix_utilities.py  # Matrix helper functions demo
│   │   ├── timing_demo.py       # Performance timing examples
│   │   └── memory_planning.py   # Memory usage planning
│   ├── educational/              # More complex learning examples
│   │   ├── neural_network_demo.py
│   │   ├── blocked_matmul.py
│   │   └── performance_analysis.py
│   ├── advanced/                 # Advanced usage patterns
│   │   ├── custom_algorithms.py
│   │   └── multi_device.py
│   └── notebooks/                # Jupyter notebooks (if you use them)
│       ├── getting_started.ipynb
│       └── performance_guide.ipynb
```