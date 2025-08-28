#!/usr/bin/env python3
"""
KPU core module features - numpy info
"""
import stillwater_kpu as kpu
import numpy as np

if __name__ == "__main__":
    print("🔍 KPU NumPy Configuration")
    print("=" * 50)
    kpu.core.show_np_config()