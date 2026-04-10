"""Benchmark improved 3D CP-SAT solver."""
from __future__ import annotations
import time
from palletizing import PalletSpec, generate_multitype_task, plan_multitype_palletizing

SEEDS = [7, 42, 99, 123]
COUNT = 70
TIME_LIMIT = 300.0

pallet = PalletSpec(length=1.2, width=1.0, max_height=1.5)

for seed in SEEDS:
    task = generate_multitype_task(count=COUNT, seed=seed)
    print(f"\n--- seed={seed}, {len(task.boxes)} boxes ---")
    t0 = time.perf_counter()
    result = plan_multitype_palletizing(
        task.boxes, pallet, strategy="3d",
        time_limit_seconds=TIME_LIMIT,
    )
    dt = time.perf_counter() - t0
    print(
        f"  packed={result.packed_count}/{result.requested_count}  "
        f"u3d={result.utilization_3d:.3f}  "
        f"time={dt:.1f}s  status={result.placements[0].item_id if result.placements else 'N/A'}"
    )
