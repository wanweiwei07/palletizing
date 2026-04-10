"""Quick benchmark: 3D bin packing with CP-SAT for 70 boxes."""
from __future__ import annotations

import time

from ortools.sat.python import cp_model

from palletizing import PalletSpec, generate_multitype_task

SCALE = 1000  # meters → mm for integer grid


def benchmark_3d_cpsat(
    n_boxes: int,
    seed: int = 7,
    time_limit: float = 30.0,
) -> None:
    task = generate_multitype_task(count=n_boxes, seed=seed)
    boxes = task.boxes
    pallet = PalletSpec(length=1.2, width=1.0, max_height=1.5)

    PW = round(pallet.length * SCALE)
    PH = round(pallet.width * SCALE)
    PZ = round(pallet.max_height * SCALE)

    model = cp_model.CpModel()

    used = []
    rot = []
    xs, ys, zs = [], [], []
    ws, ds, hs = [], [], []
    x_ends, y_ends, z_ends = [], [], []

    for i, box in enumerate(boxes):
        bw = round(box.length * SCALE)
        bd = round(box.width * SCALE)
        bh = round(box.height * SCALE)

        u = model.NewBoolVar(f"u_{i}")
        r = model.NewBoolVar(f"r_{i}")
        used.append(u)
        rot.append(r)

        x = model.NewIntVar(0, PW, f"x_{i}")
        y = model.NewIntVar(0, PH, f"y_{i}")
        z = model.NewIntVar(0, PZ, f"z_{i}")
        xs.append(x); ys.append(y); zs.append(z)

        w = model.NewIntVar(min(bw, bd), max(bw, bd), f"w_{i}")
        d = model.NewIntVar(min(bw, bd), max(bw, bd), f"d_{i}")
        h_var = model.NewConstant(bh)  # height fixed (this-side-up)
        ws.append(w); ds.append(d); hs.append(h_var)

        # Rotation in yaw only
        if bw != bd:
            model.Add(w == bw).OnlyEnforceIf(r.Not())
            model.Add(d == bd).OnlyEnforceIf(r.Not())
            model.Add(w == bd).OnlyEnforceIf(r)
            model.Add(d == bw).OnlyEnforceIf(r)
        else:
            model.Add(r == 0)
            model.Add(w == bw)
            model.Add(d == bd)

        xe = model.NewIntVar(0, PW, f"xe_{i}")
        ye = model.NewIntVar(0, PH, f"ye_{i}")
        ze = model.NewIntVar(0, PZ, f"ze_{i}")
        model.Add(xe == x + w)
        model.Add(ye == y + d)
        model.Add(ze == z + bh)
        x_ends.append(xe); y_ends.append(ye); z_ends.append(ze)

    # --- Pairwise 3D non-overlap ---
    # For each pair (i,j), if both used, they must be separated
    # in at least one of x, y, z.
    n = len(boxes)
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Only enforce when both are used
            both = model.NewBoolVar(f"both_{i}_{j}")
            model.AddBoolAnd([used[i], used[j]]).OnlyEnforceIf(both)
            model.AddBoolOr([used[i].Not(), used[j].Not()]).OnlyEnforceIf(
                both.Not()
            )

            # 6 separation directions
            b = [model.NewBoolVar(f"sep_{i}_{j}_{d}") for d in range(6)]
            model.Add(x_ends[i] <= xs[j]).OnlyEnforceIf(b[0])
            model.Add(x_ends[j] <= xs[i]).OnlyEnforceIf(b[1])
            model.Add(y_ends[i] <= ys[j]).OnlyEnforceIf(b[2])
            model.Add(y_ends[j] <= ys[i]).OnlyEnforceIf(b[3])
            model.Add(z_ends[i] <= zs[j]).OnlyEnforceIf(b[4])
            model.Add(z_ends[j] <= zs[i]).OnlyEnforceIf(b[5])

            # At least one separation when both used
            model.AddBoolOr(b).OnlyEnforceIf(both)
            pair_count += 1

    print(f"Boxes: {n}, Pairs: {pair_count}, "
          f"Vars (est): {n * 10 + pair_count * 7}")

    # Maximize total volume of placed boxes
    model.Maximize(
        sum(
            round(box.volume * SCALE**3) * used[i]
            for i, box in enumerate(boxes)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42

    print(f"Solving with time_limit={time_limit}s ...")
    t0 = time.time()
    status_code = solver.Solve(model)
    dt = time.time() - t0
    status = solver.StatusName(status_code)

    packed = 0
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        packed = sum(1 for i in range(n) if solver.Value(used[i]))

    print(f"Status: {status}  Packed: {packed}/{n}  Time: {dt:.1f}s")
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Objective: {solver.ObjectiveValue():.0f}")


if __name__ == "__main__":
    for n in [10, 20, 30, 50, 70]:
        print(f"\n{'='*50}")
        benchmark_3d_cpsat(n, time_limit=30.0)
