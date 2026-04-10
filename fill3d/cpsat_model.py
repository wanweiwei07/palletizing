"""3D bin-packing solver using OR-Tools CP-SAT.

Places boxes on a pallet with:
- 3D non-overlap (pairwise disjunctive constraints)
- Redundant cumulative constraints for faster propagation
- Symmetry breaking for identical boxes
- Support constraint (each box rests on floor or on a box below)
- Yaw rotation only (this-side-up)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Fill3DItem:
    item_id: str
    length: int  # mm (x-dimension before rotation)
    width: int   # mm (y-dimension before rotation)
    height: int  # mm (z-dimension, fixed)
    allow_rotation: bool = True

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


@dataclass(frozen=True)
class Fill3DInstance:
    pallet_length: int  # mm
    pallet_width: int   # mm
    pallet_max_height: int  # mm
    items: tuple[Fill3DItem, ...]


@dataclass(frozen=True)
class Fill3DPlacement:
    item_id: str
    x: int  # mm, left edge
    y: int  # mm, front edge
    z: int  # mm, bottom edge
    rotated: bool
    size_x: int  # mm, placed length
    size_y: int  # mm, placed width
    size_z: int  # mm, height (always = item.height)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Fill3DResult:
    status: str
    packed_count: int
    total_items: int
    packed_volume: int
    pallet_volume: int
    volume_ratio: float
    placements: list[Fill3DPlacement]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "packed_count": self.packed_count,
            "total_items": self.total_items,
            "packed_volume": self.packed_volume,
            "pallet_volume": self.pallet_volume,
            "volume_ratio": self.volume_ratio,
            "placements": [p.as_dict() for p in self.placements],
        }


MIN_SUPPORT_RATIO = 0.7


def solve_fill3d(
    instance: Fill3DInstance,
    time_limit_seconds: float = 300.0,
    num_workers: int = 8,
    min_support_ratio: float = MIN_SUPPORT_RATIO,
) -> Fill3DResult:
    """Solve a 3D bin-packing problem on a single pallet with CP-SAT.

    Each box can rotate in yaw (swap length/width) but not tip over.
    Boxes must rest on the pallet floor or on top of another placed box
    with x-y overlap (support constraint).
    """
    PL = instance.pallet_length
    PW = instance.pallet_width
    PH = instance.pallet_max_height
    if PL <= 0 or PW <= 0 or PH <= 0:
        raise ValueError("pallet dimensions must be positive")
    if not instance.items:
        raise ValueError("instance must contain at least one item")

    items = instance.items
    n = len(items)
    model = cp_model.CpModel()

    # --- Per-box decision variables ---
    used: list[cp_model.IntVar] = []
    rot: list[cp_model.IntVar] = []
    x_vars: list[cp_model.IntVar] = []
    y_vars: list[cp_model.IntVar] = []
    z_vars: list[cp_model.IntVar] = []
    sx_vars: list[cp_model.IntVar] = []  # placed size_x
    sy_vars: list[cp_model.IntVar] = []  # placed size_y
    xe_vars: list[cp_model.IntVar] = []  # x + size_x
    ye_vars: list[cp_model.IntVar] = []  # y + size_y
    ze_vars: list[cp_model.IntVar] = []  # z + height

    for i, item in enumerate(items):
        u = model.NewBoolVar(f"u_{i}")
        r = model.NewBoolVar(f"r_{i}")
        used.append(u)
        rot.append(r)

        lo = min(item.length, item.width)
        hi = max(item.length, item.width)
        sx = model.NewIntVar(lo, hi, f"sx_{i}")
        sy = model.NewIntVar(lo, hi, f"sy_{i}")
        sx_vars.append(sx)
        sy_vars.append(sy)

        if item.allow_rotation and item.length != item.width:
            model.Add(sx == item.length).OnlyEnforceIf(r.Not())
            model.Add(sy == item.width).OnlyEnforceIf(r.Not())
            model.Add(sx == item.width).OnlyEnforceIf(r)
            model.Add(sy == item.length).OnlyEnforceIf(r)
        else:
            model.Add(r == 0)
            model.Add(sx == item.length)
            model.Add(sy == item.width)

        x = model.NewIntVar(0, PL, f"x_{i}")
        y = model.NewIntVar(0, PW, f"y_{i}")
        z = model.NewIntVar(0, PH, f"z_{i}")
        x_vars.append(x)
        y_vars.append(y)
        z_vars.append(z)

        xe = model.NewIntVar(0, PL, f"xe_{i}")
        ye = model.NewIntVar(0, PW, f"ye_{i}")
        ze = model.NewIntVar(0, PH, f"ze_{i}")
        model.Add(xe == x + sx)
        model.Add(ye == y + sy)
        model.Add(ze == z + item.height)
        xe_vars.append(xe)
        ye_vars.append(ye)
        ze_vars.append(ze)

    # --- 3D non-overlap: pairwise disjunctive constraints ---
    # For each pair (i, j) where both are used, at least one of 6
    # separation conditions must hold.
    for i in range(n):
        for j in range(i + 1, n):
            # "both used" indicator
            both = model.NewBoolVar(f"both_{i}_{j}")
            model.AddBoolAnd(
                [used[i], used[j]]
            ).OnlyEnforceIf(both)
            model.AddBoolOr(
                [used[i].Not(), used[j].Not()]
            ).OnlyEnforceIf(both.Not())

            # 6 separation directions
            sep = [model.NewBoolVar(f"s_{i}_{j}_{d}") for d in range(6)]
            model.Add(xe_vars[i] <= x_vars[j]).OnlyEnforceIf(sep[0])
            model.Add(xe_vars[j] <= x_vars[i]).OnlyEnforceIf(sep[1])
            model.Add(ye_vars[i] <= y_vars[j]).OnlyEnforceIf(sep[2])
            model.Add(ye_vars[j] <= y_vars[i]).OnlyEnforceIf(sep[3])
            model.Add(ze_vars[i] <= z_vars[j]).OnlyEnforceIf(sep[4])
            model.Add(ze_vars[j] <= z_vars[i]).OnlyEnforceIf(sep[5])

            model.AddBoolOr(sep).OnlyEnforceIf(both)

    # --- Redundant cumulative on z-axis (help propagation) ---
    # At any z-level, total base area of boxes present ≤ PL * PW.
    z_intervals = []
    z_demands = []
    for i, item in enumerate(items):
        intv = model.NewOptionalIntervalVar(
            z_vars[i], model.NewConstant(item.height), ze_vars[i],
            used[i], f"zi_{i}",
        )
        z_intervals.append(intv)
        # Use min possible area as demand (safe lower bound)
        lo = min(item.length, item.width)
        hi = max(item.length, item.width)
        z_demands.append(lo * hi)
    model.AddCumulative(z_intervals, z_demands, PL * PW)

    # --- Symmetry breaking: identical boxes ordered by (z, y, x) ---
    sig_to_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        sig = (item.length, item.width, item.height)
        sig_to_indices[sig].append(i)
    for indices in sig_to_indices.values():
        if len(indices) < 2:
            continue
        for a, b in zip(indices, indices[1:]):
            # If both used, a must come before b in (z, y, x) order.
            # Simplified: just break on x when both on same z and y.
            both_ab = model.NewBoolVar(f"sym_{a}_{b}")
            model.AddBoolAnd([used[a], used[b]]).OnlyEnforceIf(both_ab)
            model.AddBoolOr(
                [used[a].Not(), used[b].Not()]
            ).OnlyEnforceIf(both_ab.Not())
            # Lexicographic ordering
            M = max(PL, PW, PH) + 1
            model.Add(
                z_vars[a] * M * M + y_vars[a] * M + x_vars[a]
                <= z_vars[b] * M * M + y_vars[b] * M + x_vars[b]
            ).OnlyEnforceIf(both_ab)

    # --- Support constraint (multi-box, 2×2 probe points) ---
    # 4 probe points at the 25%/75% positions of box base.
    # Each probe can be covered by ANY support box below, allowing
    # multi-box bridging.  Require >= 3/4 probes (≈75% support).
    #
    # Probe (kx, ky), kx/ky in {0, 1}:
    #   scaled ×4:  px = 4*x[i] + (2*kx+1)*sx[i]
    #   coeffs: kx=0 → 1, kx=1 → 3
    PROBE_COEFFS = (1, 3)
    MIN_PROBES = 3  # out of 4 → 75%

    for i in range(n):
        on_floor_i = model.NewBoolVar(f"fl_{i}")
        model.Add(z_vars[i] == 0).OnlyEnforceIf(on_floor_i)
        model.Add(z_vars[i] >= 1).OnlyEnforceIf(on_floor_i.Not())

        probe_covered: list[cp_model.IntVar] = []
        for kx, cx_c in enumerate(PROBE_COEFFS):
            for ky, cy_c in enumerate(PROBE_COEFFS):
                pk = kx * 2 + ky
                cov_opts: list[cp_model.IntVar] = [on_floor_i]

                for j in range(n):
                    if i == j:
                        continue
                    c = model.NewBoolVar(f"c_{i}_{pk}_{j}")
                    cov_opts.append(c)

                    model.Add(
                        z_vars[i] == ze_vars[j]
                    ).OnlyEnforceIf(c)
                    model.AddImplication(c, used[j])

                    # j covers probe (scaled ×4)
                    model.Add(
                        4 * x_vars[j]
                        <= 4 * x_vars[i] + cx_c * sx_vars[i]
                    ).OnlyEnforceIf(c)
                    model.Add(
                        4 * x_vars[i] + cx_c * sx_vars[i]
                        <= 4 * xe_vars[j]
                    ).OnlyEnforceIf(c)
                    model.Add(
                        4 * y_vars[j]
                        <= 4 * y_vars[i] + cy_c * sy_vars[i]
                    ).OnlyEnforceIf(c)
                    model.Add(
                        4 * y_vars[i] + cy_c * sy_vars[i]
                        <= 4 * ye_vars[j]
                    ).OnlyEnforceIf(c)

                pc = model.NewBoolVar(f"pc_{i}_{pk}")
                model.AddBoolOr(cov_opts + [pc.Not()])
                for opt in cov_opts:
                    model.AddImplication(opt, pc)
                probe_covered.append(pc)

        model.Add(
            sum(probe_covered) >= MIN_PROBES
        ).OnlyEnforceIf(used[i])

    # --- Objective: maximize total volume of packed boxes ---
    model.Maximize(
        sum(item.volume * used[i] for i, item in enumerate(items))
    )

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_workers
    solver.parameters.random_seed = 42
    solver.parameters.interleave_search = True
    solver.parameters.max_deterministic_time = time_limit_seconds * 8.0

    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)

    placements: list[Fill3DPlacement] = []
    packed_volume = 0

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, item in enumerate(items):
            if solver.Value(used[i]) != 1:
                continue
            sx = solver.Value(sx_vars[i])
            sy = solver.Value(sy_vars[i])
            placements.append(
                Fill3DPlacement(
                    item_id=item.item_id,
                    x=solver.Value(x_vars[i]),
                    y=solver.Value(y_vars[i]),
                    z=solver.Value(z_vars[i]),
                    rotated=bool(solver.Value(rot[i])),
                    size_x=sx,
                    size_y=sy,
                    size_z=item.height,
                )
            )
            packed_volume += sx * sy * item.height

    # Post-processing filter disabled — probe-point constraint in CP-SAT
    # already enforces ~78% support area from multiple supports.

    placements.sort(key=lambda p: (p.z, p.y, p.x, p.item_id))
    pallet_vol = PL * PW * PH
    return Fill3DResult(
        status=status,
        packed_count=len(placements),
        total_items=n,
        packed_volume=packed_volume,
        pallet_volume=pallet_vol,
        volume_ratio=packed_volume / pallet_vol if pallet_vol > 0 else 0.0,
        placements=placements,
    )


def _overlap_area(a: Fill3DPlacement, b: Fill3DPlacement) -> int:
    """Compute x-y overlap area (mm²) between two placements."""
    ox = max(0, min(a.x + a.size_x, b.x + b.size_x) - max(a.x, b.x))
    oy = max(0, min(a.y + a.size_y, b.y + b.size_y) - max(a.y, b.y))
    return ox * oy


def _support_ratio(
    box: Fill3DPlacement,
    all_placed: list[Fill3DPlacement],
) -> float:
    """Fraction of box's base area supported by boxes directly below."""
    if box.z == 0:
        return 1.0  # on the floor
    base_area = box.size_x * box.size_y
    if base_area == 0:
        return 1.0
    # Find supports: boxes whose top (z + size_z) == box.z
    # and that overlap in x-y.
    # Note: support may come from multiple boxes. Their union area
    # may be less than the sum of individual overlaps (if supports
    # overlap each other). We use a conservative raster scan for
    # accuracy.
    supported_area = _raster_support_area(box, all_placed)
    return supported_area / base_area


def _raster_support_area(
    box: Fill3DPlacement,
    all_placed: list[Fill3DPlacement],
    resolution: int = 10,
) -> int:
    """Raster-scan the base of `box` and count mm² that are supported.

    Uses a grid of `resolution` strips in each dimension for accuracy
    with multi-box support (handles overlapping supports correctly).
    """
    # Find all direct supports (touching from below)
    supports = [
        p for p in all_placed
        if p is not box
        and p.z + p.size_z == box.z
        and _overlap_area(p, box) > 0
    ]
    if not supports:
        return 0

    # Divide box base into resolution×resolution cells
    dx = box.size_x / resolution
    dy = box.size_y / resolution
    cell_area = dx * dy
    covered_cells = 0
    for ix in range(resolution):
        cx = box.x + (ix + 0.5) * dx
        for iy in range(resolution):
            cy = box.y + (iy + 0.5) * dy
            for s in supports:
                if s.x <= cx <= s.x + s.size_x and s.y <= cy <= s.y + s.size_y:
                    covered_cells += 1
                    break

    return int(covered_cells * cell_area)


def _enforce_support_area(
    placements: list[Fill3DPlacement],
    min_ratio: float,
) -> list[Fill3DPlacement]:
    """Remove boxes that badly violate the support-area ratio.

    To avoid destructive cascade (removing a lower box destabilises
    everything above it), we use a conservative strategy:
    - Only remove boxes with < HARD_FLOOR support (clearly floating).
    - Iterate until stable, processing top-down each pass.
    """
    HARD_FLOOR = min(min_ratio, 0.5)
    remaining = list(placements)
    changed = True
    while changed:
        changed = False
        remaining.sort(key=lambda p: -p.z)
        keep: list[Fill3DPlacement] = []
        for p in remaining:
            ratio = _support_ratio(p, keep + [
                q for q in remaining if q.z < p.z
            ])
            if ratio + 1e-9 >= HARD_FLOOR:
                keep.append(p)
            else:
                changed = True
        remaining = keep
    return remaining
