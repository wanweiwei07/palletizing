from __future__ import annotations

from dataclasses import asdict, dataclass

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Fill2DItem:
    item_id: str
    width: int
    height: int
    allow_rotation: bool = True
    value: int | None = None

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def objective_value(self) -> int:
        return self.area if self.value is None else self.value


@dataclass(frozen=True)
class Fill2DInstance:
    pallet_width: int
    pallet_height: int
    items: tuple[Fill2DItem, ...]

    @property
    def pallet_area(self) -> int:
        return self.pallet_width * self.pallet_height


@dataclass(frozen=True)
class Fill2DPlacement:
    item_id: str
    x: int
    y: int
    rotated: bool
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Fill2DResult:
    status: str
    objective_value: int
    used_item_count: int
    covered_area: int
    coverage_ratio: float
    placements: list[Fill2DPlacement]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "used_item_count": self.used_item_count,
            "covered_area": self.covered_area,
            "coverage_ratio": self.coverage_ratio,
            "placements": [placement.as_dict() for placement in self.placements],
        }


def solve_fill2d(
    instance: Fill2DInstance,
    time_limit_seconds: float = 10.0,
    num_workers: int = 8,
    exclusion_groups: list[list[int]] | None = None,
    fixed_placements: list[Fill2DPlacement] | None = None,
) -> Fill2DResult:
    """Solve a 2D bin-packing problem with CP-SAT.

    fixed_placements: already-placed rectangles that occupy space on the pallet.
    New items must not overlap them. They do not count towards the objective.
    """
    if instance.pallet_width <= 0 or instance.pallet_height <= 0:
        raise ValueError("pallet dimensions must be positive integers")
    if not instance.items and not fixed_placements:
        raise ValueError("instance must contain at least one item")

    model = cp_model.CpModel()

    # --- Fixed (already-placed) items as immovable obstacles ---------------
    fixed_x_intervals: list[cp_model.IntervalVar] = []
    fixed_y_intervals: list[cp_model.IntervalVar] = []
    if fixed_placements:
        for fi, fp in enumerate(fixed_placements):
            fx_iv = model.NewFixedSizeIntervalVar(fp.x, fp.width, f"fx_{fi}")
            fy_iv = model.NewFixedSizeIntervalVar(fp.y, fp.height, f"fy_{fi}")
            fixed_x_intervals.append(fx_iv)
            fixed_y_intervals.append(fy_iv)

    # --- Decision variables for candidate items ----------------------------
    used_vars: list[cp_model.IntVar] = []
    rot_vars: list[cp_model.IntVar] = []
    x_vars: list[cp_model.IntVar] = []
    y_vars: list[cp_model.IntVar] = []
    width_vars: list[cp_model.IntVar] = []
    height_vars: list[cp_model.IntVar] = []
    x_intervals: list[cp_model.IntervalVar] = []
    y_intervals: list[cp_model.IntervalVar] = []

    for index, item in enumerate(instance.items):
        used = model.NewBoolVar(f"u_{index}")
        rotated = model.NewBoolVar(f"r_{index}")
        w_lo = min(item.width, item.height)
        w_hi = max(item.width, item.height)
        x = model.NewIntVar(0, instance.pallet_width, f"x_{index}")
        y = model.NewIntVar(0, instance.pallet_height, f"y_{index}")
        width = model.NewIntVar(w_lo, w_hi, f"w_{index}")
        height = model.NewIntVar(w_lo, w_hi, f"h_{index}")

        if item.allow_rotation and item.width != item.height:
            model.Add(width == item.width).OnlyEnforceIf(rotated.Not())
            model.Add(height == item.height).OnlyEnforceIf(rotated.Not())
            model.Add(width == item.height).OnlyEnforceIf(rotated)
            model.Add(height == item.width).OnlyEnforceIf(rotated)
        else:
            model.Add(rotated == 0)
            model.Add(width == item.width)
            model.Add(height == item.height)

        x_end = model.NewIntVar(0, instance.pallet_width, f"xe_{index}")
        y_end = model.NewIntVar(0, instance.pallet_height, f"ye_{index}")
        model.Add(x_end == x + width)
        model.Add(y_end == y + height)

        x_iv = model.NewOptionalIntervalVar(
            x, width, x_end, used, f"xi_{index}",
        )
        y_iv = model.NewOptionalIntervalVar(
            y, height, y_end, used, f"yi_{index}",
        )

        used_vars.append(used)
        rot_vars.append(rotated)
        x_vars.append(x)
        y_vars.append(y)
        width_vars.append(width)
        height_vars.append(height)
        x_intervals.append(x_iv)
        y_intervals.append(y_iv)

    # NoOverlap2D across both fixed obstacles and candidate items.
    all_x = fixed_x_intervals + x_intervals
    all_y = fixed_y_intervals + y_intervals
    model.AddNoOverlap2D(all_x, all_y)

    if exclusion_groups:
        for group in exclusion_groups:
            model.Add(sum(used_vars[i] for i in group) <= 1)

    model.Maximize(
        sum(
            item.objective_value * used_vars[index]
            for index, item in enumerate(instance.items)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_workers
    solver.parameters.random_seed = 42
    solver.parameters.interleave_search = True
    solver.parameters.max_deterministic_time = time_limit_seconds * 8.0
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)

    placements: list[Fill2DPlacement] = []
    covered_area = 0
    objective_value = (
        int(solver.ObjectiveValue())
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        else 0
    )

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for index, item in enumerate(instance.items):
            if solver.Value(used_vars[index]) != 1:
                continue
            rotated = bool(solver.Value(rot_vars[index]))
            width = solver.Value(width_vars[index])
            height = solver.Value(height_vars[index])
            placements.append(
                Fill2DPlacement(
                    item_id=item.item_id,
                    x=solver.Value(x_vars[index]),
                    y=solver.Value(y_vars[index]),
                    rotated=rotated,
                    width=width,
                    height=height,
                )
            )
            covered_area += width * height

    placements.sort(
        key=lambda placement: (placement.y, placement.x, placement.item_id),
    )
    return Fill2DResult(
        status=status,
        objective_value=objective_value,
        used_item_count=len(placements),
        covered_area=covered_area,
        coverage_ratio=(covered_area / instance.pallet_area),
        placements=placements,
    )
