from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor, pi
from typing import Iterable


@dataclass(frozen=True)
class BoxSpec:
    length: float
    width: float
    height: float
    count: int


@dataclass(frozen=True)
class PalletSpec:
    length: float
    width: float
    max_height: float


@dataclass(frozen=True)
class Placement:
    sequence: int
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float
    layer: int
    row: int
    column: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlanResult:
    placements: list[Placement]
    packed_count: int
    requested_count: int
    utilization_2d: float
    utilization_3d: float
    layer_count: int
    layout_rows: int
    layout_columns: int
    orientation: tuple[float, float, float]
    message: str

    def as_dict(self) -> dict:
        return {
            "placements": [placement.as_dict() for placement in self.placements],
            "packed_count": self.packed_count,
            "requested_count": self.requested_count,
            "utilization_2d": self.utilization_2d,
            "utilization_3d": self.utilization_3d,
            "layer_count": self.layer_count,
            "layout_rows": self.layout_rows,
            "layout_columns": self.layout_columns,
            "orientation": {
                "size_x": self.orientation[0],
                "size_y": self.orientation[1],
                "size_z": self.orientation[2],
            },
            "rotation": {
                "yaw": self.placements[0].yaw if self.placements else 0.0,
            },
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateLayout:
    size_x: float
    size_y: float
    size_z: float
    yaw: float
    columns: int
    rows: int
    layers: int
    capacity: int
    packed_count: int
    utilization_2d: float
    utilization_3d: float


def _unique_orientations(box: BoxSpec) -> Iterable[tuple[tuple[float, float, float], float]]:
    # Keep the box upright and only rotate it in the pallet plane.
    candidates = [
        ((box.length, box.width, box.height), 0.0),
        ((box.width, box.length, box.height), pi / 2),
    ]
    seen: set[tuple[float, float, float]] = set()
    for size_triplet, yaw in candidates:
        if size_triplet not in seen:
            seen.add(size_triplet)
            yield size_triplet, yaw


def _build_candidate(
    pallet: PalletSpec,
    box: BoxSpec,
    orientation: tuple[float, float, float],
    yaw: float,
) -> CandidateLayout | None:
    size_x, size_y, size_z = orientation
    columns = floor(pallet.length / size_x)
    rows = floor(pallet.width / size_y)
    layers = floor(pallet.max_height / size_z)

    if columns <= 0 or rows <= 0 or layers <= 0:
        return None

    per_layer = columns * rows
    capacity = per_layer * layers
    packed_count = min(box.count, capacity)
    footprint = columns * size_x * rows * size_y
    stack_volume = packed_count * size_x * size_y * size_z
    pallet_volume = pallet.length * pallet.width * pallet.max_height

    return CandidateLayout(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        yaw=yaw,
        columns=columns,
        rows=rows,
        layers=layers,
        capacity=capacity,
        packed_count=packed_count,
        utilization_2d=footprint / (pallet.length * pallet.width),
        utilization_3d=stack_volume / pallet_volume,
    )


def _choose_best_layout(pallet: PalletSpec, box: BoxSpec) -> CandidateLayout:
    candidates = [
        candidate
        for orientation, yaw in _unique_orientations(box)
        if (candidate := _build_candidate(pallet, box, orientation, yaw)) is not None
    ]

    if not candidates:
        raise ValueError("No feasible layout found. Check pallet and box dimensions.")

    # Prefer fitting more boxes first, then denser base coverage, then fuller volume usage.
    return max(
        candidates,
        key=lambda candidate: (
            candidate.packed_count,
            candidate.utilization_2d,
            candidate.utilization_3d,
            -candidate.size_z,
        ),
    )


def plan_palletizing(pallet: PalletSpec, box: BoxSpec) -> PlanResult:
    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if min(box.length, box.width, box.height) <= 0:
        raise ValueError("Box dimensions must be positive.")
    if box.count <= 0:
        raise ValueError("Box count must be positive.")

    layout = _choose_best_layout(pallet, box)
    placements: list[Placement] = []
    per_layer = layout.columns * layout.rows

    for sequence in range(layout.packed_count):
        layer = sequence // per_layer
        in_layer_index = sequence % per_layer
        row = in_layer_index // layout.columns
        column = in_layer_index % layout.columns

        # Snake ordering reduces long horizontal travel on each layer.
        effective_column = column if row % 2 == 0 else (layout.columns - 1 - column)
        x = (effective_column + 0.5) * layout.size_x
        y = (row + 0.5) * layout.size_y
        z = (layer + 0.5) * layout.size_z

        placements.append(
            Placement(
                sequence=sequence + 1,
                x=x,
                y=y,
                z=z,
                yaw=layout.yaw,
                size_x=layout.size_x,
                size_y=layout.size_y,
                size_z=layout.size_z,
                layer=layer,
                row=row,
                column=effective_column,
            )
        )

    planned_layers = (layout.packed_count + per_layer - 1) // per_layer
    message = (
        "All boxes were packed successfully."
        if layout.packed_count == box.count
        else f"Only {layout.packed_count} of {box.count} boxes fit within the pallet constraints."
    )

    return PlanResult(
        placements=placements,
        packed_count=layout.packed_count,
        requested_count=box.count,
        utilization_2d=layout.utilization_2d,
        utilization_3d=(layout.packed_count * layout.size_x * layout.size_y * layout.size_z)
        / (pallet.length * pallet.width * pallet.max_height),
        layer_count=planned_layers,
        layout_rows=layout.rows,
        layout_columns=layout.columns,
        orientation=(layout.size_x, layout.size_y, layout.size_z),
        message=message,
    )
