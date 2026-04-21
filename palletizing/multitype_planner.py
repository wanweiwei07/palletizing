from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .geometry import TOL, bottom_alignment_corners, normalized_allowed_yaws, oriented_size, rect_corners
from .layer_patterns import LayerPlacement, plan_best_layer_pattern, plan_best_layer_pattern_fill2d
from .planner import BoxSpec, PalletSpec, plan_palletizing
from .task_generator import TaskBox


SUPPORT_RATIO_THRESHOLD = 0.9
DEFAULT_BEAM_WIDTH = 8
DEFAULT_BRANCHING_FACTOR = 3
SKIP_PENALTY = 2.5


@dataclass(frozen=True)
class MultiTypePlacement:
    sequence: int
    instance_id: str
    box_type_id: str
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
    frequency_group: str
    is_gap_fill: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MultiTypePlanResult:
    placements: list[MultiTypePlacement]
    packed_count: int
    requested_count: int
    unpacked_instance_ids: list[str]
    layer_count: int
    utilization_2d: float
    utilization_3d: float
    message: str

    def as_dict(self) -> dict:
        return {
            "placements": [placement.as_dict() for placement in self.placements],
            "packed_count": self.packed_count,
            "requested_count": self.requested_count,
            "unpacked_instance_ids": self.unpacked_instance_ids,
            "layer_count": self.layer_count,
            "utilization_2d": self.utilization_2d,
            "utilization_3d": self.utilization_3d,
            "message": self.message,
        }


@dataclass(frozen=True)
class _PlacedBox:
    task_box: TaskBox
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def x_min(self) -> float:
        return self.x - self.size_x / 2

    @property
    def x_max(self) -> float:
        return self.x + self.size_x / 2

    @property
    def y_min(self) -> float:
        return self.y - self.size_y / 2

    @property
    def y_max(self) -> float:
        return self.y + self.size_y / 2

    @property
    def z_min(self) -> float:
        return self.z - self.size_z / 2

    @property
    def z_max(self) -> float:
        return self.z + self.size_z / 2

    def bottom_corners(self) -> list[tuple[float, float]]:
        return [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        ]

    def top_corners(self) -> list[tuple[float, float]]:
        return self.bottom_corners()


@dataclass(frozen=True)
class _CandidatePlacement:
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float
    support_ratio: float
    anchor_level: float

    @property
    def x_min(self) -> float:
        return self.x - self.size_x / 2

    @property
    def x_max(self) -> float:
        return self.x + self.size_x / 2

    @property
    def y_min(self) -> float:
        return self.y - self.size_y / 2

    @property
    def y_max(self) -> float:
        return self.y + self.size_y / 2

    @property
    def z_min(self) -> float:
        return self.z - self.size_z / 2

    @property
    def z_max(self) -> float:
        return self.z + self.size_z / 2


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: _CandidatePlacement
    support_ratio: float
    compactness_score: float


@dataclass(frozen=True)
class _BeamState:
    next_index: int
    placed: tuple[_PlacedBox, ...]
    packed_volume: float
    local_score: float
    skipped_count: int


def _load_task_boxes(task_file: str) -> list[TaskBox]:
    with open(task_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        TaskBox(
            instance_id=item["instance_id"],
            box_type_id=item["box_type_id"],
            length=float(item["length"]),
            width=float(item["width"]),
            height=float(item["height"]),
            frequency_group=item["frequency_group"],
            allowed_yaws=tuple(item["allowed_yaws"]),
        )
        for item in payload["boxes"]
    ]


def load_and_plan_multitype_task(task_file: str, pallet: PalletSpec) -> MultiTypePlanResult:
    return plan_multitype_palletizing(_load_task_boxes(task_file), pallet)


def plan_multitype_palletizing(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    strategy: str = "auto",
    time_limit_seconds: float | None = None,
) -> MultiTypePlanResult:
    """Plan palletizing for mixed box types.

    strategy: "auto", "layer-sequence", "layer-first", "beam", "fill2d", or "3d".
    "auto" uses the grid planner for homogeneous boxes, otherwise layer-sequence.
    """
    homogeneous = _homogeneous_boxes(boxes)
    if homogeneous is not None:
        return _plan_homogeneous_boxes(boxes, pallet, homogeneous)
    if strategy == "beam":
        return plan_multitype_palletizing_beam(boxes, pallet)
    if strategy == "layer-first":
        return plan_multitype_palletizing_layer_first(boxes, pallet)
    if strategy == "fill2d":
        return plan_multitype_palletizing_fill2d(boxes, pallet)
    if strategy == "3d":
        kwargs: dict = {}
        if time_limit_seconds is not None:
            kwargs["time_limit_seconds"] = time_limit_seconds
        return plan_multitype_palletizing_3d(boxes, pallet, **kwargs)
    if strategy == "ga3d":
        kwargs_ga: dict = {}
        if time_limit_seconds is not None:
            kwargs_ga["time_limit_seconds"] = time_limit_seconds
        return plan_multitype_palletizing_ga3d(
            boxes, pallet, **kwargs_ga,
        )
    return plan_multitype_palletizing_layer_sequence(boxes, pallet)


def plan_multitype_palletizing_layer_sequence(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    max_stack_size: int = 3,
    min_support_ratio: float = SUPPORT_RATIO_THRESHOLD,
) -> MultiTypePlanResult:
    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")

    remaining_by_id = {box.instance_id: box for box in boxes}
    remaining = list(boxes)
    placements: list[MultiTypePlacement] = []
    support_layer: list[LayerPlacement] | None = None
    z_base = 0.0
    layer_index = 0

    while remaining:
        layer_pattern = plan_best_layer_pattern(
            boxes=remaining,
            pallet=pallet,
            max_stack_size=max_stack_size,
            support_placements=support_layer,
            min_support_ratio=min_support_ratio,
        )
        if not layer_pattern.placements:
            break
        if z_base + layer_pattern.target_height > pallet.max_height + TOL:
            break

        used_ids_this_layer: set[str] = set()
        next_support_layer: list[LayerPlacement] = []
        ordered_layer_blocks = sorted(layer_pattern.placements, key=lambda item: (item.y, item.x, item.block_id))
        for column_index, block in enumerate(ordered_layer_blocks):
            block_boxes = [remaining_by_id[box_id] for box_id in block.box_instance_ids]
            expanded = _expand_layer_block(
                block=block,
                block_boxes=block_boxes,
                z_base=z_base,
                layer_index=layer_index,
                column_index=column_index,
                sequence_start=len(placements) + 1,
            )
            placements.extend(expanded)
            used_ids_this_layer.update(block.box_instance_ids)
            if not block.is_gap_fill:
                next_support_layer.append(block)

        remaining = [box for box in remaining if box.instance_id not in used_ids_this_layer]
        support_layer = next_support_layer
        z_base += layer_pattern.target_height
        layer_index += 1

    placements = _resequenced_layerwise(placements)

    packed_volume = sum(item.size_x * item.size_y * item.size_z for item in placements)
    placed_boxes = [
        _PlacedBox(
            task_box=remaining_by_id[placement.instance_id],
            x=placement.x,
            y=placement.y,
            z=placement.z,
            yaw=placement.yaw,
            size_x=placement.size_x,
            size_y=placement.size_y,
            size_z=placement.size_z,
        )
        for placement in placements
    ]
    layer_footprints = _compute_layer_footprints(placed_boxes)
    pallet_footprint = pallet.length * pallet.width
    pallet_volume = pallet.length * pallet.width * pallet.max_height
    packed_ids = {item.instance_id for item in placements}
    unpacked_instance_ids = [box.instance_id for box in boxes if box.instance_id not in packed_ids]
    layer_count = len(layer_footprints)
    utilization_2d = (sum(layer_footprints.values()) / (pallet_footprint * max(layer_count, 1))) if layer_count else 0.0
    utilization_3d = packed_volume / pallet_volume if pallet_volume > 0 else 0.0
    message = (
        "All boxes were packed successfully."
        if not unpacked_instance_ids
        else f"Packed {len(placements)} of {len(boxes)} boxes with iterative layer reselection."
    )
    return MultiTypePlanResult(
        placements=placements,
        packed_count=len(placements),
        requested_count=len(boxes),
        unpacked_instance_ids=unpacked_instance_ids,
        layer_count=layer_count,
        utilization_2d=utilization_2d,
        utilization_3d=utilization_3d,
        message=message,
    )


def plan_multitype_palletizing_fill2d(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    min_support_ratio: float = SUPPORT_RATIO_THRESHOLD,
    time_limit_seconds: float = 20.0,
    num_workers: int = 8,
) -> MultiTypePlanResult:
    """Layer-sequence planner that uses CP-SAT Fill2D for each layer."""
    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")

    remaining_by_id = {box.instance_id: box for box in boxes}
    remaining = list(boxes)
    placements: list[MultiTypePlacement] = []
    support_layer: list[LayerPlacement] | None = None
    z_base = 0.0
    layer_index = 0

    while remaining:
        layer_pattern = plan_best_layer_pattern_fill2d(
            boxes=remaining,
            pallet=pallet,
            support_placements=support_layer,
            min_support_ratio=min_support_ratio,
            time_limit_seconds=time_limit_seconds,
            num_workers=num_workers,
        )
        if not layer_pattern.placements:
            break
        if z_base + layer_pattern.target_height > pallet.max_height + TOL:
            break

        used_ids_this_layer: set[str] = set()
        next_support_layer: list[LayerPlacement] = []
        ordered_layer_blocks = sorted(
            layer_pattern.placements,
            key=lambda item: (item.y, item.x, item.block_id),
        )
        for column_index, block in enumerate(ordered_layer_blocks):
            block_boxes = [remaining_by_id[bid] for bid in block.box_instance_ids]
            expanded = _expand_layer_block(
                block=block,
                block_boxes=block_boxes,
                z_base=z_base,
                layer_index=layer_index,
                column_index=column_index,
                sequence_start=len(placements) + 1,
            )
            placements.extend(expanded)
            used_ids_this_layer.update(block.box_instance_ids)
            next_support_layer.append(block)

        remaining = [b for b in remaining if b.instance_id not in used_ids_this_layer]
        support_layer = next_support_layer
        z_base += layer_pattern.target_height
        layer_index += 1

    placements = _resequenced_layerwise(placements)

    packed_volume = sum(p.size_x * p.size_y * p.size_z for p in placements)
    placed_boxes = [
        _PlacedBox(
            task_box=remaining_by_id[p.instance_id],
            x=p.x, y=p.y, z=p.z, yaw=p.yaw,
            size_x=p.size_x, size_y=p.size_y, size_z=p.size_z,
        )
        for p in placements
    ]
    layer_footprints = _compute_layer_footprints(placed_boxes)
    pallet_footprint = pallet.length * pallet.width
    pallet_volume = pallet.length * pallet.width * pallet.max_height
    packed_ids = {p.instance_id for p in placements}
    unpacked_instance_ids = [b.instance_id for b in boxes if b.instance_id not in packed_ids]
    layer_count = len(layer_footprints)
    utilization_2d = (
        sum(layer_footprints.values()) / (pallet_footprint * max(layer_count, 1))
        if layer_count else 0.0
    )
    utilization_3d = packed_volume / pallet_volume if pallet_volume > 0 else 0.0
    message = (
        "All boxes were packed successfully."
        if not unpacked_instance_ids
        else f"Packed {len(placements)} of {len(boxes)} boxes with fill2d layer planner."
    )
    return MultiTypePlanResult(
        placements=placements,
        packed_count=len(placements),
        requested_count=len(boxes),
        unpacked_instance_ids=unpacked_instance_ids,
        layer_count=layer_count,
        utilization_2d=utilization_2d,
        utilization_3d=utilization_3d,
        message=message,
    )


FILL3D_SCALE = 1000


def plan_multitype_palletizing_3d(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    time_limit_seconds: float = 300.0,
    num_workers: int = 8,
) -> MultiTypePlanResult:
    """Direct 3D CP-SAT packing — no layer decomposition."""
    from fill3d import Fill3DInstance, Fill3DItem, solve_fill3d

    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")

    S = FILL3D_SCALE
    items = tuple(
        Fill3DItem(
            item_id=box.instance_id,
            length=round(box.length * S),
            width=round(box.width * S),
            height=round(box.height * S),
            allow_rotation=(
                abs(box.length - box.width) > TOL
                and any(
                    abs(y - 1.5707963267948966) < TOL
                    for y in box.allowed_yaws
                )
            ),
        )
        for box in boxes
    )
    instance = Fill3DInstance(
        pallet_length=round(pallet.length * S),
        pallet_width=round(pallet.width * S),
        pallet_max_height=round(pallet.max_height * S),
        items=items,
    )
    result = solve_fill3d(
        instance,
        time_limit_seconds=time_limit_seconds,
        num_workers=num_workers,
    )

    box_by_id = {b.instance_id: b for b in boxes}
    placements: list[MultiTypePlacement] = []
    for seq, fp in enumerate(result.placements, start=1):
        tb = box_by_id[fp.item_id]
        sx = fp.size_x / S
        sy = fp.size_y / S
        sz = fp.size_z / S
        z_bottom = fp.z / S
        # Determine yaw from rotation
        if fp.rotated:
            from math import pi
            yaw = pi / 2
        else:
            yaw = 0.0

        # Assign layer by z-level (quantised to avoid float noise)
        placements.append(
            MultiTypePlacement(
                sequence=seq,
                instance_id=fp.item_id,
                box_type_id=tb.box_type_id,
                x=fp.x / S + sx / 2,
                y=fp.y / S + sy / 2,
                z=z_bottom + sz / 2,
                yaw=yaw,
                size_x=sx,
                size_y=sy,
                size_z=sz,
                layer=0,  # assigned below
                row=0,
                column=seq - 1,
                frequency_group=tb.frequency_group,
            )
        )

    # Assign layers by z-level grouping
    if placements:
        z_levels: dict[int, int] = {}
        for p in placements:
            z_mm = round((p.z - p.size_z / 2) * S)
            if z_mm not in z_levels:
                z_levels[z_mm] = len(z_levels)
        sorted_levels = {
            k: idx
            for idx, k in enumerate(sorted(z_levels.keys()))
        }
        placements = [
            MultiTypePlacement(
                sequence=p.sequence,
                instance_id=p.instance_id,
                box_type_id=p.box_type_id,
                x=p.x, y=p.y, z=p.z, yaw=p.yaw,
                size_x=p.size_x, size_y=p.size_y, size_z=p.size_z,
                layer=sorted_levels[round((p.z - p.size_z / 2) * S)],
                row=p.row, column=p.column,
                frequency_group=p.frequency_group,
            )
            for p in placements
        ]

    packed_volume = sum(
        p.size_x * p.size_y * p.size_z for p in placements
    )
    placed_boxes = [
        _PlacedBox(
            task_box=box_by_id[p.instance_id],
            x=p.x, y=p.y, z=p.z, yaw=p.yaw,
            size_x=p.size_x, size_y=p.size_y, size_z=p.size_z,
        )
        for p in placements
    ]
    layer_footprints = _compute_layer_footprints(placed_boxes)
    pallet_footprint = pallet.length * pallet.width
    pallet_volume = pallet.length * pallet.width * pallet.max_height
    packed_ids = {p.instance_id for p in placements}
    unpacked_ids = [
        b.instance_id for b in boxes
        if b.instance_id not in packed_ids
    ]
    layer_count = len(layer_footprints)
    utilization_2d = (
        sum(layer_footprints.values())
        / (pallet_footprint * max(layer_count, 1))
        if layer_count else 0.0
    )
    utilization_3d = (
        packed_volume / pallet_volume if pallet_volume > 0 else 0.0
    )
    message = (
        f"3D CP-SAT {result.status}: packed {len(placements)}"
        f"/{len(boxes)} boxes."
    )
    return MultiTypePlanResult(
        placements=placements,
        packed_count=len(placements),
        requested_count=len(boxes),
        unpacked_instance_ids=unpacked_ids,
        layer_count=layer_count,
        utilization_2d=utilization_2d,
        utilization_3d=utilization_3d,
        message=message,
    )


def plan_multitype_palletizing_ga3d(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    time_limit_seconds: float = 60.0,
    pop_size: int = 2048,
) -> MultiTypePlanResult:
    """GA-based 3D packing with GPU-parallel evaluation."""
    from ga3d import GA3DConfig, solve_ga3d
    from fill3d import Fill3DInstance, Fill3DItem

    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")

    S = FILL3D_SCALE
    items = tuple(
        Fill3DItem(
            item_id=box.instance_id,
            length=round(box.length * S),
            width=round(box.width * S),
            height=round(box.height * S),
            allow_rotation=(
                abs(box.length - box.width) > TOL
                and any(
                    abs(y - 1.5707963267948966) < TOL
                    for y in box.allowed_yaws
                )
            ),
        )
        for box in boxes
    )
    instance = Fill3DInstance(
        pallet_length=round(pallet.length * S),
        pallet_width=round(pallet.width * S),
        pallet_max_height=round(pallet.max_height * S),
        items=items,
    )
    cfg = GA3DConfig(
        pop_size=pop_size,
        time_limit_seconds=time_limit_seconds,
    )
    result = solve_ga3d(instance, cfg)

    box_by_id = {b.instance_id: b for b in boxes}
    placements: list[MultiTypePlacement] = []
    for seq, fp in enumerate(result.placements, start=1):
        tb = box_by_id[fp.item_id]
        sx = fp.size_x / S
        sy = fp.size_y / S
        sz = fp.size_z / S
        z_bottom = fp.z / S
        if fp.rotated:
            from math import pi
            yaw = pi / 2
        else:
            yaw = 0.0
        placements.append(
            MultiTypePlacement(
                sequence=seq,
                instance_id=fp.item_id,
                box_type_id=tb.box_type_id,
                x=fp.x / S + sx / 2,
                y=fp.y / S + sy / 2,
                z=z_bottom + sz / 2,
                yaw=yaw,
                size_x=sx, size_y=sy, size_z=sz,
                layer=0, row=0, column=seq - 1,
                frequency_group=tb.frequency_group,
            )
        )

    # Assign layers by z-level grouping
    if placements:
        z_levels: dict[int, int] = {}
        for p in placements:
            z_mm = round((p.z - p.size_z / 2) * S)
            if z_mm not in z_levels:
                z_levels[z_mm] = len(z_levels)
        sorted_levels = {
            k: idx
            for idx, k in enumerate(sorted(z_levels.keys()))
        }
        placements = [
            MultiTypePlacement(
                sequence=p.sequence,
                instance_id=p.instance_id,
                box_type_id=p.box_type_id,
                x=p.x, y=p.y, z=p.z, yaw=p.yaw,
                size_x=p.size_x, size_y=p.size_y,
                size_z=p.size_z,
                layer=sorted_levels[
                    round((p.z - p.size_z / 2) * S)
                ],
                row=p.row, column=p.column,
                frequency_group=p.frequency_group,
            )
            for p in placements
        ]

    packed_volume = sum(
        p.size_x * p.size_y * p.size_z for p in placements
    )
    placed_boxes = [
        _PlacedBox(
            task_box=box_by_id[p.instance_id],
            x=p.x, y=p.y, z=p.z, yaw=p.yaw,
            size_x=p.size_x, size_y=p.size_y, size_z=p.size_z,
        )
        for p in placements
    ]
    layer_footprints = _compute_layer_footprints(placed_boxes)
    pallet_footprint = pallet.length * pallet.width
    pallet_volume = (
        pallet.length * pallet.width * pallet.max_height
    )
    packed_ids = {p.instance_id for p in placements}
    unpacked_ids = [
        b.instance_id for b in boxes
        if b.instance_id not in packed_ids
    ]
    layer_count = len(layer_footprints)
    utilization_2d = (
        sum(layer_footprints.values())
        / (pallet_footprint * max(layer_count, 1))
        if layer_count else 0.0
    )
    utilization_3d = (
        packed_volume / pallet_volume
        if pallet_volume > 0 else 0.0
    )
    message = (
        f"GA3D {result.status}: packed {len(placements)}"
        f"/{len(boxes)} boxes, {result.generations} generations."
    )
    return MultiTypePlanResult(
        placements=placements,
        packed_count=len(placements),
        requested_count=len(boxes),
        unpacked_instance_ids=unpacked_ids,
        layer_count=layer_count,
        utilization_2d=utilization_2d,
        utilization_3d=utilization_3d,
        message=message,
    )


def plan_multitype_palletizing_ga3d_seam(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    time_limit_seconds: float = 60.0,
    pop_size: int = 2048,
) -> MultiTypePlanResult:
    """Backward-compatible alias for the GA3D service entrypoint."""
    return plan_multitype_palletizing_ga3d(
        boxes=boxes,
        pallet=pallet,
        time_limit_seconds=time_limit_seconds,
        pop_size=pop_size,
    )


def plan_multitype_palletizing_layer_first(
    boxes: list[TaskBox],
    pallet: PalletSpec,
) -> MultiTypePlanResult:
    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")

    remaining = list(
        sorted(
            boxes,
            key=lambda box: (
                -box.height,
                -(box.length * box.width),
                -box.volume,
                box.instance_id,
            ),
        )
    )

    placed_all: list[_PlacedBox] = []
    z_bottom = 0.0
    support_boxes: list[_PlacedBox] = []

    while remaining and z_bottom < pallet.max_height - TOL:
        layer_plan = _choose_best_layer_plan(remaining, support_boxes, pallet, z_bottom)
        if layer_plan is None:
            break
        layer_boxes, layer_height = layer_plan
        if not layer_boxes or z_bottom + layer_height > pallet.max_height + TOL:
            break

        placed_all.extend(layer_boxes)
        packed_ids = {box.task_box.instance_id for box in layer_boxes}
        remaining = [box for box in remaining if box.instance_id not in packed_ids]
        support_boxes = layer_boxes
        z_bottom += layer_height

    return _build_plan_result(placed_all, boxes, pallet, "layer-first")


def plan_multitype_palletizing_beam(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
) -> MultiTypePlanResult:
    if min(pallet.length, pallet.width, pallet.max_height) <= 0:
        raise ValueError("Pallet dimensions must be positive.")
    if not boxes:
        raise ValueError("boxes must not be empty")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if branching_factor <= 0:
        raise ValueError("branching_factor must be positive")

    ordered_boxes = tuple(
        sorted(
            boxes,
            key=lambda box: (
                -box.volume,
                -(box.length * box.width),
                -box.height,
                box.instance_id,
            ),
        )
    )

    beam: list[_BeamState] = [
        _BeamState(next_index=0, placed=tuple(), packed_volume=0.0, local_score=0.0, skipped_count=0)
    ]

    for next_index in range(len(ordered_boxes)):
        task_box = ordered_boxes[next_index]
        expanded_states: list[_BeamState] = []
        for state in beam:
            scored_candidates = _scored_candidates(task_box, list(state.placed), pallet)
            for scored in scored_candidates[:branching_factor]:
                candidate = scored.candidate
                placed_box = _PlacedBox(
                    task_box=task_box,
                    x=candidate.x,
                    y=candidate.y,
                    z=candidate.z,
                    yaw=candidate.yaw,
                    size_x=candidate.size_x,
                    size_y=candidate.size_y,
                    size_z=candidate.size_z,
                )
                expanded_states.append(
                    _BeamState(
                        next_index=next_index + 1,
                        placed=state.placed + (placed_box,),
                        packed_volume=state.packed_volume + placed_box.size_x * placed_box.size_y * placed_box.size_z,
                        local_score=state.local_score + scored.compactness_score + 10.0,
                        skipped_count=state.skipped_count,
                    )
                )

            expanded_states.append(
                _BeamState(
                    next_index=next_index + 1,
                    placed=state.placed,
                    packed_volume=state.packed_volume,
                    local_score=state.local_score - SKIP_PENALTY,
                    skipped_count=state.skipped_count + 1,
                )
            )

        beam = sorted(expanded_states, key=_beam_state_rank, reverse=True)[:beam_width]

    best_state = max(beam, key=_beam_state_rank)
    return _build_plan_result(list(best_state.placed), boxes, pallet, "beam-search")


def _choose_best_layer_plan(
    remaining: list[TaskBox],
    support_boxes: list[_PlacedBox],
    pallet: PalletSpec,
    z_bottom: float,
) -> tuple[list[_PlacedBox], float] | None:
    best: tuple[list[_PlacedBox], float, tuple[float, float, float, float]] | None = None
    unique_heights = sorted({box.height for box in remaining}, reverse=True)

    for height in unique_heights:
        if z_bottom + height > pallet.max_height + TOL:
            continue
        height_group = [box for box in remaining if abs(box.height - height) <= TOL]
        layer_boxes = _pack_single_height_layer(height_group, support_boxes, pallet, z_bottom)
        if not layer_boxes:
            continue

        total_area = sum(box.size_x * box.size_y for box in layer_boxes)
        bbox_area = _level_bbox_area(layer_boxes, z_bottom)
        fill_ratio = 0.0 if bbox_area <= TOL else total_area / bbox_area
        score = (
            len(layer_boxes),
            total_area,
            fill_ratio,
            -height,
        )
        if best is None or score > best[2]:
            best = (layer_boxes, height, score)

    if best is None:
        return None
    return best[0], best[1]


def _pack_single_height_layer(
    boxes: list[TaskBox],
    support_boxes: list[_PlacedBox],
    pallet: PalletSpec,
    z_bottom: float,
) -> list[_PlacedBox]:
    ordered = sorted(
        boxes,
        key=lambda box: (
            -(box.length * box.width),
            -box.volume,
            box.instance_id,
        ),
    )
    layer_boxes: list[_PlacedBox] = []

    for task_box in ordered:
        candidate = _select_best_layer_candidate(task_box, layer_boxes, support_boxes, pallet, z_bottom)
        if candidate is None:
            continue
        layer_boxes.append(
            _PlacedBox(
                task_box=task_box,
                x=candidate.x,
                y=candidate.y,
                z=candidate.z,
                yaw=candidate.yaw,
                size_x=candidate.size_x,
                size_y=candidate.size_y,
                size_z=candidate.size_z,
            )
        )
    return layer_boxes


def _build_plan_result(placed: list[_PlacedBox], boxes: list[TaskBox], pallet: PalletSpec, planner_name: str) -> MultiTypePlanResult:
    placements: list[MultiTypePlacement] = []
    layer_map = _build_layer_map(placed)
    for placed_box in placed:
        layer_index = layer_map[round(placed_box.z_min, 9)]
        placements.append(
            MultiTypePlacement(
                sequence=len(placements) + 1,
                instance_id=placed_box.task_box.instance_id,
                box_type_id=placed_box.task_box.box_type_id,
                x=placed_box.x,
                y=placed_box.y,
                z=placed_box.z,
                yaw=placed_box.yaw,
                size_x=placed_box.size_x,
                size_y=placed_box.size_y,
                size_z=placed_box.size_z,
                layer=layer_index,
                row=0,
                column=0,
                frequency_group=placed_box.task_box.frequency_group,
                is_gap_fill=False,
            )
        )

    packed_volume = sum(item.size_x * item.size_y * item.size_z for item in placed)
    layer_footprints = _compute_layer_footprints(placed)
    pallet_footprint = pallet.length * pallet.width
    pallet_volume = pallet.length * pallet.width * pallet.max_height
    packed_ids = {item.task_box.instance_id for item in placed}
    unpacked_instance_ids = [box.instance_id for box in boxes if box.instance_id not in packed_ids]
    layer_count = len(layer_footprints)
    utilization_2d = (sum(layer_footprints.values()) / (pallet_footprint * max(layer_count, 1))) if layer_count else 0.0
    utilization_3d = packed_volume / pallet_volume if pallet_volume > 0 else 0.0
    message = (
        "All boxes were packed successfully."
        if not unpacked_instance_ids
        else f"Packed {len(placed)} of {len(boxes)} boxes with the {planner_name} multitype planner."
    )
    return MultiTypePlanResult(
        placements=placements,
        packed_count=len(placed),
        requested_count=len(boxes),
        unpacked_instance_ids=unpacked_instance_ids,
        layer_count=layer_count,
        utilization_2d=utilization_2d,
        utilization_3d=utilization_3d,
        message=message,
    )


def _build_layer_map(placed: list[_PlacedBox]) -> dict[float, int]:
    levels = sorted({round(item.z_min, 9) for item in placed})
    return {level: index for index, level in enumerate(levels)}


def _compute_layer_footprints(placed: list[_PlacedBox]) -> dict[float, float]:
    footprints: dict[float, float] = {}
    for item in placed:
        level = round(item.z_min, 9)
        footprints[level] = footprints.get(level, 0.0) + item.size_x * item.size_y
    return footprints


def _select_best_candidate(task_box: TaskBox, placed: list[_PlacedBox], pallet: PalletSpec) -> _CandidatePlacement | None:
    feasible = _scored_candidates(task_box, placed, pallet)
    if not feasible:
        return None
    best = max(
        feasible,
        key=lambda item: (
            item.compactness_score,
            item.support_ratio,
            -item.candidate.z_min,
            -item.candidate.anchor_level,
            -item.candidate.y_min,
            -item.candidate.x_min,
        ),
    )
    return best.candidate


def _select_best_layer_candidate(
    task_box: TaskBox,
    layer_boxes: list[_PlacedBox],
    support_boxes: list[_PlacedBox],
    pallet: PalletSpec,
    z_bottom: float,
) -> _CandidatePlacement | None:
    candidates = _generate_layer_candidates(task_box, layer_boxes, pallet, z_bottom)
    feasible: list[tuple[_CandidatePlacement, float, float]] = []
    for candidate in candidates:
        if not _fits_inside(candidate, pallet):
            continue
        if _overlaps_any(candidate, layer_boxes):
            continue
        support_ratio = 1.0 if z_bottom <= TOL else _support_ratio(candidate, support_boxes)
        if support_ratio + TOL < SUPPORT_RATIO_THRESHOLD:
            continue
        feasible.append(
            (
                candidate,
                support_ratio,
                _layer_candidate_score(candidate, support_ratio, layer_boxes, pallet, z_bottom),
            )
        )
    if not feasible:
        return None
    best_candidate, _, _ = max(
        feasible,
        key=lambda item: (
            item[2],
            item[1],
            -item[0].y_min,
            -item[0].x_min,
        ),
    )
    return best_candidate


def _scored_candidates(task_box: TaskBox, placed: list[_PlacedBox], pallet: PalletSpec) -> list[_ScoredCandidate]:
    candidates = _generate_candidates(task_box, placed, pallet)
    feasible: list[_ScoredCandidate] = []
    for candidate in candidates:
        if not _fits_inside(candidate, pallet):
            continue
        if _overlaps_any(candidate, placed):
            continue
        support_ratio = _support_ratio(candidate, placed)
        if support_ratio + TOL < SUPPORT_RATIO_THRESHOLD:
            continue
        feasible.append(
            _ScoredCandidate(
                candidate=candidate,
                support_ratio=support_ratio,
                compactness_score=_compactness_score(candidate, support_ratio, placed, pallet),
            )
        )
    return feasible


def _generate_layer_candidates(
    task_box: TaskBox,
    layer_boxes: list[_PlacedBox],
    pallet: PalletSpec,
    z_bottom: float,
) -> list[_CandidatePlacement]:
    candidates: list[_CandidatePlacement] = []
    seen: set[tuple[float, float, float, float, float, float, float]] = set()

    def push(candidate: _CandidatePlacement) -> None:
        key = (
            round(candidate.x, 9),
            round(candidate.y, 9),
            round(candidate.z, 9),
            round(candidate.yaw, 9),
            round(candidate.size_x, 9),
            round(candidate.size_y, 9),
            round(candidate.size_z, 9),
        )
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    pallet_corners = [
        (0.0, 0.0),
        (pallet.length, 0.0),
        (pallet.length, pallet.width),
        (0.0, pallet.width),
    ]

    for size_x, size_y, size_z, yaw in _candidate_orientations(task_box):
        if abs(size_z - task_box.height) > TOL:
            continue
        local_corners = rect_corners(size_x, size_y)

        for anchor_index, anchor_corner in enumerate(pallet_corners):
            for new_corner in _ground_alignment_corners(local_corners, anchor_index):
                push(
                    _candidate_from_corner_alignment(
                        anchor_corner=anchor_corner,
                        new_corner=new_corner,
                        z_bottom=z_bottom,
                        size_x=size_x,
                        size_y=size_y,
                        size_z=size_z,
                        yaw=yaw,
                        anchor_level=z_bottom,
                    )
                )

        for anchor in layer_boxes:
            for anchor_index, anchor_corner in enumerate(anchor.bottom_corners()):
                for new_corner in bottom_alignment_corners(local_corners, anchor_index):
                    push(
                        _candidate_from_corner_alignment(
                            anchor_corner=anchor_corner,
                            new_corner=new_corner,
                            z_bottom=z_bottom,
                            size_x=size_x,
                            size_y=size_y,
                            size_z=size_z,
                            yaw=yaw,
                            anchor_level=z_bottom,
                        )
                    )
    return candidates


def _layer_candidate_score(
    candidate: _CandidatePlacement,
    support_ratio: float,
    layer_boxes: list[_PlacedBox],
    pallet: PalletSpec,
    z_bottom: float,
) -> float:
    candidate_area = candidate.size_x * candidate.size_y
    if candidate_area <= TOL:
        return float("-inf")

    box_contact_length, boundary_contact_length = _contact_lengths(candidate, layer_boxes, pallet)
    current_void = _level_void_area(layer_boxes, z_bottom)
    next_void = _level_void_area(layer_boxes, z_bottom, candidate)
    void_growth = next_void - current_void

    current_bbox = _level_bbox_area(layer_boxes, z_bottom)
    next_bbox = _level_bbox_area(layer_boxes, z_bottom, candidate)
    bbox_growth = next_bbox - current_bbox

    return (
        8.0 * (box_contact_length / max(candidate.size_x + candidate.size_y, TOL))
        + 4.0 * support_ratio
        - 7.0 * (void_growth / candidate_area)
        - 3.0 * (bbox_growth / candidate_area)
        - 1.0 * (boundary_contact_length / max(candidate.size_x + candidate.size_y, TOL))
    )


def _beam_state_rank(state: _BeamState) -> tuple[float, float, float, float]:
    expandability = _beam_expandability_score(state.placed)
    return (
        len(state.placed),
        state.packed_volume + expandability,
        expandability,
        state.local_score,
        -state.skipped_count,
    )


def _beam_expandability_score(placed: tuple[_PlacedBox, ...]) -> float:
    if not placed:
        return 0.0

    footprint_bbox = _footprint_bbox_area(list(placed))
    occupied_footprint = sum(box.size_x * box.size_y for box in placed)
    footprint_fill = occupied_footprint / max(footprint_bbox, TOL)

    top_levels = sorted({round(box.z_max, 9) for box in placed})
    top_level_penalty = max(0, len(top_levels) - 1)

    best_support_area = 0.0
    best_support_fill = 0.0
    for z_top in top_levels:
        level_boxes = [box for box in placed if abs(box.z_max - z_top) <= TOL]
        level_area = sum(box.size_x * box.size_y for box in level_boxes)
        level_bbox = _top_level_bbox_area(list(placed), z_top)
        level_fill = 0.0 if level_bbox <= TOL else level_area / level_bbox
        if level_area > best_support_area:
            best_support_area = level_area
            best_support_fill = level_fill

    avg_top_fill = 0.0
    if top_levels:
        fills = []
        for z_top in top_levels:
            level_area = _top_level_area(list(placed), z_top)
            level_bbox = _top_level_bbox_area(list(placed), z_top)
            fills.append(0.0 if level_bbox <= TOL else level_area / level_bbox)
        avg_top_fill = sum(fills) / len(fills)

    max_top_area_ratio = best_support_area / max(occupied_footprint, TOL)
    layer_quality = _layer_quality_score(placed)

    return (
        12.0 * footprint_fill
        + 10.0 * max_top_area_ratio
        + 8.0 * best_support_fill
        + 4.0 * avg_top_fill
        + 8.0 * layer_quality
        - 3.0 * top_level_penalty
    )


def _layer_quality_score(placed: tuple[_PlacedBox, ...]) -> float:
    if not placed:
        return 0.0

    levels = sorted({round(box.z_min, 9) for box in placed})
    quality_terms: list[float] = []
    for z_level in levels:
        level_boxes = [box for box in placed if abs(box.z_min - z_level) <= TOL]
        if not level_boxes:
            continue
        area = sum(box.size_x * box.size_y for box in level_boxes)
        bbox = _level_bbox_area(list(placed), z_level)
        fill = 0.0 if bbox <= TOL else area / bbox

        top_heights = [box.z_max for box in level_boxes]
        dominant_top_height = max(set(round(z, 9) for z in top_heights), key=lambda z: sum(1 for x in top_heights if abs(x - z) <= TOL))
        dominant_count = sum(1 for x in top_heights if abs(x - dominant_top_height) <= TOL)
        dominant_ratio = dominant_count / len(level_boxes)

        quality_terms.append(0.7 * fill + 0.3 * dominant_ratio)

    return sum(quality_terms) / len(quality_terms)


def _generate_candidates(task_box: TaskBox, placed: list[_PlacedBox], pallet: PalletSpec) -> list[_CandidatePlacement]:
    candidates: list[_CandidatePlacement] = []
    seen: set[tuple[float, float, float, float, float, float, float]] = set()

    def push(candidate: _CandidatePlacement) -> None:
        key = (
            round(candidate.x, 9),
            round(candidate.y, 9),
            round(candidate.z, 9),
            round(candidate.yaw, 9),
            round(candidate.size_x, 9),
            round(candidate.size_y, 9),
            round(candidate.size_z, 9),
        )
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    pallet_corners = [
        (0.0, 0.0),
        (pallet.length, 0.0),
        (pallet.length, pallet.width),
        (0.0, pallet.width),
    ]

    for size_x, size_y, size_z, yaw in _candidate_orientations(task_box):
        local_corners = rect_corners(size_x, size_y)
        for anchor_index, anchor_corner in enumerate(pallet_corners):
            for new_corner in _ground_alignment_corners(local_corners, anchor_index):
                push(
                    _candidate_from_corner_alignment(
                        anchor_corner=anchor_corner,
                        new_corner=new_corner,
                        z_bottom=0.0,
                        size_x=size_x,
                        size_y=size_y,
                        size_z=size_z,
                        yaw=yaw,
                        anchor_level=0.0,
                    )
                )

    for anchor in placed:
        for size_x, size_y, size_z, yaw in _candidate_orientations(task_box):
            local_corners = rect_corners(size_x, size_y)

            for anchor_index, anchor_corner in enumerate(anchor.bottom_corners()):
                for new_corner in bottom_alignment_corners(local_corners, anchor_index):
                    push(
                        _candidate_from_corner_alignment(
                            anchor_corner=anchor_corner,
                            new_corner=new_corner,
                            z_bottom=0.0,
                            size_x=size_x,
                            size_y=size_y,
                            size_z=size_z,
                            yaw=yaw,
                            anchor_level=0.0,
                        )
                    )

            top_anchor_corners = anchor.top_corners()
            for anchor_index, anchor_corner in enumerate(top_anchor_corners):
                new_corner = _top_alignment_corner(local_corners, anchor_index)
                push(
                    _candidate_from_corner_alignment(
                        anchor_corner=anchor_corner,
                        new_corner=new_corner,
                        z_bottom=anchor.z_max,
                        size_x=size_x,
                        size_y=size_y,
                        size_z=size_z,
                        yaw=yaw,
                        anchor_level=anchor.z_max,
                    )
                )

    # Once a higher layer has started, prefer packing tightly against that same layer's bottom corners.
    for z_bottom in sorted({box.z_min for box in placed if box.z_min > TOL}):
        same_level_boxes = [box for box in placed if abs(box.z_min - z_bottom) <= TOL]
        for level_box in same_level_boxes:
            for size_x, size_y, size_z, yaw in _candidate_orientations(task_box):
                local_corners = rect_corners(size_x, size_y)
                for anchor_index, anchor_corner in enumerate(level_box.bottom_corners()):
                    for new_corner in bottom_alignment_corners(local_corners, anchor_index):
                        push(
                            _candidate_from_corner_alignment(
                                anchor_corner=anchor_corner,
                                new_corner=new_corner,
                                z_bottom=z_bottom,
                                size_x=size_x,
                                size_y=size_y,
                                size_z=size_z,
                                yaw=yaw,
                                anchor_level=z_bottom,
                            )
                        )

    return candidates


def _candidate_from_corner_alignment(
    anchor_corner: tuple[float, float],
    new_corner: tuple[float, float],
    z_bottom: float,
    size_x: float,
    size_y: float,
    size_z: float,
    yaw: float,
    anchor_level: float,
) -> _CandidatePlacement:
    x_min = anchor_corner[0] - new_corner[0]
    y_min = anchor_corner[1] - new_corner[1]
    return _CandidatePlacement(
        x=x_min + size_x / 2,
        y=y_min + size_y / 2,
        z=z_bottom + size_z / 2,
        yaw=yaw,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        support_ratio=0.0,
        anchor_level=anchor_level,
    )


def _fits_inside(candidate: _CandidatePlacement, pallet: PalletSpec) -> bool:
    return (
        candidate.x_min >= -TOL
        and candidate.y_min >= -TOL
        and candidate.z_min >= -TOL
        and candidate.x_max <= pallet.length + TOL
        and candidate.y_max <= pallet.width + TOL
        and candidate.z_max <= pallet.max_height + TOL
    )


def _overlaps_any(candidate: _CandidatePlacement, placed: list[_PlacedBox]) -> bool:
    for other in placed:
        overlap_x = min(candidate.x_max, other.x_max) - max(candidate.x_min, other.x_min)
        overlap_y = min(candidate.y_max, other.y_max) - max(candidate.y_min, other.y_min)
        overlap_z = min(candidate.z_max, other.z_max) - max(candidate.z_min, other.z_min)
        if overlap_x > TOL and overlap_y > TOL and overlap_z > TOL:
            return True
    return False


def _support_ratio(candidate: _CandidatePlacement, placed: list[_PlacedBox]) -> float:
    if candidate.z_min <= TOL:
        return 1.0

    support_area = 0.0
    for other in placed:
        if abs(other.z_max - candidate.z_min) > TOL:
            continue
        overlap_x = max(0.0, min(candidate.x_max, other.x_max) - max(candidate.x_min, other.x_min))
        overlap_y = max(0.0, min(candidate.y_max, other.y_max) - max(candidate.y_min, other.y_min))
        support_area += overlap_x * overlap_y

    base_area = candidate.size_x * candidate.size_y
    if base_area <= TOL:
        return 0.0
    return support_area / base_area


def _compactness_score(
    candidate: _CandidatePlacement,
    support_ratio: float,
    placed: list[_PlacedBox],
    pallet: PalletSpec,
) -> float:
    candidate_area = candidate.size_x * candidate.size_y
    if candidate_area <= TOL:
        return float("-inf")

    current_bbox_area = _footprint_bbox_area(placed)
    next_bbox_area = _footprint_bbox_area(placed, candidate)
    bbox_growth = next_bbox_area - current_bbox_area

    current_support_level_area = _level_bbox_area(placed, candidate.z_min)
    next_support_level_area = _level_bbox_area(placed, candidate.z_min, candidate)
    level_growth = next_support_level_area - current_support_level_area

    box_contact_length, boundary_contact_length = _contact_lengths(candidate, placed, pallet)
    current_level_void = _level_void_area(placed, candidate.z_min)
    next_level_void = _level_void_area(placed, candidate.z_min, candidate)
    void_growth = next_level_void - current_level_void
    top_level_area = _top_level_area(placed, candidate.z_max)
    top_level_bbox_area = _top_level_bbox_area(placed, candidate.z_max)
    top_level_alignment = top_level_area / max(candidate_area, TOL)
    top_level_compactness = 0.0 if top_level_bbox_area <= TOL else top_level_area / top_level_bbox_area
    new_top_level_penalty = 0.0 if top_level_area > TOL else 1.0
    unsupported_penalty = 1.0 - support_ratio

    # Higher is better.
    return (
        7.0 * (box_contact_length / max(candidate.size_x + candidate.size_y, TOL))
        + 4.0 * support_ratio
        + 5.0 * top_level_alignment
        + 3.0 * top_level_compactness
        - 6.0 * (void_growth / candidate_area)
        - 2.5 * (bbox_growth / candidate_area)
        - 1.5 * (level_growth / candidate_area)
        - 1.5 * (boundary_contact_length / max(candidate.size_x + candidate.size_y, TOL))
        - 3.0 * new_top_level_penalty
        - 5.0 * unsupported_penalty
    )


def _footprint_bbox_area(placed: list[_PlacedBox], candidate: _CandidatePlacement | None = None) -> float:
    if not placed and candidate is None:
        return 0.0
    x_mins = [box.x_min for box in placed]
    x_maxs = [box.x_max for box in placed]
    y_mins = [box.y_min for box in placed]
    y_maxs = [box.y_max for box in placed]
    if candidate is not None:
        x_mins.append(candidate.x_min)
        x_maxs.append(candidate.x_max)
        y_mins.append(candidate.y_min)
        y_maxs.append(candidate.y_max)
    return (max(x_maxs) - min(x_mins)) * (max(y_maxs) - min(y_mins))


def _level_bbox_area(
    placed: list[_PlacedBox],
    z_bottom: float,
    candidate: _CandidatePlacement | None = None,
) -> float:
    same_level = [box for box in placed if abs(box.z_min - z_bottom) <= TOL]
    if not same_level and candidate is None:
        return 0.0
    x_mins = [box.x_min for box in same_level]
    x_maxs = [box.x_max for box in same_level]
    y_mins = [box.y_min for box in same_level]
    y_maxs = [box.y_max for box in same_level]
    if candidate is not None:
        x_mins.append(candidate.x_min)
        x_maxs.append(candidate.x_max)
        y_mins.append(candidate.y_min)
        y_maxs.append(candidate.y_max)
    return (max(x_maxs) - min(x_mins)) * (max(y_maxs) - min(y_mins))


def _level_void_area(
    placed: list[_PlacedBox],
    z_bottom: float,
    candidate: _CandidatePlacement | None = None,
) -> float:
    same_level = [box for box in placed if abs(box.z_min - z_bottom) <= TOL]
    occupied_area = sum(box.size_x * box.size_y for box in same_level)
    if candidate is not None:
        occupied_area += candidate.size_x * candidate.size_y
    bbox_area = _level_bbox_area(placed, z_bottom, candidate)
    return max(0.0, bbox_area - occupied_area)


def _top_level_area(placed: list[_PlacedBox], z_top: float) -> float:
    return sum(box.size_x * box.size_y for box in placed if abs(box.z_max - z_top) <= TOL)


def _top_level_bbox_area(placed: list[_PlacedBox], z_top: float) -> float:
    same_top = [box for box in placed if abs(box.z_max - z_top) <= TOL]
    if not same_top:
        return 0.0
    x_mins = [box.x_min for box in same_top]
    x_maxs = [box.x_max for box in same_top]
    y_mins = [box.y_min for box in same_top]
    y_maxs = [box.y_max for box in same_top]
    return (max(x_maxs) - min(x_mins)) * (max(y_maxs) - min(y_mins))


def _contact_lengths(candidate: _CandidatePlacement, placed: list[_PlacedBox], pallet: PalletSpec) -> tuple[float, float]:
    box_contact = 0.0
    boundary_contact = 0.0

    if abs(candidate.x_min) <= TOL:
        boundary_contact += candidate.size_y
    if abs(candidate.x_max - pallet.length) <= TOL:
        boundary_contact += candidate.size_y
    if abs(candidate.y_min) <= TOL:
        boundary_contact += candidate.size_x
    if abs(candidate.y_max - pallet.width) <= TOL:
        boundary_contact += candidate.size_x

    for other in placed:
        if abs(other.z_min - candidate.z_min) > TOL:
            continue

        overlap_y = max(0.0, min(candidate.y_max, other.y_max) - max(candidate.y_min, other.y_min))
        if overlap_y > TOL:
            if abs(candidate.x_min - other.x_max) <= TOL or abs(candidate.x_max - other.x_min) <= TOL:
                box_contact += overlap_y

        overlap_x = max(0.0, min(candidate.x_max, other.x_max) - max(candidate.x_min, other.x_min))
        if overlap_x > TOL:
            if abs(candidate.y_min - other.y_max) <= TOL or abs(candidate.y_max - other.y_min) <= TOL:
                box_contact += overlap_x

    return box_contact, boundary_contact


def _expand_layer_block(
    block: LayerPlacement,
    block_boxes: list[TaskBox],
    z_base: float,
    layer_index: int,
    column_index: int,
    sequence_start: int,
) -> list[MultiTypePlacement]:
    ordered_boxes = sorted(
        block_boxes,
        key=lambda box: (
            round(box.height, 9),
            box.instance_id,
        ),
    )
    current_z = z_base
    placements: list[MultiTypePlacement] = []
    for offset, task_box in enumerate(ordered_boxes):
        yaw = _resolve_box_yaw(task_box, block.size_x, block.size_y)
        size_x, size_y = _oriented_footprint(task_box, yaw)
        placements.append(
            MultiTypePlacement(
                sequence=sequence_start + offset,
                instance_id=task_box.instance_id,
                box_type_id=task_box.box_type_id,
                x=block.x,
                y=block.y,
                z=current_z + task_box.height / 2,
                yaw=yaw,
                size_x=size_x,
                size_y=size_y,
                size_z=task_box.height,
                layer=layer_index,
                row=0,
                column=column_index,
                frequency_group=task_box.frequency_group,
                is_gap_fill=block.is_gap_fill,
            )
        )
        current_z += task_box.height
    return placements


def _homogeneous_boxes(boxes: list[TaskBox]) -> TaskBox | None:
    if not boxes:
        return None
    first = boxes[0]
    for box in boxes[1:]:
        if (
            abs(box.length - first.length) > TOL
            or abs(box.width - first.width) > TOL
            or abs(box.height - first.height) > TOL
            or tuple(box.allowed_yaws) != tuple(first.allowed_yaws)
        ):
            return None
    return first


def _plan_homogeneous_boxes(boxes: list[TaskBox], pallet: PalletSpec, prototype: TaskBox) -> MultiTypePlanResult:
    single_result = plan_palletizing(
        pallet,
        BoxSpec(
            length=prototype.length,
            width=prototype.width,
            height=prototype.height,
            count=len(boxes),
        ),
    )
    placements: list[MultiTypePlacement] = []
    for placement, task_box in zip(single_result.placements, boxes, strict=False):
        placements.append(
            MultiTypePlacement(
                sequence=placement.sequence,
                instance_id=task_box.instance_id,
                box_type_id=task_box.box_type_id,
                x=placement.x,
                y=placement.y,
                z=placement.z,
                yaw=placement.yaw,
                size_x=placement.size_x,
                size_y=placement.size_y,
                size_z=placement.size_z,
                layer=placement.layer,
                row=placement.row,
                column=placement.column,
                frequency_group=task_box.frequency_group,
                is_gap_fill=False,
            )
        )
    requested_count = len(boxes)
    packed_count = len(placements)
    unpacked_instance_ids = [box.instance_id for box in boxes[packed_count:]]
    return MultiTypePlanResult(
        placements=placements,
        packed_count=packed_count,
        requested_count=requested_count,
        unpacked_instance_ids=unpacked_instance_ids,
        layer_count=single_result.layer_count,
        utilization_2d=single_result.utilization_2d,
        utilization_3d=single_result.utilization_3d,
        message=single_result.message,
    )


def _resequenced_layerwise(placements: list[MultiTypePlacement]) -> list[MultiTypePlacement]:
    ordered = sorted(
        placements,
        key=lambda placement: (
            placement.layer,
            placement.z,
            placement.y,
            placement.x,
            placement.instance_id,
        ),
    )
    resequenced: list[MultiTypePlacement] = []
    for sequence, placement in enumerate(ordered, start=1):
        resequenced.append(
            MultiTypePlacement(
                sequence=sequence,
                instance_id=placement.instance_id,
                box_type_id=placement.box_type_id,
                x=placement.x,
                y=placement.y,
                z=placement.z,
                yaw=placement.yaw,
                size_x=placement.size_x,
                size_y=placement.size_y,
                size_z=placement.size_z,
                layer=placement.layer,
                row=placement.row,
                column=placement.column,
                frequency_group=placement.frequency_group,
                is_gap_fill=placement.is_gap_fill,
            )
        )
    return resequenced


def _resolve_box_yaw(task_box: TaskBox, target_size_x: float, target_size_y: float) -> float:
    for size_x, size_y, _, yaw in _candidate_orientations(task_box):
        if abs(size_x - target_size_x) <= TOL and abs(size_y - target_size_y) <= TOL:
            return yaw
    raise ValueError(
        f"Box {task_box.instance_id} cannot match target footprint ({target_size_x}, {target_size_y})."
    )


def _oriented_footprint(task_box: TaskBox, yaw: float) -> tuple[float, float]:
    return oriented_size(task_box.length, task_box.width, yaw)


def _ground_alignment_corners(local_corners: list[tuple[float, float]], anchor_index: int) -> list[tuple[float, float]]:
    # For the pallet boundary, allow the box to start from any corner-aligned footprint.
    return list(local_corners)


def _top_alignment_corner(local_corners: list[tuple[float, float]], anchor_index: int) -> tuple[float, float]:
    # Top-on-bottom: use the angle-order mapping confirmed by the user.
    # yaw = 0   : a'>e, b'>f, c'>g, d'>h
    # yaw = pi/2: a'>f, b'>g, c'>h, d'>e
    # Because local_corners are already expressed in world positions after yaw is applied,
    # this reduces to taking the corner at the same world-order index.
    return local_corners[anchor_index]


def _candidate_orientations(task_box: TaskBox) -> list[tuple[float, float, float, float]]:
    yaws = normalized_allowed_yaws(task_box.length, task_box.width, task_box.allowed_yaws)
    return [
        (*oriented_size(task_box.length, task_box.width, yaw), task_box.height, yaw)
        for yaw in yaws
    ]
