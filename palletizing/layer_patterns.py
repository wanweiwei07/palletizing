from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

from fill2d import Fill2DInstance, Fill2DItem, solve_fill2d

from .geometry import TOL, bottom_alignment_corners, normalized_allowed_yaws, oriented_size, rect_corners
from .height_blocks import _round_height
from .planner import PalletSpec
from .task_generator import TaskBox

FILL2D_SCALE = 1000
DEFAULT_LAYER_BEAM_WIDTH = 8
DEFAULT_LAYER_BRANCHING = 10


@dataclass(frozen=True)
class LayerBlock:
    block_id: str
    target_height: float
    box_instance_ids: tuple[str, ...]
    box_type_ids: tuple[str, ...]
    component_count: int
    base_length: float
    base_width: float
    total_volume: float
    allowed_yaws: tuple[float, ...]
    is_composite: bool

    @property
    def base_area(self) -> float:
        return self.base_length * self.base_width


@dataclass(frozen=True)
class LayerPlacement:
    block_id: str
    box_instance_ids: tuple[str, ...]
    box_type_ids: tuple[str, ...]
    x: float
    y: float
    yaw: float
    size_x: float
    size_y: float
    target_height: float
    component_count: int
    is_composite: bool
    is_gap_fill: bool = False

    @property
    def area(self) -> float:
        return self.size_x * self.size_y


@dataclass(frozen=True)
class LayerPatternResult:
    target_height: float
    placements: list[LayerPlacement]
    packed_block_count: int
    used_box_count: int
    used_box_instance_ids: list[str]
    covered_area: float
    utilization_2d: float
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "target_height": self.target_height,
            "placements": [
                {
                    "block_id": placement.block_id,
                    "box_instance_ids": list(placement.box_instance_ids),
                    "box_type_ids": list(placement.box_type_ids),
                    "x": placement.x,
                    "y": placement.y,
                    "yaw": placement.yaw,
                    "size_x": placement.size_x,
                    "size_y": placement.size_y,
                    "target_height": placement.target_height,
                    "component_count": placement.component_count,
                    "is_composite": placement.is_composite,
                    "is_gap_fill": placement.is_gap_fill,
                }
                for placement in self.placements
            ],
            "packed_block_count": self.packed_block_count,
            "used_box_count": self.used_box_count,
            "used_box_instance_ids": self.used_box_instance_ids,
            "covered_area": self.covered_area,
            "utilization_2d": self.utilization_2d,
            "message": self.message,
        }


@dataclass(frozen=True)
class _PlacementChoice:
    block: LayerBlock
    x: float
    y: float
    yaw: float
    size_x: float
    size_y: float
    support_ratio: float
    score: tuple[float, ...]


@dataclass(frozen=True)
class _LayerState:
    placements: tuple[LayerPlacement, ...]
    used_box_ids: frozenset[str]
    covered_area: float
    score: tuple[float, ...]


def build_layer_blocks(
    boxes: list[TaskBox],
    target_height: float,
    max_stack_size: int = 3,
) -> list[LayerBlock]:
    rounded_target = _round_height(target_height)
    blocks: list[LayerBlock] = []

    for box in boxes:
        if _round_height(box.height) != rounded_target:
            continue
        blocks.append(
            LayerBlock(
                block_id=f"single:{box.instance_id}",
                target_height=rounded_target,
                box_instance_ids=(box.instance_id,),
                box_type_ids=(box.box_type_id,),
                component_count=1,
                base_length=box.length,
                base_width=box.width,
                total_volume=box.volume,
                allowed_yaws=normalized_allowed_yaws(box.length, box.width, box.allowed_yaws),
                is_composite=False,
            )
        )

    max_size = min(max_stack_size, len(boxes))
    for stack_size in range(2, max_size + 1):
        _extend_composite_blocks(boxes, rounded_target, stack_size, blocks)

    return sorted(
        blocks,
        key=lambda block: (
            -block.base_area,
            -block.component_count,
            -block.total_volume,
            block.block_id,
        ),
    )


def plan_best_layer_pattern(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    max_stack_size: int = 3,
    support_placements: list[LayerPlacement] | None = None,
    min_support_ratio: float = 0.9,
    beam_width: int = DEFAULT_LAYER_BEAM_WIDTH,
    branching_factor: int = DEFAULT_LAYER_BRANCHING,
) -> LayerPatternResult:
    target_heights = sorted({_round_height(box.height) for box in boxes})
    if not target_heights:
        raise ValueError("boxes must not be empty")

    best_result: LayerPatternResult | None = None
    for target_height in target_heights:
        candidate = plan_layer_pattern_for_height(
            boxes=boxes,
            pallet=pallet,
            target_height=target_height,
            max_stack_size=max_stack_size,
            support_placements=support_placements,
            min_support_ratio=min_support_ratio,
            beam_width=beam_width,
            branching_factor=branching_factor,
        )
        if best_result is None or _layer_result_key(candidate) > _layer_result_key(best_result):
            best_result = candidate

    assert best_result is not None
    return best_result


def plan_layer_pattern_for_height(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    target_height: float,
    max_stack_size: int = 3,
    support_placements: list[LayerPlacement] | None = None,
    min_support_ratio: float = 0.9,
    beam_width: int = DEFAULT_LAYER_BEAM_WIDTH,
    branching_factor: int = DEFAULT_LAYER_BRANCHING,
) -> LayerPatternResult:
    if min(pallet.length, pallet.width) <= 0:
        raise ValueError("pallet footprint must be positive")

    blocks = build_layer_blocks(boxes, target_height, max_stack_size=max_stack_size)
    if not blocks:
        return LayerPatternResult(
            target_height=_round_height(target_height),
            placements=[],
            packed_block_count=0,
            used_box_count=0,
            used_box_instance_ids=[],
            covered_area=0.0,
            utilization_2d=0.0,
            message="No layer blocks match the requested target height.",
        )

    initial_state = _LayerState(
        placements=tuple(),
        used_box_ids=frozenset(),
        covered_area=0.0,
        score=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    beam = [initial_state]
    best_state = initial_state

    for _ in range(len(blocks)):
        expansions: list[_LayerState] = []
        for state in beam:
            choices = _generate_layer_choices(
                blocks=blocks,
                state=state,
                pallet=pallet,
                support_placements=support_placements,
                min_support_ratio=min_support_ratio,
            )
            for choice in choices[:branching_factor]:
                placement = LayerPlacement(
                    block_id=choice.block.block_id,
                    box_instance_ids=choice.block.box_instance_ids,
                    box_type_ids=choice.block.box_type_ids,
                    x=choice.x + choice.size_x / 2,
                    y=choice.y + choice.size_y / 2,
                    yaw=choice.yaw,
                    size_x=choice.size_x,
                    size_y=choice.size_y,
                    target_height=choice.block.target_height,
                    component_count=choice.block.component_count,
                    is_composite=choice.block.is_composite,
                    # True gap-fill should mean cross-height corner filling.
                    # The current planner only places one target height per layer,
                    # so same-height in-layer fills are still part of the main layer.
                    is_gap_fill=False,
                )
                next_placements = tuple([*state.placements, placement])
                next_used_box_ids = frozenset([*state.used_box_ids, *choice.block.box_instance_ids])
                next_covered_area = state.covered_area + placement.area
                next_score = _layer_state_score(next_placements, next_covered_area, pallet)
                expansions.append(
                    _LayerState(
                        placements=next_placements,
                        used_box_ids=next_used_box_ids,
                        covered_area=next_covered_area,
                        score=next_score,
                    )
                )

        if not expansions:
            break

        beam = _dedupe_states(expansions)
        beam = sorted(beam, key=lambda state: state.score, reverse=True)[:beam_width]
        if beam[0].score > best_state.score:
            best_state = beam[0]

        if best_state.covered_area >= pallet.length * pallet.width - TOL:
            break

    placements = list(best_state.placements)
    covered_area = best_state.covered_area
    pallet_area = pallet.length * pallet.width
    used_box_list = sorted(best_state.used_box_ids)
    return LayerPatternResult(
        target_height=_round_height(target_height),
        placements=placements,
        packed_block_count=len(placements),
        used_box_count=len(used_box_list),
        used_box_instance_ids=used_box_list,
        covered_area=covered_area,
        utilization_2d=(covered_area / pallet_area if pallet_area > TOL else 0.0),
        message=f"Generated {len(placements)} layer blocks at target height {target_height:.2f} m.",
    )


def _extend_composite_blocks(
    boxes: list[TaskBox],
    rounded_target: float,
    stack_size: int,
    blocks: list[LayerBlock],
) -> None:
    eligible = [box for box in boxes if box.height <= rounded_target + TOL]
    stack: list[int] = []

    def dfs(start_index: int, remaining_height: float) -> None:
        if len(stack) == stack_size:
            if abs(remaining_height) <= TOL:
                subset = [eligible[index] for index in stack]
                composite = _build_composite_block(subset, rounded_target)
                if composite is not None:
                    blocks.append(composite)
            return

        for index in range(start_index, len(eligible)):
            box = eligible[index]
            next_remaining = _round_height(remaining_height - box.height)
            if next_remaining < -TOL:
                continue
            stack.append(index)
            dfs(index + 1, next_remaining)
            stack.pop()

    dfs(0, rounded_target)


def _build_composite_block(subset: list[TaskBox], rounded_target: float) -> LayerBlock | None:
    if len(subset) <= 1:
        return None

    common_orientations = _common_footprint_orientations(subset)
    if not common_orientations:
        return None

    size_x, size_y, allowed_yaws = common_orientations
    signature = "+".join(box.instance_id for box in subset)
    return LayerBlock(
        block_id=f"stack:{signature}",
        target_height=rounded_target,
        box_instance_ids=tuple(box.instance_id for box in subset),
        box_type_ids=tuple(box.box_type_id for box in subset),
        component_count=len(subset),
        base_length=size_x,
        base_width=size_y,
        total_volume=sum(box.volume for box in subset),
        allowed_yaws=allowed_yaws,
        is_composite=True,
    )


def _common_footprint_orientations(subset: list[TaskBox]) -> tuple[float, float, tuple[float, ...]] | None:
    first_box = subset[0]
    common: dict[tuple[float, float], set[float]] = {}
    for yaw in normalized_allowed_yaws(first_box.length, first_box.width, first_box.allowed_yaws):
        oriented = oriented_size(first_box.length, first_box.width, yaw)
        common.setdefault(oriented, set()).add(yaw)

    for box in subset[1:]:
        next_common: dict[tuple[float, float], set[float]] = {}
        for yaw in normalized_allowed_yaws(box.length, box.width, box.allowed_yaws):
            oriented = oriented_size(box.length, box.width, yaw)
            if oriented in common:
                next_common.setdefault(oriented, set()).add(yaw)
        common = next_common
        if not common:
            return None

    best_dims = max(common.keys(), key=lambda dims: (dims[0] * dims[1], dims[0], dims[1]))
    allowed_yaws = (0.0, pi / 2) if abs(best_dims[0] - best_dims[1]) > TOL else (0.0,)
    return best_dims[0], best_dims[1], allowed_yaws




def _generate_layer_choices(
    blocks: list[LayerBlock],
    state: _LayerState,
    pallet: PalletSpec,
    support_placements: list[LayerPlacement] | None = None,
    min_support_ratio: float = 0.9,
) -> list[_PlacementChoice]:
    choices: list[_PlacementChoice] = []
    anchors = _ground_anchors(pallet) if not state.placements else _layer_anchor_points(state.placements)
    for block in blocks:
        if state.used_box_ids.intersection(block.box_instance_ids):
            continue
        for yaw in block.allowed_yaws:
            size_x, size_y = oriented_size(block.base_length, block.base_width, yaw)
            local_corners = rect_corners(size_x, size_y)
            for anchor_x, anchor_y, anchor_index, is_ground in anchors:
                new_corners = local_corners if is_ground else bottom_alignment_corners(local_corners, anchor_index)
                for corner_x, corner_y in new_corners:
                    x = anchor_x - corner_x
                    y = anchor_y - corner_y
                    if not _fits_layer_bounds(x, y, size_x, size_y, pallet):
                        continue
                    placement = LayerPlacement(
                        block_id=block.block_id,
                        box_instance_ids=block.box_instance_ids,
                        box_type_ids=block.box_type_ids,
                        x=x + size_x / 2,
                        y=y + size_y / 2,
                        yaw=yaw,
                        size_x=size_x,
                        size_y=size_y,
                        target_height=block.target_height,
                        component_count=block.component_count,
                        is_composite=block.is_composite,
                    )
                    if _overlaps_layer(placement, state.placements):
                        continue
                    support_ratio = _layer_support_ratio(
                        x=x,
                        y=y,
                        size_x=size_x,
                        size_y=size_y,
                        support_placements=support_placements,
                    )
                    if support_ratio + TOL < min_support_ratio:
                        continue
                    seam_score = _placement_seam_score(placement, state.placements, pallet)
                    next_placements = tuple([*state.placements, placement])
                    next_covered_area = state.covered_area + placement.area
                    future_score = _layer_state_score(next_placements, next_covered_area, pallet)
                    gap_fill_score = _gap_fill_score(placement, state.placements)
                    next_used = state.used_box_ids | frozenset(block.box_instance_ids)
                    remaining_available = sum(
                        1.0 for b in blocks
                        if not next_used.intersection(b.box_instance_ids)
                    )
                    choices.append(
                        _PlacementChoice(
                            block=block,
                            x=x,
                            y=y,
                            yaw=yaw,
                            size_x=size_x,
                            size_y=size_y,
                            support_ratio=support_ratio,
                            score=(
                                future_score[0],
                                future_score[1],
                                gap_fill_score[0],
                                gap_fill_score[1],
                                remaining_available,
                                seam_score[0],
                                seam_score[1],
                                seam_score[2],
                                support_ratio,
                                float(block.component_count),
                            ),
                        )
                    )
    deduped = _dedupe_choices(choices)
    return sorted(deduped, key=lambda choice: choice.score, reverse=True)


def _placement_seam_score(
    candidate: LayerPlacement,
    placements: tuple[LayerPlacement, ...],
    pallet: PalletSpec,
) -> tuple[float, float, float]:
    x_min = candidate.x - candidate.size_x / 2
    x_max = candidate.x + candidate.size_x / 2
    y_min = candidate.y - candidate.size_y / 2
    y_max = candidate.y + candidate.size_y / 2

    exact_contacts = 0.0
    contact_length = 0.0
    for other in placements:
        ox_min = other.x - other.size_x / 2
        ox_max = other.x + other.size_x / 2
        oy_min = other.y - other.size_y / 2
        oy_max = other.y + other.size_y / 2
        overlap_y = max(0.0, min(y_max, oy_max) - max(y_min, oy_min))
        if overlap_y > TOL and (abs(x_min - ox_max) <= TOL or abs(x_max - ox_min) <= TOL):
            contact_length += overlap_y
            exact_contacts += 1.0
        overlap_x = max(0.0, min(x_max, ox_max) - max(x_min, ox_min))
        if overlap_x > TOL and (abs(y_min - oy_max) <= TOL or abs(y_max - oy_min) <= TOL):
            contact_length += overlap_x
            exact_contacts += 1.0

    edge_flush = 0.0
    if abs(x_min) <= TOL or abs(x_max - pallet.length) <= TOL:
        edge_flush += 1.0
    if abs(y_min) <= TOL or abs(y_max - pallet.width) <= TOL:
        edge_flush += 1.0

    bbox_area = _placements_bbox_area([*placements, candidate])
    covered_area = sum(item.area for item in placements) + candidate.area
    void_penalty = max(0.0, bbox_area - covered_area)
    return exact_contacts + edge_flush, contact_length, -void_penalty


def _gap_fill_score(candidate: LayerPlacement, placements: tuple[LayerPlacement, ...]) -> tuple[float, float]:
    if not placements:
        return (0.0, 0.0)

    candidate_x_min = candidate.x - candidate.size_x / 2
    candidate_x_max = candidate.x + candidate.size_x / 2
    candidate_y_min = candidate.y - candidate.size_y / 2
    candidate_y_max = candidate.y + candidate.size_y / 2

    x_mins = [placement.x - placement.size_x / 2 for placement in placements]
    x_maxs = [placement.x + placement.size_x / 2 for placement in placements]
    y_mins = [placement.y - placement.size_y / 2 for placement in placements]
    y_maxs = [placement.y + placement.size_y / 2 for placement in placements]
    bbox_x_min = min(x_mins)
    bbox_x_max = max(x_maxs)
    bbox_y_min = min(y_mins)
    bbox_y_max = max(y_maxs)

    inside_current_bbox = (
        candidate_x_min >= bbox_x_min - TOL
        and candidate_x_max <= bbox_x_max + TOL
        and candidate_y_min >= bbox_y_min - TOL
        and candidate_y_max <= bbox_y_max + TOL
    )

    current_bbox_area = _placements_bbox_area(placements)
    next_bbox_area = _placements_bbox_area([*placements, candidate])
    bbox_growth = next_bbox_area - current_bbox_area
    current_covered = sum(placement.area for placement in placements)
    next_void = max(0.0, next_bbox_area - (current_covered + candidate.area))
    current_void = max(0.0, current_bbox_area - current_covered)
    void_reduction = current_void - next_void
    return (1.0 if inside_current_bbox else 0.0, void_reduction - bbox_growth)


def _layer_support_ratio(
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    support_placements: list[LayerPlacement] | None,
) -> float:
    if not support_placements:
        return 1.0

    x_min = x
    x_max = x + size_x
    y_min = y
    y_max = y + size_y
    support_area = 0.0
    for placement in support_placements:
        other_x_min = placement.x - placement.size_x / 2
        other_x_max = placement.x + placement.size_x / 2
        other_y_min = placement.y - placement.size_y / 2
        other_y_max = placement.y + placement.size_y / 2
        overlap_x = max(0.0, min(x_max, other_x_max) - max(x_min, other_x_min))
        overlap_y = max(0.0, min(y_max, other_y_max) - max(y_min, other_y_min))
        support_area += overlap_x * overlap_y

    base_area = size_x * size_y
    if base_area <= TOL:
        return 0.0
    return support_area / base_area


def _ground_anchors(pallet: PalletSpec) -> list[tuple[float, float, int, bool]]:
    return [
        (0.0, 0.0, 0, True),
        (pallet.length, 0.0, 1, True),
        (pallet.length, pallet.width, 2, True),
        (0.0, pallet.width, 3, True),
    ]


def _layer_anchor_points(placements: Iterable[LayerPlacement]) -> list[tuple[float, float, int, bool]]:
    anchors: list[tuple[float, float, int, bool]] = []
    for placement in placements:
        for index, corner in enumerate(_placement_corners(placement)):
            anchors.append((corner[0], corner[1], index, False))
    return anchors


def _placement_corners(placement: LayerPlacement) -> list[tuple[float, float]]:
    x_min = placement.x - placement.size_x / 2
    x_max = placement.x + placement.size_x / 2
    y_min = placement.y - placement.size_y / 2
    y_max = placement.y + placement.size_y / 2
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]


def _fits_layer_bounds(x: float, y: float, size_x: float, size_y: float, pallet: PalletSpec) -> bool:
    return (
        x >= -TOL
        and y >= -TOL
        and x + size_x <= pallet.length + TOL
        and y + size_y <= pallet.width + TOL
    )


def _overlaps_layer(candidate: LayerPlacement, placements: Iterable[LayerPlacement]) -> bool:
    x_min = candidate.x - candidate.size_x / 2
    x_max = candidate.x + candidate.size_x / 2
    y_min = candidate.y - candidate.size_y / 2
    y_max = candidate.y + candidate.size_y / 2
    for other in placements:
        ox_min = other.x - other.size_x / 2
        ox_max = other.x + other.size_x / 2
        oy_min = other.y - other.size_y / 2
        oy_max = other.y + other.size_y / 2
        overlap_x = min(x_max, ox_max) - max(x_min, ox_min)
        overlap_y = min(y_max, oy_max) - max(y_min, oy_min)
        if overlap_x > TOL and overlap_y > TOL:
            return True
    return False


def _placements_bbox_area(placements: Iterable[LayerPlacement]) -> float:
    placements = list(placements)
    if not placements:
        return 0.0
    x_mins = [placement.x - placement.size_x / 2 for placement in placements]
    x_maxs = [placement.x + placement.size_x / 2 for placement in placements]
    y_mins = [placement.y - placement.size_y / 2 for placement in placements]
    y_maxs = [placement.y + placement.size_y / 2 for placement in placements]
    return (max(x_maxs) - min(x_mins)) * (max(y_maxs) - min(y_mins))




def _layer_state_score(
    placements: tuple[LayerPlacement, ...],
    covered_area: float,
    pallet: PalletSpec,
) -> tuple[float, float, float, float, float, float]:
    if not placements:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pallet_area = pallet.length * pallet.width
    bbox_area = _placements_bbox_area(placements)
    contact_score = sum(_placement_seam_score(placement, tuple(item for item in placements if item != placement), pallet)[1] for placement in placements)
    void_penalty = max(0.0, bbox_area - covered_area)
    fill_ratio = covered_area / max(pallet_area, TOL)
    full_pallet_bonus = 1.0 if fill_ratio >= 1.0 - 1e-9 else 0.0
    return (
        full_pallet_bonus,
        fill_ratio,
        covered_area,
        -void_penalty,
        contact_score,
        -bbox_area,
    )


def _dedupe_choices(choices: list[_PlacementChoice]) -> list[_PlacementChoice]:
    best_by_key: dict[tuple[str, float, float, float], _PlacementChoice] = {}
    for choice in choices:
        key = (
            choice.block.block_id,
            round(choice.x, 9),
            round(choice.y, 9),
            round(choice.yaw, 9),
        )
        existing = best_by_key.get(key)
        if existing is None or choice.score > existing.score:
            best_by_key[key] = choice
    return list(best_by_key.values())


def _dedupe_states(states: list[_LayerState]) -> list[_LayerState]:
    best_by_key: dict[tuple[tuple[float, ...], ...], _LayerState] = {}
    for state in states:
        key = tuple(
            sorted(
                (
                    round(placement.x, 9),
                    round(placement.y, 9),
                    round(placement.size_x, 9),
                    round(placement.size_y, 9),
                    round(placement.yaw, 9),
                )
                for placement in state.placements
            )
        )
        existing = best_by_key.get(key)
        if existing is None or state.score > existing.score:
            best_by_key[key] = state
    return list(best_by_key.values())



def _layer_result_key(
    result: LayerPatternResult,
) -> tuple[float, int, int, float]:
    return (
        result.utilization_2d,
        result.used_box_count,
        result.packed_block_count,
        -result.target_height,
    )


# ---------------------------------------------------------------------------
# Fill2D-based layer planning (CP-SAT exact solver)
# ---------------------------------------------------------------------------


MAX_HEIGHT_CANDIDATES = 4


def plan_best_layer_pattern_fill2d(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    max_stack_size: int = 3,
    support_placements: list[LayerPlacement] | None = None,
    min_support_ratio: float = 0.9,
    time_limit_seconds: float = 30.0,
    num_workers: int = 8,
) -> LayerPatternResult:
    target_heights = sorted({_round_height(box.height) for box in boxes})
    if not target_heights:
        raise ValueError("boxes must not be empty")

    # Pre-build blocks for every height and score them.
    # Score = total footprint area of all blocks (single + composite).
    # This estimates how much pallet area this height could potentially cover.
    blocks_by_height: dict[float, list[LayerBlock]] = {}
    height_scores: dict[float, float] = {}
    for h in target_heights:
        blks = build_layer_blocks(boxes, h, max_stack_size)
        blocks_by_height[h] = blks
        # Sum of unique-box-weighted area: prefer heights that can pack
        # many distinct boxes over many overlapping composites.
        unique_box_ids: set[str] = set()
        for b in blks:
            unique_box_ids.update(b.box_instance_ids)
        height_scores[h] = len(unique_box_ids)

    # Only attempt the top-N most promising heights.
    ranked = sorted(
        target_heights,
        key=lambda h: height_scores[h],
        reverse=True,
    )
    candidates = ranked[:MAX_HEIGHT_CANDIDATES]

    # Distribute time equally among the selected heights.
    per_height_limit = max(
        time_limit_seconds * 0.85 / max(len(candidates), 1),
        1.0,
    )

    best_result: LayerPatternResult | None = None
    for target_height in candidates:
        result = plan_layer_pattern_fill2d_for_height(
            boxes=boxes,
            pallet=pallet,
            target_height=target_height,
            max_stack_size=max_stack_size,
            support_placements=support_placements,
            min_support_ratio=min_support_ratio,
            time_limit_seconds=per_height_limit,
            num_workers=num_workers,
        )
        if (
            best_result is None
            or _layer_result_key(result)
            > _layer_result_key(best_result)
        ):
            best_result = result

    assert best_result is not None

    # --- Gap-fill pass: try to pack remaining boxes into the gaps ----------
    primary_box_ids = set(best_result.used_box_instance_ids)
    remaining = [b for b in boxes if b.instance_id not in primary_box_ids]
    if remaining and best_result.utilization_2d < 1.0 - TOL:
        gap_time = max(time_limit_seconds * 0.15, 2.0)
        gap_result = _fill_gaps(
            remaining_boxes=remaining,
            primary_placements=best_result.placements,
            pallet=pallet,
            target_height=best_result.target_height,
            support_placements=support_placements,
            min_support_ratio=min_support_ratio,
            time_limit_seconds=gap_time,
            num_workers=num_workers,
        )
        if gap_result:
            all_placements = best_result.placements + gap_result
            all_box_ids = sorted(
                bid for p in all_placements for bid in p.box_instance_ids
            )
            covered = sum(p.area for p in all_placements)
            pallet_area = pallet.length * pallet.width
            best_result = LayerPatternResult(
                target_height=best_result.target_height,
                placements=all_placements,
                packed_block_count=len(all_placements),
                used_box_count=len(all_box_ids),
                used_box_instance_ids=all_box_ids,
                covered_area=covered,
                utilization_2d=covered / pallet_area if pallet_area > TOL else 0.0,
                message=best_result.message + f" +{len(gap_result)} gap-fill.",
            )

    return best_result


def plan_layer_pattern_fill2d_for_height(
    boxes: list[TaskBox],
    pallet: PalletSpec,
    target_height: float,
    max_stack_size: int = 3,
    support_placements: list[LayerPlacement] | None = None,
    min_support_ratio: float = 0.9,
    time_limit_seconds: float = 10.0,
    num_workers: int = 8,
) -> LayerPatternResult:
    if min(pallet.length, pallet.width) <= 0:
        raise ValueError("pallet footprint must be positive")

    rounded_target = _round_height(target_height)

    # Reuse build_layer_blocks to get both single boxes and composite stacks.
    blocks = build_layer_blocks(boxes, rounded_target, max_stack_size=max_stack_size)
    if not blocks:
        return _empty_layer_result(rounded_target)

    # Convert each block to a Fill2DItem using its base footprint.
    items: list[Fill2DItem] = []
    for block in blocks:
        can_rotate = len(block.allowed_yaws) > 1
        items.append(
            Fill2DItem(
                item_id=block.block_id,
                width=round(block.base_length * FILL2D_SCALE),
                height=round(block.base_width * FILL2D_SCALE),
                allow_rotation=can_rotate,
                # Prefer composites: weight area by the number of boxes packed.
                value=round(block.base_area * block.component_count * FILL2D_SCALE * FILL2D_SCALE),
            )
        )

    # Build exclusion groups: blocks sharing any box_instance_id can't coexist.
    box_id_to_indices: dict[str, list[int]] = {}
    for idx, block in enumerate(blocks):
        for bid in block.box_instance_ids:
            box_id_to_indices.setdefault(bid, []).append(idx)
    exclusion_groups = [
        indices for indices in box_id_to_indices.values() if len(indices) > 1
    ]

    instance = Fill2DInstance(
        pallet_width=round(pallet.length * FILL2D_SCALE),
        pallet_height=round(pallet.width * FILL2D_SCALE),
        items=tuple(items),
    )
    result = solve_fill2d(
        instance, time_limit_seconds, num_workers,
        exclusion_groups=exclusion_groups,
    )

    # Map solver placements back to LayerPlacements.
    block_by_id = {b.block_id: b for b in blocks}
    raw_placements: list[LayerPlacement] = []
    for fp in result.placements:
        block = block_by_id[fp.item_id]
        size_x = fp.width / FILL2D_SCALE
        size_y = fp.height / FILL2D_SCALE
        if (
            abs(size_x - block.base_length) < TOL
            and abs(size_y - block.base_width) < TOL
        ):
            yaw = 0.0
        else:
            yaw = pi / 2
        raw_placements.append(
            LayerPlacement(
                block_id=block.block_id,
                box_instance_ids=block.box_instance_ids,
                box_type_ids=block.box_type_ids,
                x=(fp.x + fp.width / 2) / FILL2D_SCALE,
                y=(fp.y + fp.height / 2) / FILL2D_SCALE,
                yaw=yaw,
                size_x=size_x,
                size_y=size_y,
                target_height=rounded_target,
                component_count=block.component_count,
                is_composite=block.is_composite,
            )
        )

    # Post-filter by support constraint (upper layers must rest on lower layer).
    if support_placements:
        placements = [
            p for p in raw_placements
            if _layer_support_ratio(
                x=p.x - p.size_x / 2,
                y=p.y - p.size_y / 2,
                size_x=p.size_x,
                size_y=p.size_y,
                support_placements=support_placements,
            ) + TOL >= min_support_ratio
        ]
    else:
        placements = raw_placements

    used_box_ids = sorted(
        bid for p in placements for bid in p.box_instance_ids
    )
    covered_area = sum(p.area for p in placements)
    pallet_area = pallet.length * pallet.width

    return LayerPatternResult(
        target_height=rounded_target,
        placements=placements,
        packed_block_count=len(placements),
        used_box_count=len(used_box_ids),
        used_box_instance_ids=used_box_ids,
        covered_area=covered_area,
        utilization_2d=covered_area / pallet_area if pallet_area > TOL else 0.0,
        message=f"Fill2D {result.status}: {len(placements)} blocks at h={target_height:.2f}m.",
    )


def _empty_layer_result(target_height: float) -> LayerPatternResult:
    return LayerPatternResult(
        target_height=target_height,
        placements=[],
        packed_block_count=0,
        used_box_count=0,
        used_box_instance_ids=[],
        covered_area=0.0,
        utilization_2d=0.0,
        message="No boxes match the requested target height.",
    )


def _fill_gaps(
    remaining_boxes: list[TaskBox],
    primary_placements: list[LayerPlacement],
    pallet: PalletSpec,
    target_height: float,
    support_placements: list[LayerPlacement] | None,
    min_support_ratio: float,
    time_limit_seconds: float,
    num_workers: int,
) -> list[LayerPlacement]:
    """Pack remaining boxes into gaps left by the primary layer placements.

    Only single boxes (no composites) are considered — any box whose height
    does not exceed the layer's target_height can be used as gap filler.
    """
    from fill2d import Fill2DPlacement as _FP

    eligible = [b for b in remaining_boxes if b.height <= target_height + TOL]
    if not eligible:
        return []

    # Limit candidates to keep CP-SAT tractable — prefer smaller boxes that
    # are more likely to fit into irregular gaps.
    MAX_GAP_CANDIDATES = 30
    eligible.sort(key=lambda b: (b.length * b.width, b.volume))
    eligible = eligible[:MAX_GAP_CANDIDATES]

    # Convert primary placements to fixed Fill2DPlacements (obstacles).
    fixed: list[_FP] = []
    for p in primary_placements:
        fixed.append(
            _FP(
                item_id=p.block_id,
                x=round((p.x - p.size_x / 2) * FILL2D_SCALE),
                y=round((p.y - p.size_y / 2) * FILL2D_SCALE),
                rotated=False,
                width=round(p.size_x * FILL2D_SCALE),
                height=round(p.size_y * FILL2D_SCALE),
            )
        )

    # Build Fill2DItems for eligible remaining boxes.
    items: list[Fill2DItem] = []
    for box in eligible:
        can_rotate = (
            abs(box.length - box.width) > TOL
            and any(abs(y - pi / 2) < TOL for y in box.allowed_yaws)
        )
        items.append(
            Fill2DItem(
                item_id=box.instance_id,
                width=round(box.length * FILL2D_SCALE),
                height=round(box.width * FILL2D_SCALE),
                allow_rotation=can_rotate,
            )
        )

    instance = Fill2DInstance(
        pallet_width=round(pallet.length * FILL2D_SCALE),
        pallet_height=round(pallet.width * FILL2D_SCALE),
        items=tuple(items),
    )
    result = solve_fill2d(
        instance, time_limit_seconds, num_workers,
        fixed_placements=fixed,
    )

    if not result.placements:
        return []

    box_by_id = {b.instance_id: b for b in eligible}
    gap_placements: list[LayerPlacement] = []
    for fp in result.placements:
        task_box = box_by_id[fp.item_id]
        size_x = fp.width / FILL2D_SCALE
        size_y = fp.height / FILL2D_SCALE
        if (
            abs(size_x - task_box.length) < TOL
            and abs(size_y - task_box.width) < TOL
        ):
            yaw = 0.0
        else:
            yaw = pi / 2

        # Support check for upper layers.
        if support_placements:
            sr = _layer_support_ratio(
                x=fp.x / FILL2D_SCALE,
                y=fp.y / FILL2D_SCALE,
                size_x=size_x,
                size_y=size_y,
                support_placements=support_placements,
            )
            if sr + TOL < min_support_ratio:
                continue

        gap_placements.append(
            LayerPlacement(
                block_id=f"gap:{fp.item_id}",
                box_instance_ids=(fp.item_id,),
                box_type_ids=(task_box.box_type_id,),
                x=(fp.x + fp.width / 2) / FILL2D_SCALE,
                y=(fp.y + fp.height / 2) / FILL2D_SCALE,
                yaw=yaw,
                size_x=size_x,
                size_y=size_y,
                target_height=_round_height(target_height),
                component_count=1,
                is_composite=False,
                is_gap_fill=True,
            )
        )

    return gap_placements
