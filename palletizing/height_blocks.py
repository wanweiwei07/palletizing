from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .task_generator import TaskBox

TOL = 1e-9


def _round_height(value: float, digits: int = 6) -> float:
    return round(value, digits)


@dataclass(frozen=True)
class HeightGroup:
    target_height: float
    box_instance_ids: tuple[str, ...]
    box_type_counts: dict[str, int]
    count: int


@dataclass(frozen=True)
class BlockCandidate:
    target_height: float
    box_instance_ids: tuple[str, ...]
    box_type_ids: tuple[str, ...]
    component_heights: tuple[float, ...]
    component_count: int
    total_volume: float

    def as_dict(self) -> dict[str, object]:
        return {
            "target_height": self.target_height,
            "box_instance_ids": list(self.box_instance_ids),
            "box_type_ids": list(self.box_type_ids),
            "component_heights": list(self.component_heights),
            "component_count": self.component_count,
            "total_volume": self.total_volume,
        }


@dataclass(frozen=True)
class HeightAnalysis:
    height_groups: tuple[HeightGroup, ...]
    block_candidates_by_target: dict[float, tuple[BlockCandidate, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "height_groups": [
                {
                    "target_height": group.target_height,
                    "count": group.count,
                    "box_instance_ids": list(group.box_instance_ids),
                    "box_type_counts": group.box_type_counts,
                }
                for group in self.height_groups
            ],
            "block_candidates_by_target": {
                f"{target_height:.6f}": [candidate.as_dict() for candidate in candidates]
                for target_height, candidates in sorted(self.block_candidates_by_target.items())
            },
        }


def analyze_box_heights(boxes: list[TaskBox], max_combination_size: int = 3) -> HeightAnalysis:
    if max_combination_size < 2:
        raise ValueError("max_combination_size must be at least 2")

    grouped_boxes: dict[float, list[TaskBox]] = {}
    for box in boxes:
        grouped_boxes.setdefault(_round_height(box.height), []).append(box)

    height_groups = tuple(
        HeightGroup(
            target_height=target_height,
            box_instance_ids=tuple(box.instance_id for box in group_boxes),
            box_type_counts=_count_box_types(group_boxes),
            count=len(group_boxes),
        )
        for target_height, group_boxes in sorted(grouped_boxes.items())
    )

    target_heights = tuple(grouped_boxes.keys())
    combinations_by_target: dict[float, list[BlockCandidate]] = {height: [] for height in target_heights}
    seen_signatures: dict[float, set[tuple[str, ...]]] = {height: set() for height in target_heights}

    max_size = min(max_combination_size, len(boxes))
    for combination_size in range(2, max_size + 1):
        for subset in combinations(boxes, combination_size):
            total_height = _round_height(sum(box.height for box in subset))
            if total_height not in combinations_by_target:
                continue

            signature = tuple(sorted(box.instance_id for box in subset))
            if signature in seen_signatures[total_height]:
                continue
            seen_signatures[total_height].add(signature)

            combinations_by_target[total_height].append(
                BlockCandidate(
                    target_height=total_height,
                    box_instance_ids=signature,
                    box_type_ids=tuple(box.box_type_id for box in subset),
                    component_heights=tuple(box.height for box in subset),
                    component_count=combination_size,
                    total_volume=sum(box.volume for box in subset),
                )
            )

    finalized_candidates = {
        target_height: tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.component_count,
                    candidate.total_volume,
                    candidate.box_instance_ids,
                ),
            )
        )
        for target_height, candidates in combinations_by_target.items()
        if candidates
    }

    return HeightAnalysis(
        height_groups=height_groups,
        block_candidates_by_target=finalized_candidates,
    )


def _count_box_types(boxes: list[TaskBox]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.box_type_id] = counts.get(box.box_type_id, 0) + 1
    return dict(sorted(counts.items()))
