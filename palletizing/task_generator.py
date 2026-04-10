from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import BOX_TYPE_CATALOG, BoxType


@dataclass(frozen=True)
class TaskBox:
    instance_id: str
    box_type_id: str
    length: float
    width: float
    height: float
    frequency_group: str
    allowed_yaws: tuple[float, float]

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["volume"] = self.volume
        return payload


@dataclass(frozen=True)
class GeneratedTask:
    boxes: list[TaskBox]
    seed: int | None
    summary: dict[str, object]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "summary": self.summary,
            "boxes": [box.as_dict() for box in self.boxes],
        }


def generate_multitype_task(
    count: int = 70,
    seed: int | None = None,
    catalog: tuple[BoxType, ...] = BOX_TYPE_CATALOG,
) -> GeneratedTask:
    if count <= 0:
        raise ValueError("count must be positive")
    if not catalog:
        raise ValueError("catalog must not be empty")

    rng = random.Random(seed)
    weights = [box_type.sampling_weight for box_type in catalog]
    sampled_types = rng.choices(catalog, weights=weights, k=count)

    boxes = [
        TaskBox(
            instance_id=f"box_{index + 1:03d}",
            box_type_id=box_type.box_type_id,
            length=box_type.length,
            width=box_type.width,
            height=box_type.height,
            frequency_group=box_type.frequency_group,
            allowed_yaws=box_type.allowed_yaws,
        )
        for index, box_type in enumerate(sampled_types)
    ]

    type_counter = Counter(box.box_type_id for box in boxes)
    group_counter = Counter(box.frequency_group for box in boxes)
    total_volume = sum(box.volume for box in boxes)
    summary = {
        "count": count,
        "unique_box_type_count": len(type_counter),
        "box_type_counts": dict(sorted(type_counter.items())),
        "frequency_group_counts": dict(sorted(group_counter.items())),
        "total_volume": total_volume,
    }
    return GeneratedTask(boxes=boxes, seed=seed, summary=summary)


def load_task_boxes(task_file: str | Path) -> list[TaskBox]:
    task_path = Path(task_file)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    raw_boxes = payload.get("boxes")
    if not isinstance(raw_boxes, list):
        raise ValueError("task file must contain a 'boxes' list")

    boxes: list[TaskBox] = []
    for raw_box in raw_boxes:
        boxes.append(
            TaskBox(
                instance_id=str(raw_box["instance_id"]),
                box_type_id=str(raw_box["box_type_id"]),
                length=float(raw_box["length"]),
                width=float(raw_box["width"]),
                height=float(raw_box["height"]),
                frequency_group=str(raw_box["frequency_group"]),
                allowed_yaws=tuple(float(value) for value in raw_box["allowed_yaws"]),
            )
        )
    return boxes
