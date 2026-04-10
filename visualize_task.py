from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pyglet
import pyglet.window.key as key

import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


@dataclass(frozen=True)
class PreviewBox:
    instance_id: str
    box_type_id: str
    length: float
    width: float
    height: float
    frequency_group: str
    yaw: float
    x: float
    y: float
    z: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a generated multitype task in the one viewer.")
    parser.add_argument("--task-file", type=str, default="task_70_seed7.json")
    parser.add_argument("--unit-scale", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=0.04, help="Gap between boxes in preview layout, in meters.")
    parser.add_argument("--row-width", type=float, default=3.0, help="Maximum preview row width, in meters.")
    parser.add_argument("--step-interval", type=float, default=0.15, help="Seconds between two boxes.")
    parser.add_argument("--start-paused", action="store_true")
    return parser


def _color_for_group(group: str) -> np.ndarray:
    mapping = {
        "high": const.Tab20.BLUE_LIGHT,
        "medium": const.Tab20.ORANGE_LIGHT,
        "low": const.Tab20.GREEN_LIGHT,
    }
    return mapping.get(group, const.Tab20.GRAY_LIGHT)


def _load_preview_boxes(task_file: str, spacing: float) -> list[PreviewBox]:
    with open(task_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    preview_boxes: list[PreviewBox] = []
    cursor_x = 0.0
    cursor_y = 0.0
    current_row_depth = 0.0
    row_width = payload.get("summary", {}).get("preview_row_width", None)

    for raw_box in payload["boxes"]:
        length = float(raw_box["length"])
        width = float(raw_box["width"])
        height = float(raw_box["height"])

        preview_boxes.append(
            PreviewBox(
                instance_id=raw_box["instance_id"],
                box_type_id=raw_box["box_type_id"],
                length=length,
                width=width,
                height=height,
                frequency_group=raw_box["frequency_group"],
                yaw=0.0,
                x=cursor_x + length / 2,
                y=cursor_y + width / 2,
                z=height / 2,
            )
        )
        cursor_x += length + spacing
        current_row_depth = max(current_row_depth, width)

        if row_width is not None and cursor_x >= float(row_width):
            cursor_x = 0.0
            cursor_y += current_row_depth + spacing
            current_row_depth = 0.0

    return preview_boxes


def _layout_preview_boxes(task_file: str, spacing: float, row_width: float) -> list[PreviewBox]:
    with open(task_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    preview_boxes: list[PreviewBox] = []
    cursor_x = 0.0
    cursor_y = 0.0
    current_row_depth = 0.0

    for raw_box in payload["boxes"]:
        length = float(raw_box["length"])
        width = float(raw_box["width"])
        height = float(raw_box["height"])

        if cursor_x > 0.0 and cursor_x + length > row_width:
            cursor_x = 0.0
            cursor_y += current_row_depth + spacing
            current_row_depth = 0.0

        preview_boxes.append(
            PreviewBox(
                instance_id=raw_box["instance_id"],
                box_type_id=raw_box["box_type_id"],
                length=length,
                width=width,
                height=height,
                frequency_group=raw_box["frequency_group"],
                yaw=0.0,
                x=cursor_x + length / 2,
                y=cursor_y + width / 2,
                z=height / 2,
            )
        )
        cursor_x += length + spacing
        current_row_depth = max(current_row_depth, width)

    return preview_boxes


def _scene_extent(boxes: list[PreviewBox]) -> tuple[float, float, float]:
    max_x = max(box.x + box.length / 2 for box in boxes)
    max_y = max(box.y + box.width / 2 for box in boxes)
    max_z = max(box.height for box in boxes)
    return max_x, max_y, max_z


def _make_box_object(box: PreviewBox, unit_scale: float):
    center = np.array([box.x * unit_scale, box.y * unit_scale, box.z * unit_scale], dtype=np.float32)
    local_size_x = box.length
    local_size_y = box.width
    if abs(box.yaw - np.pi / 2) < 1e-9:
        local_size_x, local_size_y = box.width, box.length
    half_extents = np.array(
        [local_size_x * unit_scale / 2, local_size_y * unit_scale / 2, box.height * unit_scale / 2],
        dtype=np.float32,
    )
    rotmat = math_utils.rotmat_from_axangle(const.StandardAxis.Z, box.yaw)
    return prim.box(
        pos=center,
        half_extents=half_extents,
        rotmat=rotmat,
        rgb=_color_for_group(box.frequency_group),
        alpha=1.0,
    )


class AnimatedTaskViewer(world_view.World):
    def __init__(
        self,
        boxes: list[PreviewBox],
        unit_scale: float,
        step_interval: float,
        start_paused: bool,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
        ground_size: tuple[float, float],
    ):
        super().__init__(cam_pos=cam_pos, cam_lookat_pos=cam_lookat_pos, toggle_auto_cam_orbit=False)
        self.boxes = boxes
        self.unit_scale = unit_scale
        self.step_interval = step_interval
        self.current_index = 0
        self.paused = start_paused
        self.total = len(boxes)
        self.status_label = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=14,
            x=12,
            y=12,
            anchor_x="left",
            anchor_y="bottom",
            color=(20, 20, 20, 255),
        )

        ground = prim.plane(
            pos=(ground_size[0] / 2, ground_size[1] / 2, 0.0),
            size=ground_size,
            rgb=const.ExtendedColor.MOON_GRAY,
            alpha=1.0,
        )
        ground.attach_to(self.scene)
        self._update_status()
        self.schedule_interval(self._advance_animation, interval=max(step_interval, 0.01))

    def _advance_animation(self, dt: float) -> None:
        if self.paused or self.current_index >= self.total:
            self._update_status()
            return
        self._add_next_box()

    def _add_next_box(self) -> None:
        if self.current_index >= self.total:
            self._update_status()
            return
        box = self.boxes[self.current_index]
        _make_box_object(box, self.unit_scale).attach_to(self.scene)
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        next_box = self.boxes[self.current_index] if self.current_index < self.total else self.boxes[-1]
        self.status_label.text = (
            f"shown: {self.current_index}/{self.total} | "
            f"next: {next_box.instance_id} {next_box.box_type_id} | "
            f"state: {state} | "
            "space: pause/resume, right: next, r: restart"
        )
        self.set_caption(f"Task Preview | shown {self.current_index}/{self.total}")

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)
        if symbol == key.SPACE:
            self.paused = not self.paused
            self._update_status()
        elif symbol == key.RIGHT:
            self._add_next_box()
        elif symbol == key.R:
            self._restart()

    def _restart(self) -> None:
        for sobj in tuple(self.scene.sobjs)[1:]:
            self.scene.remove(sobj)
        self.current_index = 0
        self.paused = True
        self._update_status()


def main() -> None:
    args = build_parser().parse_args()
    boxes = _layout_preview_boxes(args.task_file, spacing=args.spacing, row_width=args.row_width)
    extent_x, extent_y, extent_z = _scene_extent(boxes)

    ground_size = (max(extent_x * args.unit_scale + 0.5, 2.0), max(extent_y * args.unit_scale + 0.5, 2.0))
    cam_lookat = (ground_size[0] / 2, ground_size[1] / 2, extent_z * args.unit_scale / 2)
    cam_pos = (
        ground_size[0] * 0.65,
        -ground_size[1] * 0.85,
        max(extent_z * args.unit_scale * 8.0, 2.0),
    )

    viewer = AnimatedTaskViewer(
        boxes=boxes,
        unit_scale=args.unit_scale,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
        ground_size=ground_size,
    )
    viewer.run()


if __name__ == "__main__":
    main()
