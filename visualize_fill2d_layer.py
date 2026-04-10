from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pyglet
import pyglet.window.key as key

from fill2d import Fill2DInstance, Fill2DItem, Fill2DPlacement, solve_fill2d
from palletizing import PalletSpec, generate_multitype_task, load_task_boxes
from palletizing.task_generator import TaskBox
import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


SCALE = 100  # meters → centimeters (integer grid for CP-SAT)
TOL = 1e-9

ITEM_COLORS = [
    const.Tab20.BLUE_LIGHT,
    const.Tab20.ORANGE_LIGHT,
    const.Tab20.GREEN_LIGHT,
    const.Tab20.RED_LIGHT,
    const.Tab20.PURPLE_LIGHT,
    const.Tab20.BROWN_LIGHT,
    const.Tab20.PINK_LIGHT,
    const.Tab20.CYAN_LIGHT,
    const.Tab20.OLIVE_LIGHT,
    const.Tab20.BLUE_DEEP,
    const.Tab20.ORANGE_DEEP,
    const.Tab20.GREEN_DEEP,
    const.Tab20.RED_DEEP,
    const.Tab20.PURPLE_DEEP,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick one height group from a task, solve 2D packing with CP-SAT, visualize.",
    )
    parser.add_argument("--task-file", type=str, default=None,
                        help="Path to a task JSON. If omitted, generate with --seed.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=70)
    parser.add_argument("--target-height", type=float, default=None,
                        help="Which height group to pack. Default: most common height.")
    parser.add_argument("--pallet-length", type=float, default=1.2)
    parser.add_argument("--pallet-width", type=float, default=1.0)
    parser.add_argument("--pallet-thickness", type=float, default=0.02,
                        help="Visual pallet base thickness in meters.")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--step-interval", type=float, default=0.3)
    parser.add_argument("--start-paused", action="store_true")
    return parser


def _select_height_group(
    boxes: list[TaskBox], target_height: float | None,
) -> tuple[float, list[TaskBox]]:
    height_counts = Counter(round(b.height, 6) for b in boxes)
    if target_height is not None:
        chosen = round(target_height, 6)
    else:
        chosen = height_counts.most_common(1)[0][0]
    group = [b for b in boxes if abs(b.height - chosen) < TOL]
    return chosen, group


def _boxes_to_fill2d_instance(
    group: list[TaskBox],
    pallet_length: float,
    pallet_width: float,
) -> Fill2DInstance:
    items: list[Fill2DItem] = []
    for box in group:
        items.append(
            Fill2DItem(
                item_id=box.instance_id,
                width=round(box.length * SCALE),
                height=round(box.width * SCALE),
                allow_rotation=True,
            )
        )
    return Fill2DInstance(
        pallet_width=round(pallet_length * SCALE),
        pallet_height=round(pallet_width * SCALE),
        items=tuple(items),
    )


def _color_for_box_type(box_type_id: str, type_list: list[str]) -> np.ndarray:
    if box_type_id not in type_list:
        type_list.append(box_type_id)
    idx = type_list.index(box_type_id)
    return ITEM_COLORS[idx % len(ITEM_COLORS)]


def _add_pallet_base(scene, pallet_length: float, pallet_width: float, thickness: float) -> None:
    pallet_size = np.array([pallet_length, pallet_width, thickness], dtype=np.float32)
    pallet_center = pallet_size / 2
    prim.box(
        pos=pallet_center,
        half_extents=pallet_size / 2,
        rgb=const.Tab20.BROWN_DEEP,
        alpha=1.0,
    ).attach_to(scene)
    prim.plane(
        pos=(pallet_length / 2, pallet_width / 2, 0.0),
        size=(pallet_length * 2.0, pallet_width * 2.0),
        rgb=const.ExtendedColor.MOON_GRAY,
        alpha=1.0,
    ).attach_to(scene)


class Fill2DLayerViewer(world_view.World):
    def __init__(
        self,
        placements: list[Fill2DPlacement],
        group: list[TaskBox],
        target_height: float,
        pallet_length: float,
        pallet_width: float,
        pallet_thickness: float,
        coverage_ratio: float,
        status: str,
        step_interval: float,
        start_paused: bool,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
    ):
        super().__init__(
            cam_pos=cam_pos,
            cam_lookat_pos=cam_lookat_pos,
            toggle_auto_cam_orbit=False,
        )
        self.placements = placements
        self.group = group
        self.target_height = target_height
        self.pallet_thickness = pallet_thickness
        self.coverage_ratio = coverage_ratio
        self.solve_status = status
        self.current_index = 0
        self.paused = start_paused
        self.total = len(placements)
        self.type_list: list[str] = []

        # pre-build box lookup
        self.box_by_id = {b.instance_id: b for b in group}

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
        _add_pallet_base(self.scene, pallet_length, pallet_width, pallet_thickness)
        self._update_status()
        self.schedule_interval(self._advance, interval=max(step_interval, 0.01))

    def _advance(self, dt: float) -> None:
        if self.paused or self.current_index >= self.total:
            self._update_status()
            return
        self._add_next()

    def _add_next(self) -> None:
        if self.current_index >= self.total:
            self._update_status()
            return
        p = self.placements[self.current_index]
        task_box = self.box_by_id[p.item_id]
        base_z = self.pallet_thickness

        # convert back from integer grid to meters
        cx = (p.x + p.width / 2) / SCALE
        cy = (p.y + p.height / 2) / SCALE
        sx = p.width / SCALE
        sy = p.height / SCALE
        sz = self.target_height

        center = np.array([cx, cy, base_z + sz / 2], dtype=np.float32)
        half = np.array([sx / 2, sy / 2, sz / 2], dtype=np.float32)
        color = _color_for_box_type(task_box.box_type_id, self.type_list)
        prim.box(
            pos=center,
            half_extents=half,
            rgb=color,
            alpha=1.0,
        ).attach_to(self.scene)

        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        self.status_label.text = (
            f"boxes: {self.current_index}/{self.total} | "
            f"height: {self.target_height:.2f}m | "
            f"solver: {self.solve_status} | "
            f"coverage: {self.coverage_ratio:.1%} | "
            f"state: {state} | "
            "space: pause/resume  right: next  r: restart"
        )
        self.set_caption(
            f"Fill2D Layer (h={self.target_height:.2f}m) | "
            f"{self.current_index}/{self.total} | "
            f"{self.coverage_ratio:.1%}"
        )

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.paused = not self.paused
            self._update_status()
        elif symbol == key.RIGHT:
            self._add_next()
        elif symbol == key.R:
            self._restart()

    def _restart(self) -> None:
        for sobj in tuple(self.scene.sobjs)[2:]:
            self.scene.remove(sobj)
        self.current_index = 0
        self.paused = True
        self.type_list.clear()
        self._update_status()


def main() -> None:
    args = build_parser().parse_args()

    if args.task_file:
        boxes = load_task_boxes(args.task_file)
    else:
        task = generate_multitype_task(count=args.count, seed=args.seed)
        boxes = task.boxes

    target_height, group = _select_height_group(boxes, args.target_height)
    print(f"Height group: {target_height:.4f}m  ({len(group)} boxes)")
    for b in group:
        print(f"  {b.instance_id}  type={b.box_type_id}  "
              f"L={b.length}  W={b.width}  H={b.height}")

    instance = _boxes_to_fill2d_instance(group, args.pallet_length, args.pallet_width)
    print(f"\nSolving fill2d: pallet={instance.pallet_width}x{instance.pallet_height} "
          f"({len(instance.items)} items, time_limit={args.time_limit}s) ...")
    result = solve_fill2d(instance, time_limit_seconds=args.time_limit, num_workers=args.workers)
    print(f"Status: {result.status}  placed: {result.used_item_count}/{len(group)}  "
          f"coverage: {result.coverage_ratio:.1%}")

    pl = args.pallet_length
    pw = args.pallet_width
    stack_h = args.pallet_thickness + target_height
    cam_pos = (pl * 1.6, -pw * 1.4, stack_h * 3.5)
    cam_lookat = (pl / 2, pw / 2, stack_h / 2)

    viewer = Fill2DLayerViewer(
        placements=result.placements,
        group=group,
        target_height=target_height,
        pallet_length=pl,
        pallet_width=pw,
        pallet_thickness=args.pallet_thickness,
        coverage_ratio=result.coverage_ratio,
        status=result.status,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
    )
    viewer.run()


if __name__ == "__main__":
    main()
