from __future__ import annotations

import argparse

import numpy as np
import pyglet
import pyglet.window.key as key

from palletizing import MultiTypePlacement, PalletSpec, load_and_plan_multitype_task
import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a multitype task packed onto a real pallet.")
    parser.add_argument("--task-file", type=str, default="task_70_seed7.json")
    parser.add_argument("--pallet-length", type=float, default=1.2)
    parser.add_argument("--pallet-width", type=float, default=1.0)
    parser.add_argument("--pallet-max-height", type=float, default=1.5)
    parser.add_argument("--pallet-thickness", type=float, default=0.144)
    parser.add_argument("--step-interval", type=float, default=0.2)
    parser.add_argument("--start-paused", action="store_true")
    return parser


def _color_for_group(group: str) -> np.ndarray:
    mapping = {
        "high": const.Tab20.BLUE_LIGHT,
        "medium": const.Tab20.ORANGE_LIGHT,
        "low": const.Tab20.GREEN_LIGHT,
    }
    return mapping.get(group, const.Tab20.GRAY_LIGHT)


def _make_box_object(placement: MultiTypePlacement, pallet_thickness: float):
    center = np.array(
        [placement.x, placement.y, placement.z + pallet_thickness],
        dtype=np.float32,
    )
    local_size_x = placement.size_x
    local_size_y = placement.size_y
    if abs(placement.yaw - np.pi / 2) < 1e-9:
        local_size_x, local_size_y = placement.size_y, placement.size_x
    half_extents = np.array(
        [local_size_x / 2, local_size_y / 2, placement.size_z / 2],
        dtype=np.float32,
    )
    rotmat = math_utils.rotmat_from_axangle(const.StandardAxis.Z, placement.yaw)
    return prim.box(
        pos=center,
        half_extents=half_extents,
        rotmat=rotmat,
        rgb=_color_for_group(placement.frequency_group),
        alpha=1.0,
    )


def add_pallet(scene, pallet: PalletSpec, pallet_thickness: float) -> None:
    pallet_size = np.array([pallet.length, pallet.width, pallet_thickness], dtype=np.float32)
    pallet_center = np.array([pallet.length / 2, pallet.width / 2, pallet_thickness / 2], dtype=np.float32)
    prim.box(
        pos=pallet_center,
        half_extents=pallet_size / 2,
        rgb=const.Tab20.BROWN_DEEP,
        alpha=1.0,
    ).attach_to(scene)
    prim.plane(
        pos=(pallet.length / 2, pallet.width / 2, 0.0),
        size=(pallet.length * 1.8, pallet.width * 1.8),
        rgb=const.ExtendedColor.MOON_GRAY,
        alpha=1.0,
    ).attach_to(scene)


class AnimatedPackedTaskViewer(world_view.World):
    def __init__(
        self,
        placements: list[MultiTypePlacement],
        pallet: PalletSpec,
        pallet_thickness: float,
        step_interval: float,
        start_paused: bool,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
        unpacked_count: int,
    ):
        super().__init__(cam_pos=cam_pos, cam_lookat_pos=cam_lookat_pos, toggle_auto_cam_orbit=False)
        self.placements = placements
        self.pallet_thickness = pallet_thickness
        self.current_index = 0
        self.paused = start_paused
        self.total = len(placements)
        self.unpacked_count = unpacked_count
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
        add_pallet(self.scene, pallet, pallet_thickness)
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
        placement = self.placements[self.current_index]
        _make_box_object(placement, self.pallet_thickness).attach_to(self.scene)
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        if self.total == 0:
            self.status_label.text = "packed: 0/0 | unpacked: 0 | state: empty"
            self.set_caption("Packed Task | empty")
            return
        next_item = self.placements[self.current_index] if self.current_index < self.total else self.placements[-1]
        self.status_label.text = (
            f"packed: {self.current_index}/{self.total} | "
            f"next: {next_item.instance_id} {next_item.box_type_id} | "
            f"unpacked: {self.unpacked_count} | "
            f"state: {state} | "
            "space: pause/resume, right: next, r: restart"
        )
        self.set_caption(f"Packed Task | shown {self.current_index}/{self.total}")

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.paused = not self.paused
            self._update_status()
        elif symbol == key.RIGHT:
            self._add_next_box()
        elif symbol == key.R:
            self._restart()

    def _restart(self) -> None:
        for sobj in tuple(self.scene.sobjs)[2:]:
            self.scene.remove(sobj)
        self.current_index = 0
        self.paused = True
        self._update_status()


def main() -> None:
    args = build_parser().parse_args()
    pallet = PalletSpec(
        length=args.pallet_length,
        width=args.pallet_width,
        max_height=args.pallet_max_height,
    )
    result = load_and_plan_multitype_task(args.task_file, pallet)
    stack_span = pallet.max_height + args.pallet_thickness
    cam_pos = (pallet.length * 1.7, -pallet.width * 1.5, stack_span * 1.3)
    cam_lookat = (pallet.length / 2, pallet.width / 2, stack_span / 2)
    viewer = AnimatedPackedTaskViewer(
        placements=result.placements,
        pallet=pallet,
        pallet_thickness=args.pallet_thickness,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
        unpacked_count=result.requested_count - result.packed_count,
    )
    viewer.run()


if __name__ == "__main__":
    main()
