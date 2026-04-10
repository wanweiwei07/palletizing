from __future__ import annotations

import argparse

import numpy as np
import pyglet
import pyglet.window.key as key

from fill2d import Fill2DInstance, Fill2DItem, Fill2DPlacement, Fill2DResult, solve_fill2d
import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.viewer.world as world_view


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

ITEM_HEIGHT = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a Fill2D solution.")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unit-scale", type=float, default=0.1,
                        help="Scale integer grid units to viewer units.")
    parser.add_argument("--step-interval", type=float, default=0.3)
    parser.add_argument("--start-paused", action="store_true")
    return parser


def _color_for_index(index: int) -> np.ndarray:
    return ITEM_COLORS[index % len(ITEM_COLORS)]


def _make_item_object(
    placement: Fill2DPlacement,
    color_index: int,
    unit_scale: float,
    base_z: float,
):
    center = np.array(
        [
            (placement.x + placement.width / 2) * unit_scale,
            (placement.y + placement.height / 2) * unit_scale,
            base_z + ITEM_HEIGHT / 2,
        ],
        dtype=np.float32,
    )
    half_extents = np.array(
        [
            placement.width * unit_scale / 2,
            placement.height * unit_scale / 2,
            ITEM_HEIGHT / 2,
        ],
        dtype=np.float32,
    )
    return prim.box(
        pos=center,
        half_extents=half_extents,
        rgb=_color_for_index(color_index),
        alpha=1.0,
    )


def _add_pallet_base(scene, pallet_w: int, pallet_h: int, unit_scale: float) -> None:
    thickness = 0.02
    pallet_size = np.array(
        [pallet_w * unit_scale, pallet_h * unit_scale, thickness],
        dtype=np.float32,
    )
    pallet_center = np.array(
        [pallet_size[0] / 2, pallet_size[1] / 2, thickness / 2],
        dtype=np.float32,
    )
    prim.box(
        pos=pallet_center,
        half_extents=pallet_size / 2,
        rgb=const.Tab20.BROWN_DEEP,
        alpha=1.0,
    ).attach_to(scene)

    prim.plane(
        pos=(pallet_size[0] / 2, pallet_size[1] / 2, 0.0),
        size=(pallet_w * unit_scale * 2.0, pallet_h * unit_scale * 2.0),
        rgb=const.ExtendedColor.MOON_GRAY,
        alpha=1.0,
    ).attach_to(scene)


class Fill2DViewer(world_view.World):
    def __init__(
        self,
        result: Fill2DResult,
        pallet_w: int,
        pallet_h: int,
        unit_scale: float,
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
        self.result = result
        self.placements = result.placements
        self.pallet_w = pallet_w
        self.pallet_h = pallet_h
        self.unit_scale = unit_scale
        self.current_index = 0
        self.paused = start_paused
        self.total = len(self.placements)
        self.base_z = 0.02

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

        _add_pallet_base(self.scene, pallet_w, pallet_h, unit_scale)
        self._update_status()
        self.schedule_interval(
            self._advance_animation, interval=max(step_interval, 0.01),
        )

    def _advance_animation(self, dt: float) -> None:
        if self.paused or self.current_index >= self.total:
            self._update_status()
            return
        self._add_next_item()

    def _add_next_item(self) -> None:
        if self.current_index >= self.total:
            self._update_status()
            return
        placement = self.placements[self.current_index]
        obj = _make_item_object(
            placement, self.current_index, self.unit_scale, self.base_z,
        )
        obj.attach_to(self.scene)
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        self.status_label.text = (
            f"items: {self.current_index}/{self.total} | "
            f"status: {self.result.status} | "
            f"coverage: {self.result.coverage_ratio:.1%} | "
            f"state: {state} | "
            "space: pause/resume, right: next, r: restart"
        )
        self.set_caption(
            f"Fill2D | placed {self.current_index}/{self.total} "
            f"| coverage {self.result.coverage_ratio:.1%}"
        )

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.paused = not self.paused
            self._update_status()
        elif symbol == key.RIGHT:
            self._add_next_item()
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

    instance = Fill2DInstance(
        pallet_width=12,
        pallet_height=10,
        items=(
            Fill2DItem("A", 6, 5, True),
            Fill2DItem("B", 6, 5, True),
            Fill2DItem("C", 6, 5, True),
            Fill2DItem("D", 6, 5, True),
            Fill2DItem("E", 4, 5, True),
            Fill2DItem("F", 4, 5, True),
        ),
    )
    result = solve_fill2d(
        instance,
        time_limit_seconds=args.time_limit,
        num_workers=args.workers,
    )

    pw = instance.pallet_width
    ph = instance.pallet_height
    s = args.unit_scale
    span_x = pw * s
    span_y = ph * s
    cam_pos = (span_x / 2, -span_y * 0.8, max(span_x, span_y) * 1.5)
    cam_lookat = (span_x / 2, span_y / 2, 0.0)

    viewer = Fill2DViewer(
        result=result,
        pallet_w=pw,
        pallet_h=ph,
        unit_scale=s,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
    )
    viewer.run()


if __name__ == "__main__":
    main()
