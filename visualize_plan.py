from __future__ import annotations

import argparse

import numpy as np
import pyglet
import pyglet.window.key as key

from palletizing import BoxSpec, PalletSpec, Placement, plan_palletizing
import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a palletizing plan with animated placement order.")
    parser.add_argument("--pallet-length", type=float, default=1200.0)
    parser.add_argument("--pallet-width", type=float, default=1000.0)
    parser.add_argument("--pallet-max-height", type=float, default=1500.0)
    parser.add_argument("--box-length", type=float, default=300.0)
    parser.add_argument("--box-width", type=float, default=200.0)
    parser.add_argument("--box-height", type=float, default=250.0)
    parser.add_argument("--count", type=int, default=70)
    parser.add_argument("--unit-scale", type=float, default=0.001, help="Scale model units to viewer units.")
    parser.add_argument("--pallet-thickness", type=float, default=144.0, help="Visual pallet thickness, same unit as dimensions.")
    parser.add_argument("--step-interval", type=float, default=0.2, help="Seconds between two placements.")
    parser.add_argument("--start-paused", action="store_true", help="Start the animation paused.")
    return parser


def _color_for_layer(layer: int) -> np.ndarray:
    palette = [
        const.Tab20.BLUE_LIGHT,
        const.Tab20.ORANGE_LIGHT,
        const.Tab20.GREEN_LIGHT,
        const.Tab20.RED_LIGHT,
        const.Tab20.BROWN_LIGHT,
        const.Tab20.CYAN_LIGHT,
    ]
    return palette[layer % len(palette)]


def _make_box_object(placement: Placement, unit_scale: float, pallet_thickness: float):
    z_offset = pallet_thickness * unit_scale
    local_size_x = placement.size_x
    local_size_y = placement.size_y
    if abs(placement.yaw - np.pi / 2) < 1e-9:
        local_size_x, local_size_y = placement.size_y, placement.size_x
    center = np.array(
        [
            placement.x * unit_scale,
            placement.y * unit_scale,
            placement.z * unit_scale + z_offset,
        ],
        dtype=np.float32,
    )
    half_extents = np.array(
        [
            local_size_x * unit_scale / 2,
            local_size_y * unit_scale / 2,
            placement.size_z * unit_scale / 2,
        ],
        dtype=np.float32,
    )
    rotmat = math_utils.rotmat_from_axangle(const.StandardAxis.Z, placement.yaw)
    return prim.box(
        pos=center,
        half_extents=half_extents,
        rotmat=rotmat,
        rgb=_color_for_layer(placement.layer),
        alpha=1.0,
    )


def add_pallet(scene, pallet: PalletSpec, unit_scale: float, pallet_thickness: float) -> None:
    pallet_size = np.array(
        [pallet.length * unit_scale, pallet.width * unit_scale, pallet_thickness * unit_scale],
        dtype=np.float32,
    )
    pallet_center = np.array([pallet_size[0] / 2, pallet_size[1] / 2, pallet_size[2] / 2], dtype=np.float32)
    pallet_box = prim.box(
        pos=pallet_center,
        half_extents=pallet_size / 2,
        rgb=const.Tab20.BROWN_DEEP,
        alpha=1.0,
    )
    pallet_box.attach_to(scene)

    ground = prim.plane(
        pos=(pallet_size[0] / 2, pallet_size[1] / 2, 0.0),
        size=(pallet.length * unit_scale * 2.0, pallet.width * unit_scale * 2.0),
        rgb=const.ExtendedColor.MOON_GRAY,
        alpha=1.0,
    )
    ground.attach_to(scene)


class AnimatedPlanViewer(world_view.World):
    def __init__(
        self,
        placements: list[Placement],
        pallet: PalletSpec,
        unit_scale: float,
        pallet_thickness: float,
        step_interval: float,
        start_paused: bool,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
    ):
        super().__init__(cam_pos=cam_pos, cam_lookat_pos=cam_lookat_pos, toggle_auto_cam_orbit=False)
        self.placements = placements
        self.unit_scale = unit_scale
        self.pallet_thickness = pallet_thickness
        self.step_interval = step_interval
        self.current_index = 0
        self.paused = start_paused
        self.total = len(placements)
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

        add_pallet(self.scene, pallet, unit_scale, pallet_thickness)
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
        box_obj = _make_box_object(placement, self.unit_scale, self.pallet_thickness)
        box_obj.attach_to(self.scene)
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        next_sequence = self.current_index + 1 if self.current_index < self.total else self.total
        self.status_label.text = (
            f"sequence: {self.current_index}/{self.total} | "
            f"next: {next_sequence} | state: {state} | "
            "space: pause/resume, right: next, r: restart"
        )
        self.set_caption(f"WRS World | placed {self.current_index}/{self.total}")

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
    box = BoxSpec(
        length=args.box_length,
        width=args.box_width,
        height=args.box_height,
        count=args.count,
    )
    plan = plan_palletizing(pallet, box)

    pallet_span = max(args.pallet_length, args.pallet_width) * args.unit_scale
    stack_span = args.pallet_max_height * args.unit_scale + args.pallet_thickness * args.unit_scale
    cam_pos = (pallet_span * 1.6, -pallet_span * 1.4, stack_span * 1.5)
    cam_lookat = (pallet.length * args.unit_scale / 2, pallet.width * args.unit_scale / 2, stack_span / 2)

    viewer = AnimatedPlanViewer(
        placements=plan.placements,
        pallet=pallet,
        unit_scale=args.unit_scale,
        pallet_thickness=args.pallet_thickness,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
    )
    viewer.run()


if __name__ == "__main__":
    main()
