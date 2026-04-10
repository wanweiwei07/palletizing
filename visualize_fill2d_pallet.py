from __future__ import annotations

import argparse

import numpy as np
import pyglet
import pyglet.window.key as key

from palletizing import (
    MultiTypePlacement,
    PalletSpec,
    generate_multitype_task,
    load_task_boxes,
    plan_multitype_palletizing,
)
import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


LAYER_COLORS = [
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
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize multitype palletizing with selectable strategy.",
    )
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=70)
    parser.add_argument("--strategy", type=str, default="fill2d",
                        choices=[
            "auto", "layer-sequence", "layer-first",
            "beam", "fill2d", "3d", "ga3d",
        ])
    parser.add_argument("--pallet-length", type=float, default=1.2)
    parser.add_argument("--pallet-width", type=float, default=1.0)
    parser.add_argument("--pallet-max-height", type=float, default=1.5)
    parser.add_argument("--pallet-thickness", type=float, default=0.144)
    parser.add_argument("--step-interval", type=float, default=0.15)
    parser.add_argument("--start-paused", action="store_true")
    return parser


def _color_for_layer(layer: int) -> np.ndarray:
    return LAYER_COLORS[layer % len(LAYER_COLORS)]


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
        rgb=_color_for_layer(placement.layer),
        alpha=1.0,
    )


def _add_pallet(scene, pallet: PalletSpec, thickness: float) -> None:
    pallet_size = np.array(
        [pallet.length, pallet.width, thickness], dtype=np.float32,
    )
    prim.box(
        pos=pallet_size / 2,
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


class StrategyViewer(world_view.World):
    def __init__(
        self,
        placements: list[MultiTypePlacement],
        pallet: PalletSpec,
        pallet_thickness: float,
        step_interval: float,
        start_paused: bool,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
        packed_count: int,
        requested_count: int,
        layer_count: int,
        utilization_2d: float,
        utilization_3d: float,
        strategy: str,
        replan_fn=None,
    ):
        super().__init__(
            cam_pos=cam_pos,
            cam_lookat_pos=cam_lookat_pos,
            toggle_auto_cam_orbit=False,
        )
        self.placements = placements
        self.pallet_thickness = pallet_thickness
        self.current_index = 0
        self.paused = start_paused
        self.total = len(placements)
        self.packed_count = packed_count
        self.requested_count = requested_count
        self.layer_count = layer_count
        self.utilization_2d = utilization_2d
        self.utilization_3d = utilization_3d
        self.strategy = strategy

        self.status_label = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=13,
            x=12,
            y=12,
            anchor_x="left",
            anchor_y="bottom",
            color=(20, 20, 20, 255),
        )
        self.replan_fn = replan_fn
        self.pallet = pallet
        _add_pallet(self.scene, pallet, pallet_thickness)
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
        _make_box_object(p, self.pallet_thickness).attach_to(self.scene)
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else "playing"
        if self.current_index >= self.total:
            state = "finished"
        self.status_label.text = (
            f"[{self.strategy}] "
            f"placed: {self.current_index}/{self.total} | "
            f"layers: {self.layer_count} | "
            f"u2d: {self.utilization_2d:.3f}  u3d: {self.utilization_3d:.3f} | "
            f"{state} | "
            "space  right  r"
        )
        self.set_caption(
            f"{self.strategy} | "
            f"{self.packed_count}/{self.requested_count} packed | "
            f"{self.current_index}/{self.total} shown"
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
        if self.replan_fn is not None:
            result = self.replan_fn()
            self.placements = result.placements
            self.total = len(result.placements)
            self.packed_count = result.packed_count
            self.requested_count = result.requested_count
            self.layer_count = result.layer_count
            self.utilization_2d = result.utilization_2d
            self.utilization_3d = result.utilization_3d
        self.current_index = 0
        self.paused = False
        self._update_status()


def main() -> None:
    import random as _random

    args = build_parser().parse_args()
    def _do_plan(seed=None):
        if args.task_file:
            boxes = load_task_boxes(args.task_file)
        else:
            if seed is None:
                seed = _random.randint(0, 99999)
            task = generate_multitype_task(count=args.count, seed=seed)
            boxes = task.boxes
        print(f"Planning with strategy={args.strategy} seed={seed} "
              f"({len(boxes)} boxes) ...")
        result = plan_multitype_palletizing(
            boxes, pallet, strategy=args.strategy,
        )
        print(
            f"Done: packed={result.packed_count}/{result.requested_count}  "
            f"layers={result.layer_count}  "
            f"u2d={result.utilization_2d:.3f}  "
            f"u3d={result.utilization_3d:.3f}"
        )
        return result

    pallet = PalletSpec(
        length=args.pallet_length,
        width=args.pallet_width,
        max_height=args.pallet_max_height,
    )

    result = _do_plan(seed=args.seed)

    stack_span = pallet.max_height + args.pallet_thickness
    cam_pos = (pallet.length * 1.7, -pallet.width * 1.5, stack_span * 1.3)
    cam_lookat = (pallet.length / 2, pallet.width / 2, stack_span / 2)

    viewer = StrategyViewer(
        placements=result.placements,
        pallet=pallet,
        pallet_thickness=args.pallet_thickness,
        step_interval=args.step_interval,
        start_paused=args.start_paused,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
        packed_count=result.packed_count,
        requested_count=result.requested_count,
        layer_count=result.layer_count,
        utilization_2d=result.utilization_2d,
        utilization_3d=result.utilization_3d,
        strategy=args.strategy,
        replan_fn=_do_plan,
    )
    viewer.run()


if __name__ == "__main__":
    main()
