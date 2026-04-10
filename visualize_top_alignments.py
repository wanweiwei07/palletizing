from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi

import numpy as np
import pyglet
import pyglet.window.key as key

import one.scene.scene_object_primitive as prim
import one.utils.constant as const
import one.utils.math as math_utils
import one.viewer.world as world_view


@dataclass(frozen=True)
class TopAlignmentCase:
    index: int
    yaw: float
    old_corner_name: str
    new_corner_name: str
    old_corner: tuple[float, float]
    new_corner: tuple[float, float]
    new_center: tuple[float, float, float]
    local_size_x: float
    local_size_y: float
    size_x: float
    size_y: float
    size_z: float


OLD_TOP_CORNER_NAMES = ("a'", "b'", "c'", "d'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the 8 top-to-bottom corner alignment cases.")
    parser.add_argument("--old-length", type=float, default=0.45)
    parser.add_argument("--old-width", type=float, default=0.35)
    parser.add_argument("--old-height", type=float, default=0.25)
    parser.add_argument("--new-length", type=float, default=0.30)
    parser.add_argument("--new-width", type=float, default=0.22)
    parser.add_argument("--new-height", type=float, default=0.18)
    parser.add_argument("--pallet-thickness", type=float, default=0.144)
    return parser


def _rect_corners(size_x: float, size_y: float) -> list[tuple[float, float]]:
    return [
        (0.0, 0.0),
        (size_x, 0.0),
        (size_x, size_y),
        (0.0, size_y),
    ]


def _world_corners(center_x: float, center_y: float, size_x: float, size_y: float) -> list[tuple[float, float]]:
    x_min = center_x - size_x / 2
    y_min = center_y - size_y / 2
    return [
        (x_min, y_min),
        (x_min + size_x, y_min),
        (x_min + size_x, y_min + size_y),
        (x_min, y_min + size_y),
    ]


def _new_corner_names_for_yaw(yaw: float) -> tuple[str, str, str, str]:
    if abs(yaw) < 1e-9:
        return ("e", "f", "g", "h")
    return ("f", "g", "h", "e")


def generate_top_alignment_cases(
    old_length: float,
    old_width: float,
    old_height: float,
    new_length: float,
    new_width: float,
    new_height: float,
) -> tuple[list[TopAlignmentCase], tuple[float, float, float], tuple[float, float, float]]:
    old_center = (0.0, 0.0, old_height / 2)
    old_size = (old_length, old_width, old_height)
    old_world_corners = _world_corners(old_center[0], old_center[1], old_length, old_width)

    cases: list[TopAlignmentCase] = []
    case_index = 1
    for yaw, size_x, size_y in (
        (0.0, new_length, new_width),
        (pi / 2, new_width, new_length),
    ):
        local_corners = _rect_corners(size_x, size_y)
        new_corner_names = _new_corner_names_for_yaw(yaw)
        for old_index, old_corner in enumerate(old_world_corners):
            new_corner = local_corners[old_index]
            cases.append(
                TopAlignmentCase(
                    index=case_index,
                    yaw=yaw,
                    old_corner_name=OLD_TOP_CORNER_NAMES[old_index],
                    new_corner_name=new_corner_names[old_index],
                    old_corner=old_corner,
                    new_corner=new_corner,
                    new_center=(old_corner[0] - new_corner[0] + size_x / 2, old_corner[1] - new_corner[1] + size_y / 2, old_height + new_height / 2),
                    local_size_x=new_length,
                    local_size_y=new_width,
                    size_x=size_x,
                    size_y=size_y,
                    size_z=new_height,
                )
            )
            case_index += 1
    return cases, old_center, old_size


class TopAlignmentViewer(world_view.World):
    def __init__(
        self,
        cases: list[TopAlignmentCase],
        old_center: tuple[float, float, float],
        old_size: tuple[float, float, float],
        pallet_thickness: float,
        cam_pos: tuple[float, float, float],
        cam_lookat_pos: tuple[float, float, float],
    ):
        super().__init__(cam_pos=cam_pos, cam_lookat_pos=cam_lookat_pos, toggle_auto_cam_orbit=False)
        self.cases = cases
        self.old_center = old_center
        self.old_size = old_size
        self.pallet_thickness = pallet_thickness
        self.current = 0
        self.new_box = None
        self.marker = None
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
        self._build_static_scene()
        self._show_case(0)

    def _build_static_scene(self) -> None:
        old_pos = np.array(
            [self.old_center[0], self.old_center[1], self.old_center[2] + self.pallet_thickness],
            dtype=np.float32,
        )
        prim.box(
            pos=old_pos,
            half_extents=np.array(self.old_size, dtype=np.float32) / 2,
            rgb=const.Tab20.BLUE_LIGHT,
            alpha=1.0,
        ).attach_to(self.scene)
        prim.plane(
            pos=(0.0, 0.0, 0.0),
            size=(2.0, 2.0),
            rgb=const.ExtendedColor.MOON_GRAY,
            alpha=1.0,
        ).attach_to(self.scene)

    def _show_case(self, case_index: int) -> None:
        if self.new_box is not None:
            self.scene.remove(self.new_box)
            self.new_box = None
        if self.marker is not None:
            self.scene.remove(self.marker)
            self.marker = None

        self.current = case_index % len(self.cases)
        case = self.cases[self.current]
        center = np.array(
            [case.new_center[0], case.new_center[1], case.new_center[2] + self.pallet_thickness],
            dtype=np.float32,
        )
        rotmat = math_utils.rotmat_from_axangle(const.StandardAxis.Z, case.yaw)
        self.new_box = prim.box(
            pos=center,
            half_extents=np.array([case.local_size_x / 2, case.local_size_y / 2, case.size_z / 2], dtype=np.float32),
            rotmat=rotmat,
            rgb=const.Tab20.ORANGE_LIGHT,
            alpha=1.0,
        )
        self.new_box.attach_to(self.scene)

        marker_pos = np.array([case.old_corner[0], case.old_corner[1], self.old_size[2] + self.pallet_thickness + 0.01], dtype=np.float32)
        self.marker = prim.sphere(
            pos=marker_pos,
            radius=0.015,
            rgb=const.BasicColor.RED,
            alpha=1.0,
        )
        self.marker.attach_to(self.scene)
        self._update_status()

    def _update_status(self) -> None:
        case = self.cases[self.current]
        yaw_text = "0" if abs(case.yaw) < 1e-9 else "pi/2"
        self.status_label.text = (
            f"case {case.index}/8 | yaw={yaw_text} | "
            f"{case.old_corner_name} > {case.new_corner_name} | "
            "space/right: next, left: prev"
        )
        self.set_caption(f"Top Alignment Demo | case {case.index}/8")

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)
        if symbol in (key.SPACE, key.RIGHT):
            self._show_case(self.current + 1)
        elif symbol == key.LEFT:
            self._show_case(self.current - 1)


def main() -> None:
    args = build_parser().parse_args()
    cases, old_center, old_size = generate_top_alignment_cases(
        old_length=args.old_length,
        old_width=args.old_width,
        old_height=args.old_height,
        new_length=args.new_length,
        new_width=args.new_width,
        new_height=args.new_height,
    )
    cam_pos = (1.2, -1.1, 1.4)
    cam_lookat = (0.0, 0.0, args.pallet_thickness + args.old_height + args.new_height * 0.5)
    viewer = TopAlignmentViewer(
        cases=cases,
        old_center=old_center,
        old_size=old_size,
        pallet_thickness=args.pallet_thickness,
        cam_pos=cam_pos,
        cam_lookat_pos=cam_lookat,
    )
    viewer.run()


if __name__ == "__main__":
    main()
