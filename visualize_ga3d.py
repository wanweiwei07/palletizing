"""Run GA3D and visualize the result with the one viewer."""
from __future__ import annotations

import time

import numpy as np
import pyglet
import pyglet.window.key as key

from fill3d.cpsat_model import Fill3DInstance, Fill3DItem, Fill3DPlacement
from ga3d.solver import solve_ga3d, GA3DConfig
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


def _color_for_item(item_id: str, type_map: dict[str, int]) -> np.ndarray:
    # Group by item_id prefix (letter part)
    prefix = item_id.rstrip("0123456789")
    if prefix not in type_map:
        type_map[prefix] = len(type_map)
    return ITEM_COLORS[type_map[prefix] % len(ITEM_COLORS)]


def _compute_support_ratios(
    placements: list[Fill3DPlacement],
) -> dict[str, float]:
    """For each box, return (covered base area) / (base area). 1.0 = floor.

    Uses 5mm-cell boolean mask + numpy slice OR — handles overlapping
    supports correctly without Python per-cell loops.
    """
    ratios: dict[str, float] = {}
    CELL = 5  # mm per mask cell
    for b in placements:
        if b.z == 0:
            ratios[b.item_id] = 1.0
            continue
        nx = b.size_x // CELL
        ny = b.size_y // CELL
        if nx == 0 or ny == 0:
            ratios[b.item_id] = 1.0
            continue
        mask = np.zeros((nx, ny), dtype=bool)
        for s in placements:
            if s is b or s.z + s.size_z != b.z:
                continue
            x0 = max(b.x, s.x)
            x1 = min(b.x + b.size_x, s.x + s.size_x)
            y0 = max(b.y, s.y)
            y1 = min(b.y + b.size_y, s.y + s.size_y)
            if x1 <= x0 or y1 <= y0:
                continue
            ix0 = (x0 - b.x) // CELL
            ix1 = (x1 - b.x) // CELL
            iy0 = (y0 - b.y) // CELL
            iy1 = (y1 - b.y) // CELL
            mask[ix0:ix1, iy0:iy1] = True
        ratios[b.item_id] = float(mask.sum()) / (nx * ny)
    return ratios


class GA3DViewer(world_view.World):
    def __init__(
        self,
        placements: list[Fill3DPlacement],
        pallet_length: float,
        pallet_width: float,
        pallet_max_height: float,
        result_info: str,
        step_interval: float = 0.15,
    ):
        pl_m = pallet_length / 1000
        pw_m = pallet_width / 1000
        ph_m = pallet_max_height / 1000
        cam_pos = (pl_m * 1.8, -pw_m * 1.6, ph_m * 1.2)
        cam_lookat = (pl_m / 2, pw_m / 2, ph_m / 3)
        super().__init__(cam_pos=cam_pos, cam_lookat_pos=cam_lookat)

        self.placements = placements
        self.pl_m = pl_m
        self.pw_m = pw_m
        self.result_info = result_info
        self.current_index = 0
        self.paused = False
        self.total = len(placements)
        self.type_map: dict[str, int] = {}
        self.support_ratios = _compute_support_ratios(placements)
        self.last_ratio: float | None = None
        bad = [(p.item_id, self.support_ratios[p.item_id])
               for p in placements
               if self.support_ratios[p.item_id] < 0.9]
        bad.sort(key=lambda x: x[1])
        if bad:
            print(f"Support < 0.9: {len(bad)} boxes")
            for bid, r in bad[:15]:
                print(f"  {bid}: {r:.0%}")
        else:
            print("All boxes >= 0.9 support")

        # Pallet base
        thickness = 0.02
        prim.box(
            pos=np.array([pl_m / 2, pw_m / 2, thickness / 2], dtype=np.float32),
            half_extents=np.array([pl_m / 2, pw_m / 2, thickness / 2], dtype=np.float32),
            rgb=const.Tab20.BROWN_DEEP,
            alpha=1.0,
        ).attach_to(self.scene)
        prim.plane(
            pos=(pl_m / 2, pw_m / 2, 0.0),
            size=(pl_m * 2.5, pw_m * 2.5),
            rgb=const.ExtendedColor.MOON_GRAY,
            alpha=1.0,
        ).attach_to(self.scene)
        self.base_z = thickness

        self.status_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=14,
            x=12, y=12, anchor_x="left", anchor_y="bottom",
            color=(20, 20, 20, 255),
        )
        self._update_status()
        self.schedule_interval(self._advance, interval=max(step_interval, 0.01))

    def _advance(self, dt: float) -> None:
        if self.paused or self.current_index >= self.total:
            return
        self._add_next()

    def _add_next(self) -> None:
        if self.current_index >= self.total:
            return
        p = self.placements[self.current_index]
        # mm -> meters
        sx = p.size_x / 1000
        sy = p.size_y / 1000
        sz = p.size_z / 1000
        cx = p.x / 1000 + sx / 2
        cy = p.y / 1000 + sy / 2
        cz = self.base_z + p.z / 1000 + sz / 2

        color = _color_for_item(p.item_id, self.type_map)
        ratio = self.support_ratios.get(p.item_id, 1.0)
        # Floating boxes (ratio < 0.9): red tint override
        if ratio < 0.9:
            color = const.Tab20.RED_DEEP
        prim.box(
            pos=np.array([cx, cy, cz], dtype=np.float32),
            half_extents=np.array([sx / 2, sy / 2, sz / 2], dtype=np.float32),
            rgb=color,
            alpha=1.0,
        ).attach_to(self.scene)

        self.last_ratio = ratio
        self.current_index += 1
        self._update_status()

    def _update_status(self) -> None:
        state = "paused" if self.paused else ("finished" if self.current_index >= self.total else "playing")
        ratio_str = (
            f"sup={self.last_ratio:.0%} | " if self.last_ratio is not None
            else ""
        )
        self.status_label.text = (
            f"boxes: {self.current_index}/{self.total} | "
            f"{ratio_str}"
            f"{self.result_info} | {state} | "
            "space:pause  right:next  r:restart"
        )
        self.set_caption(f"GA3D Result | {self.current_index}/{self.total}")

    def on_draw(self):
        super().on_draw()
        self.status_label.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.RIGHT:
            self._add_next()
        elif symbol == key.R:
            for sobj in tuple(self.scene.sobjs)[2:]:
                self.scene.remove(sobj)
            self.current_index = 0
            self.paused = True
            self.type_map.clear()
        self._update_status()


def main():
    import json
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "42"
    if arg == "lx":
        task_file = "task_lx.json"
        with open(task_file) as f:
            data = json.load(f)
        payload = data["payload"]
        raw = payload["materialItems"]
        pallet = payload["scene"]["pallet"]
        PL = round(pallet["dimensions"]["length"])
        PW = round(pallet["dimensions"]["width"])
        PH = round(pallet["maxLoadHeight"])
        items = tuple(
            Fill3DItem(
                item_id=it["itemId"],
                length=round(it["dimensions"]["length"]),
                width=round(it["dimensions"]["width"]),
                height=round(it["dimensions"]["height"]),
                allow_rotation=it["orientationRules"]["canRotateZ"],
            )
            for it in raw
        )
    else:
        seed = int(arg)
        task_file = f"task_70_seed{seed}_dims.json"
        with open(task_file) as f:
            boxes = json.load(f)
        items = tuple(
            Fill3DItem(
                item_id=f"box_{i:03d}",
                length=round(b["length"] * 1000),
                width=round(b["width"] * 1000),
                height=round(b["height"] * 1000),
            )
            for i, b in enumerate(boxes)
        )
        PL, PW, PH = 1200, 800, 1500
    print(f"Task: {task_file} ({len(items)} boxes) pallet={PL}x{PW}x{PH}")
    inst = Fill3DInstance(
        pallet_length=PL, pallet_width=PW,
        pallet_max_height=PH, items=items,
    )

    cfg = GA3DConfig(
        pop_size=2048, time_limit_seconds=60.0,
        min_support_ratio=0.9, seed=42,
    )
    print(f"Running GA3D: {len(items)} items, "
          f"time_limit={cfg.time_limit_seconds}s ...")
    t0 = time.perf_counter()
    result = solve_ga3d(inst, cfg)
    dt = time.perf_counter() - t0
    info = f"packed={result.packed_count}/{result.total_items} vol={result.volume_ratio:.1%} gens={result.generations} time={dt:.1f}s"
    print(info)

    viewer = GA3DViewer(
        placements=result.placements,
        pallet_length=PL,
        pallet_width=PW,
        pallet_max_height=PH,
        result_info=info,
    )
    viewer.run()


if __name__ == "__main__":
    main()
