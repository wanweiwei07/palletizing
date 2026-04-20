"""Gymnasium environment for 3D palletizing.

One step = pick (box, rotation); decoder places it at greedy best-fit
position using the same scoring as GA3D. Action masking via sb3-contrib.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import Env, spaces

from fill3d.cpsat_model import Fill3DInstance, Fill3DItem, Fill3DPlacement


def load_task_lx(path: str = "task_lx.json") -> Fill3DInstance:
    with open(path) as f:
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
    return Fill3DInstance(
        pallet_length=PL, pallet_width=PW, pallet_max_height=PH,
        items=items,
    )


class PalletEnv(Env):
    """Pick (box, rot); decoder greedy-places at best-fit position."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: Fill3DInstance,
        grid_res_xy: int = 10,
        grid_res_z: int = 10,
        min_support_ratio: float = 0.9,
        max_infeasible: int = 2,
    ):
        super().__init__()
        self.inst = instance
        self.gr_xy = grid_res_xy
        self.gr_z = grid_res_z
        self.min_sup = min_support_ratio
        self.max_infeasible = max_infeasible

        items = instance.items
        self.n = len(items)
        self.PL = instance.pallet_length
        self.PW = instance.pallet_width
        self.PH = instance.pallet_max_height
        self.GX = self.PL // self.gr_xy
        self.GY = self.PW // self.gr_xy
        self.PH_v = self.PH // self.gr_z

        lengths = np.array([it.length for it in items], dtype=np.int32)
        widths = np.array([it.width for it in items], dtype=np.int32)
        heights = np.array([it.height for it in items], dtype=np.int32)
        self.lengths = lengths
        self.widths = widths
        self.heights = heights
        self.can_rotate = np.array(
            [it.allow_rotation and it.length != it.width for it in items],
            dtype=bool,
        )
        self.fw_norot = (lengths + self.gr_xy - 1) // self.gr_xy
        self.fh_norot = (widths + self.gr_xy - 1) // self.gr_xy
        self.fw_rot = self.fh_norot.copy()
        self.fh_rot = self.fw_norot.copy()
        self.fz = (heights + self.gr_z - 1) // self.gr_z
        self.item_vols = (
            lengths.astype(np.int64) * widths * heights
        ).astype(np.float64)

        # Normalized box dims for observation.
        self.box_dims_norm = np.stack([
            lengths / self.PL, widths / self.PW, heights / self.PH,
        ], axis=1).astype(np.float32)

        self.pallet_area_mm2 = self.PL * self.PW

        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(
                0.0, 1.0, shape=(1, self.GX, self.GY), dtype=np.float32,
            ),
            "remaining": spaces.Box(
                0.0, 1.0, shape=(self.n,), dtype=np.float32,
            ),
            "box_dims": spaces.Box(
                0.0, 2.0, shape=(self.n, 3), dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(self.n * 2)

        # Scoring constants (match GA3D decoder).
        self.PENALTY_W = 2.0
        self.CONTACT_W = 500.0
        self.DIST_W = 1.0 * (self.gr_xy / 50.0)
        gx_ax = torch.arange(self.GX, dtype=torch.float32).unsqueeze(1)
        gy_ax = torch.arange(self.GY, dtype=torch.float32).unsqueeze(0)
        self.dist_map = (gx_ax + gy_ax) * self.DIST_W  # (GX, GY)

        self._reset_state()

    def _reset_state(self):
        self.hmap = torch.zeros(
            (self.GX, self.GY), dtype=torch.int32,
        )
        self.placed = np.zeros(self.n, dtype=bool)
        self.placements: list[Fill3DPlacement] = []
        self.infeasible_streak = 0
        self._cached_mask: np.ndarray | None = None
        self._cached_best: dict[int, tuple[int, int, int]] = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def _obs(self):
        hm = (self.hmap.numpy().astype(np.float32) /
              max(self.PH_v, 1))[None, ...]
        return {
            "heightmap": hm,
            "remaining": (~self.placed).astype(np.float32),
            "box_dims": self.box_dims_norm.copy(),
        }

    def _score_one_fp(self, fw: int, fh: int):
        """Compute score grid for one footprint; return (score, mp) tensors."""
        hmap = self.hmap.float().unsqueeze(0).unsqueeze(0)
        mp = F.max_pool2d(hmap, (fw, fh), stride=1).squeeze()
        ap = F.avg_pool2d(hmap, (fw, fh), stride=1).squeeze()

        occ = (hmap > 0).float()
        occ_p = F.pad(occ, (1, 1, 1, 1), value=1.0)
        big_area = (fw + 2) * (fh + 2)
        inner = fw * fh
        border = big_area - inner
        big_sum = F.avg_pool2d(
            occ_p, (fw + 2, fh + 2), stride=1,
        ).squeeze() * big_area
        inner_sum = F.avg_pool2d(
            occ, (fw, fh), stride=1,
        ).squeeze() * inner
        contact = (big_sum - inner_sum) / border

        penalty = mp - ap
        NX, NY = mp.shape
        score = (
            (mp + penalty * self.PENALTY_W) * self.gr_z
            - contact * self.CONTACT_W
            + self.dist_map[:NX, :NY]
        )
        ratio = torch.where(mp > 0, ap / mp, torch.ones_like(mp))
        score = torch.where(
            ratio < self.min_sup,
            torch.full_like(score, float("inf")), score,
        )
        return score, mp

    def _find_position(self, fw: int, fh: int, fz: int):
        """Return (gx, gy, z_bot, valid). Uses cache per-step."""
        if fw <= 0 or fh <= 0 or fw > self.GX or fh > self.GY:
            return 0, 0, 0, False
        score, mp = self._score_one_fp(fw, fh)
        # z cap: z_bot + fz must fit.
        z_top = mp + fz
        score = torch.where(
            z_top > self.PH_v,
            torch.full_like(score, float("inf")), score,
        )
        flat = score.flatten()
        min_v, min_i = flat.min(dim=0)
        if not torch.isfinite(min_v):
            return 0, 0, 0, False
        NY = mp.shape[1]
        gx = int(min_i.item() // NY)
        gy = int(min_i.item() % NY)
        z = int(mp[gx, gy].item())
        return gx, gy, z, True

    def action_masks(self):
        """(n*2,) bool: which (box, rot) actions are feasible."""
        if self._cached_mask is not None:
            return self._cached_mask
        mask = np.zeros(self.n * 2, dtype=bool)
        best = {}
        # Group candidate (fw, fh) to avoid redundant pool passes.
        groups: dict[tuple[int, int], list[tuple[int, int, int]]] = (
            defaultdict(list)
        )
        for i in range(self.n):
            if self.placed[i]:
                continue
            for rot in range(2):
                if rot == 1 and not self.can_rotate[i]:
                    continue
                fw = int(self.fw_rot[i] if rot else self.fw_norot[i])
                fh = int(self.fh_rot[i] if rot else self.fh_norot[i])
                fz = int(self.fz[i])
                groups[(fw, fh)].append((i, rot, fz))

        for (fw, fh), cands in groups.items():
            if fw <= 0 or fh <= 0 or fw > self.GX or fh > self.GY:
                continue
            score, mp = self._score_one_fp(fw, fh)
            NY = mp.shape[1]
            for i, rot, fz in cands:
                z_top = mp + fz
                sc = torch.where(
                    z_top > self.PH_v,
                    torch.full_like(score, float("inf")), score,
                )
                flat = sc.flatten()
                min_v, min_i = flat.min(dim=0)
                if torch.isfinite(min_v):
                    mask[i * 2 + rot] = True
                    gx = int(min_i.item() // NY)
                    gy = int(min_i.item() % NY)
                    z = int(mp[gx, gy].item())
                    best[i * 2 + rot] = (gx, gy, z)
        self._cached_mask = mask
        self._cached_best = best
        return mask

    def step(self, action: int):
        # Ensure mask is computed (so cache is populated for placement).
        self.action_masks()
        box_i = int(action // 2)
        rot = int(action % 2)

        best = self._cached_best.get(int(action))
        if best is None or self.placed[box_i]:
            # Infeasible — mask should prevent this, but guard anyway.
            self.infeasible_streak += 1
            terminated = (
                self.infeasible_streak >= self.max_infeasible
                or bool(self.placed.all())
            )
            return self._obs(), 0.0, terminated, False, {"invalid": True}

        gx, gy, z = best
        fw = int(self.fw_rot[box_i] if rot else self.fw_norot[box_i])
        fh = int(self.fh_rot[box_i] if rot else self.fh_norot[box_i])
        fz = int(self.fz[box_i])

        self.hmap[gx:gx + fw, gy:gy + fh] = z + fz
        self.placed[box_i] = True
        self.infeasible_streak = 0
        self._cached_mask = None
        self._cached_best = {}

        item = self.inst.items[box_i]
        sx = item.width if rot else item.length
        sy = item.length if rot else item.width
        sz = item.height
        self.placements.append(Fill3DPlacement(
            item_id=item.item_id,
            x=gx * self.gr_xy, y=gy * self.gr_xy, z=z * self.gr_z,
            rotated=bool(rot), size_x=sx, size_y=sy, size_z=sz,
        ))

        # Dense reward: fraction of placed volume vs pallet base * PH.
        reward = float(sx * sy * sz) / (self.pallet_area_mm2 * self.PH)

        terminated = bool(self.placed.all())
        if not terminated:
            # If no feasible action remains, end the episode.
            if not self.action_masks().any():
                terminated = True

        info: dict[str, object] = {}
        if terminated:
            packed_vol = sum(
                p.size_x * p.size_y * p.size_z for p in self.placements
            )
            max_top = max(
                (p.z + p.size_z for p in self.placements), default=0,
            )
            used_vol = (
                self.pallet_area_mm2 * max_top if max_top > 0 else 1
            )
            info["vol_ratio"] = packed_vol / used_vol
            info["packed"] = int(self.placed.sum())

        return self._obs(), reward, terminated, False, info
