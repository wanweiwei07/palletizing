"""GPU-native VecEnv for 3D palletizing, implementing rl_games IVecEnv.

All state, obs, masks stay on GPU. step() returns torch tensors;
rl_games' a2c_common detects tensor obs and skips numpy conversion.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from rl_games.common.ivecenv import IVecEnv

from fill3d.cpsat_model import Fill3DInstance

import os as _os
_USE_TRITON = _os.environ.get("RL3D_USE_TRITON", "0") == "1"
if _USE_TRITON:
    from rl3d.triton_decode import triton_scan_rl
else:
    triton_scan_rl = None  # type: ignore


class PalletVecEnv(IVecEnv):
    """rl_games IVecEnv: discrete (box, rot); greedy best-fit decoder."""

    def __init__(
        self,
        instance: Fill3DInstance,
        num_envs: int = 4096,
        grid_res_xy: int = 10,
        grid_res_z: int = 10,
        min_support_ratio: float = 0.9,
        obs_pool: int = 4,
        device: str | torch.device | None = None,
    ):
        self.N = int(num_envs)
        self.inst = instance
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        items = instance.items
        self.n = len(items)
        self.PL = instance.pallet_length
        self.PW = instance.pallet_width
        self.PH = instance.pallet_max_height
        self.gr_xy = grid_res_xy
        self.gr_z = grid_res_z
        self.GX = self.PL // self.gr_xy
        self.GY = self.PW // self.gr_xy
        self.PH_v = self.PH // self.gr_z
        self.min_sup = min_support_ratio
        self.obs_pool = int(obs_pool)
        self._use_triton = _USE_TRITON

        dev = self.device
        lengths = torch.tensor(
            [it.length for it in items], dtype=torch.int32, device=dev,
        )
        widths = torch.tensor(
            [it.width for it in items], dtype=torch.int32, device=dev,
        )
        heights = torch.tensor(
            [it.height for it in items], dtype=torch.int32, device=dev,
        )
        self.can_rotate = torch.tensor(
            [it.allow_rotation and it.length != it.width for it in items],
            dtype=torch.bool, device=dev,
        )
        fw_norot = ((lengths + self.gr_xy - 1) // self.gr_xy).to(torch.int32)
        fh_norot = ((widths + self.gr_xy - 1) // self.gr_xy).to(torch.int32)
        self.fz = ((heights + self.gr_z - 1) // self.gr_z).to(torch.int32)
        self.vol_mm3 = lengths.to(torch.int64) * widths * heights

        self.box_dims_flat = torch.stack([
            lengths.float() / self.PL,
            widths.float() / self.PW,
            heights.float() / self.PH,
        ], dim=1).flatten().to(dev)  # (n*3,)

        n = self.n
        self.A = n * 2
        box_of = torch.arange(n, device=dev).repeat_interleave(2)
        rot_of = torch.tensor(
            [0, 1] * n, dtype=torch.int32, device=dev,
        )
        fw_of = torch.where(
            rot_of == 1, fh_norot[box_of], fw_norot[box_of],
        )
        fh_of = torch.where(
            rot_of == 1, fw_norot[box_of], fh_norot[box_of],
        )
        valid_rot = (rot_of == 0) | self.can_rotate[box_of]
        fw_of = torch.where(valid_rot, fw_of, torch.zeros_like(fw_of))
        fh_of = torch.where(valid_rot, fh_of, torch.zeros_like(fh_of))
        self.box_of = box_of
        self.rot_of = rot_of
        self.fw_of = fw_of
        self.fh_of = fh_of
        self.fz_of = self.fz[box_of]
        self.vol_of = self.vol_mm3[box_of]
        self.valid_rot = valid_rot

        # Contiguous int32 copies for Triton kernel.
        self._fw_t = self.fw_of.to(torch.int32).contiguous()
        self._fh_t = self.fh_of.to(torch.int32).contiguous()
        self._fz_t = self.fz_of.to(torch.int32).contiguous()

        self.PENALTY_W = 2.0
        self.CONTACT_W = 500.0
        self.DIST_W = 1.0 * (self.gr_xy / 50.0)
        gx_ax = torch.arange(self.GX, device=dev, dtype=torch.float32)
        gy_ax = torch.arange(self.GY, device=dev, dtype=torch.float32)
        self.dist_map = (
            gx_ax.unsqueeze(1) + gy_ax.unsqueeze(0)
        ) * self.DIST_W

        self.pallet_area = self.PL * self.PW
        self.pallet_base_ph = float(self.pallet_area * self.PH)

        p = self.obs_pool
        self.OX = (self.GX + p - 1) // p if p > 1 else self.GX
        self.OY = (self.GY + p - 1) // p if p > 1 else self.GY
        self.obs_dim = self.OX * self.OY + n + n * 3
        self.observation_space = spaces.Box(
            low=-1e4, high=1e4, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.A)

        self._reset_state()
        self._mask: torch.Tensor | None = None
        self._best: torch.Tensor | None = None

    def get_env_info(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
            "value_size": 1,
        }

    def get_number_of_agents(self):
        return 1

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        if self._mask is None:
            self._mask, self._best = self._compute_mask_and_best()
        return self._mask

    def seed(self, seed):
        if seed is not None:
            torch.manual_seed(int(seed))

    def _reset_state(self):
        N, dev = self.N, self.device
        self.hmap = torch.zeros(
            (N, self.GX, self.GY), dtype=torch.int32, device=dev,
        )
        self.placed = torch.zeros(
            (N, self.n), dtype=torch.bool, device=dev,
        )
        self.packed_vol = torch.zeros(N, dtype=torch.int64, device=dev)
        self.packed_cnt = torch.zeros(N, dtype=torch.int32, device=dev)
        self.max_top_z = torch.zeros(N, dtype=torch.int32, device=dev)

    def _reset_some(self, done: torch.Tensor):
        if not done.any():
            return
        self.hmap[done] = 0
        self.placed[done] = False
        self.packed_vol[done] = 0
        self.packed_cnt[done] = 0
        self.max_top_z[done] = 0

    def _compute_mask_and_best(self):
        """Return (mask (N, A) bool, best (N, A, 3) int32) on GPU."""
        if self._use_triton:
            return self._compute_mask_and_best_triton()
        return self._compute_mask_and_best_torch()

    def _compute_mask_and_best_triton(self):
        hmap_c = self.hmap.contiguous()
        bx, by, bz, bs = triton_scan_rl(
            hmap_c, self._fw_t, self._fh_t, self._fz_t,
            ph_v=self.PH_v,
            penalty_w=self.PENALTY_W,
            contact_w=self.CONTACT_W,
            dist_w=self.DIST_W,
            gr_z=self.gr_z,
            min_support_ratio=self.min_sup,
        )
        best = torch.stack([bx, by, bz], dim=-1)
        feas = torch.isfinite(bs)
        placed_by_a = self.placed[:, self.box_of.long()]
        mask = feas & (~placed_by_a) & self.valid_rot.unsqueeze(0)
        return mask, best

    def _compute_mask_and_best_torch(self):
        """Pure-torch: F.max_pool2d per footprint + vectorized perimeter."""
        N = self.N
        GX, GY = self.GX, self.GY
        A = self.A
        dev = self.device
        hmap_f = self.hmap.float()

        best_score = torch.full((N, A), float("inf"), device=dev)
        best_x = torch.zeros((N, A), dtype=torch.int32, device=dev)
        best_y = torch.zeros((N, A), dtype=torch.int32, device=dev)
        best_z = torch.zeros((N, A), dtype=torch.int32, device=dev)

        nz_pad = F.pad(
            (self.hmap > 0).float(), (1, 1, 1, 1), value=1.0,
        )  # (N, GX+2, GY+2)

        fw_cpu = self.fw_of.cpu().tolist()
        fh_cpu = self.fh_of.cpu().tolist()
        fz_cpu = self.fz_of.cpu().tolist()

        # Group valid actions by (fw, fh).
        groups: dict[tuple[int, int], list[int]] = {}
        for a in range(A):
            fw, fh = fw_cpu[a], fh_cpu[a]
            if fw <= 0 or fh <= 0 or fw > GX or fh > GY:
                continue
            groups.setdefault((fw, fh), []).append(a)

        dist_map = self.dist_map  # (GX, GY)

        for (fw, fh), acts in groups.items():
            NX = GX - fw + 1
            NY = GY - fh + 1
            hm4 = hmap_f.unsqueeze(1)  # (N, 1, GX, GY)
            zmax = F.max_pool2d(hm4, (fw, fh), stride=1).squeeze(1)
            hsum = F.avg_pool2d(hm4, (fw, fh), stride=1).squeeze(1) * (fw * fh)
            ap = hsum / (fw * fh)

            # Contact perimeter (cells with h>0 adjacent; walls count as 1).
            nzp4 = nz_pad.unsqueeze(1)  # (N, 1, GX+2, GY+2)
            # 1D sum along x of width fw+2 -> (N, NX, GY+2)
            sum_x = (
                F.avg_pool2d(nzp4, (fw + 2, 1), stride=1).squeeze(1) * (fw + 2)
            )
            # 1D sum along y of width fh -> (N, GX+2, NY+2)
            sum_y = (
                F.avg_pool2d(nzp4, (1, fh), stride=1).squeeze(1) * fh
            )
            top = sum_x[:, :, :NY]                        # (N, NX, NY)
            bot = sum_x[:, :, fh + 1:fh + 1 + NY]
            left = sum_y[:, :NX, 1:1 + NY]
            right = sum_y[:, fw + 1:fw + 1 + NX, 1:1 + NY]
            contact = top + bot + left + right
            bt = 2 * (fw + fh) + 4
            contact_ratio = contact / (bt + 1e-9)

            penalty = zmax - ap
            dist = dist_map[:NX, :NY]
            score_base = (
                (zmax + penalty * self.PENALTY_W) * self.gr_z
                - contact_ratio * self.CONTACT_W + dist
            )

            # Exact support ratio = ap/zmax when zmax>0 else 1.0
            ratio = torch.where(
                zmax > 0, ap / zmax.clamp(min=1e-9),
                torch.ones_like(ap),
            )
            bad_sup = ratio < self.min_sup

            for a in acts:
                fz = fz_cpu[a]
                bad_top = (zmax + fz) > self.PH_v
                score = score_base.masked_fill(
                    bad_sup | bad_top, float("inf"),
                )
                flat = score.view(N, -1)
                min_val, argmin = flat.min(dim=1)
                gx_i = (argmin // NY).to(torch.int32)
                gy_i = (argmin % NY).to(torch.int32)
                gz_i = (
                    zmax.view(N, -1)
                    .gather(1, argmin.unsqueeze(1))
                    .squeeze(1)
                    .to(torch.int32)
                )
                best_score[:, a] = min_val
                best_x[:, a] = gx_i
                best_y[:, a] = gy_i
                best_z[:, a] = gz_i

        best = torch.stack([best_x, best_y, best_z], dim=-1)
        feas = torch.isfinite(best_score)
        placed_by_a = self.placed[:, self.box_of.long()]
        mask = feas & (~placed_by_a) & self.valid_rot.unsqueeze(0)
        return mask, best

    def _obs(self) -> torch.Tensor:
        hm = self.hmap.float() / max(self.PH_v, 1)
        p = self.obs_pool
        if p > 1:
            hm = F.max_pool2d(
                hm.unsqueeze(1), (p, p), ceil_mode=True,
            ).squeeze(1)
        hm_flat = hm.reshape(self.N, -1)
        remaining = (~self.placed).float()
        box_dims = self.box_dims_flat.unsqueeze(0).expand(self.N, -1)
        return torch.cat([hm_flat, remaining, box_dims], dim=1)

    def reset(self):
        self._reset_state()
        self._mask, self._best = self._compute_mask_and_best()
        return self._obs()

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(
                actions, device=self.device, dtype=torch.long,
            )
        actions = actions.long().to(self.device)
        if self._mask is None:
            self._mask, self._best = self._compute_mask_and_best()

        env_idx = torch.arange(self.N, device=self.device)
        chosen_feas = self._mask[env_idx, actions]
        best_ab = self._best[env_idx, actions]
        gx = best_ab[:, 0].long()
        gy = best_ab[:, 1].long()
        z = best_ab[:, 2].long()
        fw = self.fw_of[actions].long()
        fh = self.fh_of[actions].long()
        fz = self.fz_of[actions].long()
        box_i = self.box_of[actions].long()

        vol = torch.where(
            chosen_feas,
            self.vol_of[actions],
            torch.zeros_like(self.vol_of[actions]),
        )
        reward = vol.float() / self.pallet_base_ph

        feas_idx = chosen_feas.nonzero(as_tuple=True)[0]
        if feas_idx.numel() > 0:
            self.placed[feas_idx, box_i[feas_idx]] = True
            self.packed_vol[feas_idx] += self.vol_of[actions][feas_idx]
            self.packed_cnt[feas_idx] += 1
            new_top = (z + fz).to(torch.int32)
            self.max_top_z[feas_idx] = torch.maximum(
                self.max_top_z[feas_idx], new_top[feas_idx],
            )
            self._scatter_hmap(
                feas_idx, gx[feas_idx], gy[feas_idx],
                fw[feas_idx], fh[feas_idx], new_top[feas_idx],
            )

        self._mask, self._best = self._compute_mask_and_best()
        done_all = self.placed.all(dim=1)
        no_feas = ~self._mask.any(dim=1)
        dones = done_all | no_feas

        info: dict = {}
        if dones.any():
            # Snapshot episode-end stats before reset clears them.
            info["final_packed_vol"] = self.packed_vol.clone()
            info["final_packed_cnt"] = self.packed_cnt.clone()
            info["final_max_top_z"] = self.max_top_z.clone()
            self._reset_some(dones)
            self._mask, self._best = self._compute_mask_and_best()

        return self._obs(), reward, dones, info

    def _scatter_hmap(self, env_ids, gx, gy, fw, fh, top_vals):
        K = env_ids.numel()
        if K == 0:
            return
        max_fw = int(fw.max().item())
        max_fh = int(fh.max().item())
        dx = torch.arange(max_fw, device=self.device)
        dy = torch.arange(max_fh, device=self.device)
        fw_mask = dx.unsqueeze(0) < fw.unsqueeze(1)
        fh_mask = dy.unsqueeze(0) < fh.unsqueeze(1)
        cell_ok = fw_mask.unsqueeze(2) & fh_mask.unsqueeze(1)
        cx = gx.view(K, 1, 1) + dx.view(1, -1, 1)
        cy = gy.view(K, 1, 1) + dy.view(1, 1, -1)
        cx, cy = torch.broadcast_tensors(cx, cy)
        cx = cx.clamp(0, self.GX - 1)
        cy = cy.clamp(0, self.GY - 1)
        env_b = env_ids.view(K, 1, 1).expand(K, max_fw, max_fh)
        top_b = top_vals.view(K, 1, 1).expand(K, max_fw, max_fh)
        flat_env = env_b[cell_ok]
        flat_cx = cx[cell_ok]
        flat_cy = cy[cell_ok]
        values = top_b[cell_ok].to(torch.int32)
        flat_idx = (
            flat_env * (self.GX * self.GY)
            + flat_cx * self.GY
            + flat_cy
        )
        self.hmap.view(-1)[flat_idx] = values
