"""Fused per-(env, action) placement scan for RL VecEnv.

One Triton block per (env_i, action_i) pair. Each block reads its env's
hmap, scans all (gx, gy) for action's (fw, fh) in BX*BY tiles, applies
support-ratio and z-top filters, and returns argmin. Replaces the
Python loop over distinct footprints in vec_env._compute_mask_and_best.
"""
from __future__ import annotations

import os

import torch


def _install_windows_triton_patches():
    """Windows-only fixes for Triton + MinGW gcc.

    1. Strip `\\\\?\\` long-path prefix (MinGW can't handle it).
    2. Point cuda.lib lookup at bundled NVIDIA backend (user-site
       installs break `sysconfig.get_paths()['platlib']` probing).
    """
    if os.name != "nt":
        return

    import sysconfig
    from pathlib import Path

    for var in ("TRITON_LIBCUDA_PATH", "CUDA_PATH"):
        if os.environ.get(var):
            continue
        candidates = [
            Path(sysconfig.get_paths()["platlib"]),
            Path(sysconfig.get_paths()["purelib"]),
            Path(__file__).resolve().parents[2],
        ]
        import site
        candidates.extend(Path(p) for p in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
        for base in candidates:
            nvidia = base / "triton" / "backends" / "nvidia"
            if (nvidia / "lib" / "x64" / "cuda.lib").exists():
                if var == "TRITON_LIBCUDA_PATH":
                    os.environ[var] = str(nvidia / "lib" / "x64")
                else:
                    os.environ[var] = str(nvidia)
                break

    def _no_prefix(path: str) -> str:
        p = os.path.abspath(path).replace("/", "\\")
        if p.startswith("\\\\?\\"):
            p = p[4:]
        return p

    import triton.runtime.build as _b
    import triton.windows_utils as _wu
    _b.normalize_path = _no_prefix
    _wu.normalize_path = _no_prefix


_install_windows_triton_patches()

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _scan_kernel_rl(
    hmap_ptr,          # (N, GX, GY) int32, contiguous
    fw_ptr, fh_ptr, fz_ptr,   # (A,) int32
    best_x_ptr, best_y_ptr, best_z_ptr, best_score_ptr,  # (N, A)
    A: tl.constexpr,
    GX: tl.constexpr,
    GY: tl.constexpr,
    PH_V: tl.constexpr,
    PENALTY_W: tl.constexpr,
    CONTACT_W: tl.constexpr,
    DIST_W: tl.constexpr,
    GR_Z: tl.constexpr,
    MIN_SUP: tl.constexpr,
    BX: tl.constexpr,
    BY: tl.constexpr,
):
    pid = tl.program_id(0)
    env_i = pid // A
    act_i = pid % A

    fw = tl.load(fw_ptr + act_i)
    fh = tl.load(fh_ptr + act_i)
    fz = tl.load(fz_ptr + act_i)

    INF = float('inf')
    # Invalid footprint (rotation disabled, or oversize) -> infeasible.
    invalid = (fw <= 0) | (fh <= 0) | (fw > GX) | (fh > GY)

    NX = GX - fw + 1
    NY = GY - fh + 1

    hmap_base = hmap_ptr + env_i * GX * GY

    best_score = INF
    best_x = 0
    best_y = 0
    best_z = 0

    fw_f = fw.to(tl.float32)
    fh_f = fh.to(tl.float32)
    inv_inner = 1.0 / (fw_f * fh_f + 1e-9)
    bt = 2 * (fw + fh) + 4
    inv_border = 1.0 / (bt.to(tl.float32) + 1e-9)

    dx_r = tl.arange(0, BX)
    dy_r = tl.arange(0, BY)

    if not invalid:
        for tx in range(0, NX, BX):
            for ty in range(0, NY, BY):
                xi = tx + dx_r[:, None]
                yi = ty + dy_r[None, :]
                pos_mask = (xi < NX) & (yi < NY)

                z_bot = tl.zeros((BX, BY), dtype=tl.int32)
                sum_h = tl.zeros((BX, BY), dtype=tl.int32)
                for ddx in range(fw):
                    for ddy in range(fh):
                        xx = xi + ddx
                        yy = yi + ddy
                        off = xx * GY + yy
                        h = tl.load(hmap_base + off,
                                    mask=pos_mask, other=0)
                        z_bot = tl.maximum(z_bot, h)
                        sum_h = sum_h + h

                contact_cnt = tl.zeros((BX, BY), dtype=tl.int32)

                yy_top = yi - 1
                yy_bot = yi + fh
                for dd in range(fw + 2):
                    xx = xi - 1 + dd
                    valid_x = (xx >= 0) & (xx < GX)

                    vt = valid_x & (yy_top >= 0)
                    h_t = tl.load(hmap_base + xx * GY + yy_top,
                                  mask=pos_mask & vt, other=0)
                    cell_t = tl.where(vt, (h_t > 0).to(tl.int32), 1)
                    contact_cnt += tl.where(pos_mask, cell_t, 0)

                    vb = valid_x & (yy_bot < GY)
                    h_b = tl.load(hmap_base + xx * GY + yy_bot,
                                  mask=pos_mask & vb, other=0)
                    cell_b = tl.where(vb, (h_b > 0).to(tl.int32), 1)
                    contact_cnt += tl.where(pos_mask, cell_b, 0)

                xx_lft = xi - 1
                xx_rgt = xi + fw
                for dd in range(fh):
                    yy = yi + dd
                    valid_y = (yy >= 0) & (yy < GY)

                    vl = valid_y & (xx_lft >= 0)
                    h_l = tl.load(hmap_base + xx_lft * GY + yy,
                                  mask=pos_mask & vl, other=0)
                    cell_l = tl.where(vl, (h_l > 0).to(tl.int32), 1)
                    contact_cnt += tl.where(pos_mask, cell_l, 0)

                    vr = valid_y & (xx_rgt < GX)
                    h_r = tl.load(hmap_base + xx_rgt * GY + yy,
                                  mask=pos_mask & vr, other=0)
                    cell_r = tl.where(vr, (h_r > 0).to(tl.int32), 1)
                    contact_cnt += tl.where(pos_mask, cell_r, 0)

                contact_ratio = contact_cnt.to(tl.float32) * inv_border

                z_f = z_bot.to(tl.float32)
                ap = sum_h.to(tl.float32) * inv_inner
                penalty = z_f - ap
                dist = (
                    xi.to(tl.float32) + yi.to(tl.float32)
                ) * DIST_W
                score = (
                    (z_f + penalty * PENALTY_W) * GR_Z
                    - contact_ratio * CONTACT_W + dist
                )

                ratio = tl.where(
                    z_bot > 0,
                    ap / tl.where(z_bot > 0, z_f, 1.0),
                    1.0,
                )
                bad_sup = ratio < MIN_SUP
                bad_top = (z_bot + fz) > PH_V
                score = tl.where(bad_sup | bad_top, INF, score)
                score = tl.where(pos_mask, score, INF)

                flat_score = tl.reshape(score, (BX * BY,))
                flat_z = tl.reshape(z_bot, (BX * BY,))
                flat_xi = tl.reshape(
                    xi + tl.zeros((BX, BY), dtype=tl.int32),
                    (BX * BY,),
                )
                flat_yi = tl.reshape(
                    yi + tl.zeros((BX, BY), dtype=tl.int32),
                    (BX * BY,),
                )
                idxs = tl.arange(0, BX * BY)

                tile_min = tl.min(flat_score, axis=0)
                is_min = flat_score == tile_min
                BIG = BX * BY + 1
                argmin_cand = tl.where(is_min, idxs, BIG)
                tile_argmin = tl.min(argmin_cand, axis=0)
                pick = (idxs == tile_argmin).to(tl.int32)
                tile_z = tl.sum(flat_z * pick, axis=0)
                tile_x = tl.sum(flat_xi * pick, axis=0)
                tile_y = tl.sum(flat_yi * pick, axis=0)

                update = tile_min < best_score
                best_score = tl.where(update, tile_min, best_score)
                best_x = tl.where(update, tile_x, best_x)
                best_y = tl.where(update, tile_y, best_y)
                best_z = tl.where(update, tile_z, best_z)

    out_off = env_i * A + act_i
    tl.store(best_x_ptr + out_off, best_x)
    tl.store(best_y_ptr + out_off, best_y)
    tl.store(best_z_ptr + out_off, best_z)
    tl.store(best_score_ptr + out_off, best_score)


def triton_scan_rl(
    hmap: torch.Tensor,    # (N, GX, GY) int32
    fw_of_a: torch.Tensor,  # (A,) int32
    fh_of_a: torch.Tensor,  # (A,) int32
    fz_of_a: torch.Tensor,  # (A,) int32
    *,
    ph_v: int,
    penalty_w: float,
    contact_w: float,
    dist_w: float,
    gr_z: int,
    min_support_ratio: float,
    bx: int = 16,
    by: int = 16,
    num_warps: int = 4,
):
    """Return (best_x, best_y, best_z, best_score) shape (N, A) on GPU."""
    assert hmap.is_cuda and hmap.dtype == torch.int32
    assert hmap.is_contiguous()
    assert fw_of_a.dtype == torch.int32
    assert fh_of_a.dtype == torch.int32
    assert fz_of_a.dtype == torch.int32
    N, GX, GY = hmap.shape
    A = fw_of_a.numel()
    dev = hmap.device

    best_x = torch.zeros((N, A), dtype=torch.int32, device=dev)
    best_y = torch.zeros((N, A), dtype=torch.int32, device=dev)
    best_z = torch.zeros((N, A), dtype=torch.int32, device=dev)
    best_s = torch.empty((N, A), dtype=torch.float32, device=dev)

    grid = (N * A,)
    _scan_kernel_rl[grid](
        hmap, fw_of_a, fh_of_a, fz_of_a,
        best_x, best_y, best_z, best_s,
        A=int(A),
        GX=int(GX), GY=int(GY),
        PH_V=int(ph_v),
        PENALTY_W=float(penalty_w),
        CONTACT_W=float(contact_w),
        DIST_W=float(dist_w),
        GR_Z=int(gr_z),
        MIN_SUP=float(min_support_ratio),
        BX=int(bx), BY=int(by),
        num_warps=num_warps,
    )
    return best_x, best_y, best_z, best_s
