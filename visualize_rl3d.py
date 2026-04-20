"""Run a trained rl_games PPO on task_lx and visualize the best rollout.

Default: N=16384 stochastic rollouts at T=1.5, pick the env with highest
fill rate, replay its placement sequence in the viewer.
"""
from __future__ import annotations

import argparse

import torch
from rl_games.algos_torch.players import PpoPlayerDiscrete

import rl3d.eval as rl_eval
from fill3d.cpsat_model import Fill3DPlacement
from rl3d.env import load_task_lx
from rl3d.vec_env import PalletVecEnv
from rl3d.eval import _patched_get_masked_action, build_params
from visualize_ga3d import GA3DViewer


PpoPlayerDiscrete.get_masked_action = _patched_get_masked_action


def best_rollout(
    env: PalletVecEnv, player, deterministic: bool,
) -> tuple[list[Fill3DPlacement], float]:
    """Return (placements_of_best_env, its_vol_ratio)."""
    obs = env.reset()
    items = env.inst.items
    N = env.N
    dev = env.device
    # Per-env placement log. Each entry: (step_idx, box_i, rot, gx, gy, gz).
    logs: list[list[tuple[int, int, bool, int, int, int]]] = [
        [] for _ in range(N)
    ]
    episode_done = torch.zeros(N, dtype=torch.bool, device=dev)
    final_vol = torch.zeros(N, dtype=torch.float32, device=dev)
    final_cnt = torch.zeros(N, dtype=torch.int32, device=dev)

    for step in range(env.n + 4):
        if episode_done.all():
            break
        masks = env.get_action_masks()
        acts = player.get_masked_action(
            obs, masks, is_deterministic=deterministic,
        )
        if acts.ndim == 0:
            acts = acts.view(1)

        env_idx = torch.arange(N, device=dev)
        feas = env._mask[env_idx, acts]          # (N,) bool
        best = env._best[env_idx, acts]          # (N, 3) int
        box_i = env.box_of[acts]                 # (N,) int
        rot_i = env.rot_of[acts]                 # (N,) int

        feas_cpu = feas.cpu().tolist()
        bxy_cpu = best.cpu().tolist()
        bi_cpu = box_i.cpu().tolist()
        ri_cpu = rot_i.cpu().tolist()
        done_cpu = episode_done.cpu().tolist()
        for i in range(N):
            if done_cpu[i] or not feas_cpu[i]:
                continue
            gx, gy, gz = bxy_cpu[i]
            logs[i].append(
                (step, bi_cpu[i], ri_cpu[i] == 1, gx, gy, gz),
            )

        obs, _, dones, info = env.step(acts)
        newly = dones & (~episode_done)
        if newly.any():
            idx = newly.nonzero(as_tuple=True)[0]
            top = info["final_max_top_z"][idx].to(torch.int64) * env.gr_z
            used = torch.where(
                top > 0,
                env.pallet_area * top,
                torch.ones_like(top),
            )
            vr = info["final_packed_vol"][idx].float() / used.float()
            final_vol[idx] = vr
            final_cnt[idx] = info["final_packed_cnt"][idx]
            episode_done |= newly

    best_env = int(final_vol.argmax().item())
    best_vr = float(final_vol[best_env].item())

    placements: list[Fill3DPlacement] = []
    for (_step, bi, rotated, gx, gy, gz) in logs[best_env]:
        it = items[bi]
        sx = it.width if rotated else it.length
        sy = it.length if rotated else it.width
        placements.append(Fill3DPlacement(
            item_id=it.item_id,
            x=gx * env.gr_xy,
            y=gy * env.gr_xy,
            z=gz * env.gr_z,
            rotated=rotated,
            size_x=sx,
            size_y=sy,
            size_z=it.height,
        ))
    return placements, best_vr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_lx.json")
    ap.add_argument(
        "--model", default="runs/pallet_ppo/nn/pallet_ppo.pth",
    )
    ap.add_argument("--min-support", type=float, default=0.9)
    ap.add_argument("--n-envs", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    rl_eval._TEMPERATURE = args.temperature

    inst = load_task_lx(args.task)
    env = PalletVecEnv(
        instance=inst, num_envs=args.n_envs,
        min_support_ratio=args.min_support,
    )
    params = build_params(env)
    player = PpoPlayerDiscrete(params=params)
    player.restore(args.model)
    player.has_batch_dimension = True

    placements, vr = best_rollout(env, player, args.deterministic)

    packed_vol = sum(p.size_x * p.size_y * p.size_z for p in placements)
    max_top = max((p.z + p.size_z for p in placements), default=0)
    info = (
        f"RL3D N={args.n_envs} T={args.temperature} "
        f"packed={len(placements)}/{len(inst.items)} "
        f"vol={vr:.1%} top={max_top}mm"
    )
    print(info)

    viewer = GA3DViewer(
        placements=placements,
        pallet_length=inst.pallet_length,
        pallet_width=inst.pallet_width,
        pallet_max_height=inst.pallet_max_height,
        result_info=info,
    )
    viewer.run()


if __name__ == "__main__":
    main()
