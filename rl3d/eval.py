"""Evaluate a trained rl_games PPO on task_lx."""
from __future__ import annotations

import argparse

import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete

from rl3d.env import load_task_lx
from rl3d.vec_env import PalletVecEnv


_TEMPERATURE = 1.0


def _patched_get_masked_action(self, obs, action_masks, is_deterministic=True):
    """rl_games default does torch.Tensor(x) on action_masks which copies
    GPU->CPU. Our env returns masks already on GPU — accept them directly.

    Also supports temperature via module-level _TEMPERATURE (>1 flattens,
    <1 sharpens). T=1.0 is the model's native distribution.
    """
    if not self.has_batch_dimension:
        from rl_games.common.tr_helpers import unsqueeze_obs
        obs = unsqueeze_obs(obs)
    obs = self._preproc_obs(obs)
    if not isinstance(action_masks, torch.Tensor):
        action_masks = torch.as_tensor(
            action_masks, device=self.device,
        )
    action_masks = action_masks.to(
        self.device, dtype=torch.bool, non_blocking=True,
    )
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": obs,
        "action_masks": action_masks,
        "rnn_states": self.states,
    }
    self.model.eval()
    with torch.no_grad():
        res_dict = self.model(input_dict)
    logits = res_dict["logits"].detach()
    self.states = res_dict["rnn_states"]
    if is_deterministic:
        return torch.argmax(logits, axis=-1).squeeze()
    if _TEMPERATURE != 1.0:
        logits = logits / _TEMPERATURE
    action = torch.distributions.Categorical(logits=logits).sample()
    return action.squeeze()


PpoPlayerDiscrete.get_masked_action = _patched_get_masked_action


def build_params(env: PalletVecEnv) -> dict:
    return {
        "seed": 0,
        "algo": {"name": "a2c_discrete"},
        "model": {"name": "discrete_a2c"},
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {"discrete": {}},
            "mlp": {
                "units": [512, 256, 128],
                "activation": "elu",
                "initializer": {"name": "default"},
            },
        },
        "config": {
            "name": "pallet_eval",
            "env_name": "pallet_env",
            "env_info": env.get_env_info(),
            "vec_env": env,
            "num_actors": env.N,
            "use_action_masks": True,
            "reward_shaper": {"scale_value": 1.0},
            "device": str(env.device),
            "ppo": True,
            "normalize_input": False,
            "normalize_value": False,
            "max_epochs": 1,
            "horizon_length": 32,
            "minibatch_size": env.N * 32,
            "mini_epochs": 1,
            "e_clip": 0.2,
            "entropy_coef": 0.0,
            "critic_coef": 1.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_lx.json")
    ap.add_argument("--model", required=True, help="path to .pth checkpoint")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--min-support", type=float, default=0.9)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    global _TEMPERATURE
    _TEMPERATURE = args.temperature

    inst = load_task_lx(args.task)
    env = PalletVecEnv(
        instance=inst, num_envs=args.n_envs,
        min_support_ratio=args.min_support,
    )
    params = build_params(env)

    player = PpoPlayerDiscrete(params=params)
    player.restore(args.model)
    player.has_batch_dimension = True

    obs = env.reset()
    dones = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    episode_done = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    best_vol = torch.zeros(env.N, dtype=torch.float32, device=env.device)
    best_cnt = torch.zeros(env.N, dtype=torch.int32, device=env.device)

    steps = 0
    max_steps = env.n + 4
    while not episode_done.all() and steps < max_steps:
        masks = env.get_action_masks()
        action = player.get_masked_action(
            obs, masks, is_deterministic=args.deterministic,
        )
        obs, _rew, dones, info = env.step(action)
        newly_done = dones & (~episode_done)
        if newly_done.any():
            idx = newly_done.nonzero(as_tuple=True)[0]
            f_top = info["final_max_top_z"]
            f_vol = info["final_packed_vol"]
            f_cnt = info["final_packed_cnt"]
            max_top_mm = f_top[idx].to(torch.int64) * env.gr_z
            used = torch.where(
                max_top_mm > 0,
                env.pallet_area * max_top_mm,
                torch.ones_like(max_top_mm),
            )
            vr = f_vol[idx].float() / used.float()
            best_vol[idx] = vr
            best_cnt[idx] = f_cnt[idx]
        episode_done |= dones
        steps += 1

    vr_cpu = best_vol.cpu().numpy()
    cnt_cpu = best_cnt.cpu().numpy()
    best_idx = int(vr_cpu.argmax())
    print(
        f"rollouts={env.N} mean_vol={vr_cpu.mean():.1%} "
        f"best_vol={vr_cpu[best_idx]:.1%} packed={cnt_cpu[best_idx]}/{env.n}"
    )


if __name__ == "__main__":
    main()
