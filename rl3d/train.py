"""Train masked discrete PPO on task_lx using rl_games + GPU VecEnv."""
from __future__ import annotations

import argparse
import time

import torch
from rl_games.algos_torch import a2c_discrete
from rl_games.torch_runner import Runner

from rl3d.env import load_task_lx
from rl3d.vec_env import PalletVecEnv


def _patched_get_masked_action_values(self, obs, action_masks):
    """rl_games default forces torch.BoolTensor(x) which copies GPU->CPU.
    Our env returns masks already on GPU — accept them directly."""
    processed_obs = self._preproc_obs(obs["obs"])
    if not isinstance(action_masks, torch.Tensor):
        action_masks = torch.as_tensor(action_masks, device=self.ppo_device)
    action_masks = action_masks.to(
        self.ppo_device, dtype=torch.bool, non_blocking=True,
    )
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": processed_obs,
        "action_masks": action_masks,
        "rnn_states": self.rnn_states,
    }
    with torch.no_grad():
        res_dict = self.model(input_dict)
        if self.has_central_value:
            value = self.get_central_value(
                {"is_train": False, "states": obs["states"]}
            )
            res_dict["values"] = value
    res_dict["action_masks"] = action_masks
    return res_dict


a2c_discrete.DiscreteA2CAgent.get_masked_action_values = (
    _patched_get_masked_action_values
)


def build_params(env: PalletVecEnv, args) -> dict:
    return {
        "params": {
            "seed": args.seed,
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
                "name": "pallet_ppo",
                "env_name": "pallet_env",
                "env_info": env.get_env_info(),
                "vec_env": env,
                "num_actors": args.n_envs,
                "use_action_masks": True,
                "reward_shaper": {"scale_value": 1.0},
                "device": str(env.device),
                "ppo": True,
                "gamma": 0.995,
                "tau": 0.95,
                "learning_rate": args.lr,
                "horizon_length": args.n_steps,
                "minibatch_size": args.batch_size,
                "mini_epochs": 4,
                "e_clip": 0.2,
                "entropy_coef": 0.01,
                "critic_coef": 1.0,
                "normalize_advantage": True,
                "normalize_input": False,
                "normalize_value": False,
                "max_epochs": args.epochs,
                "seq_length": 1,
                "save_frequency": 0,
                "save_best_after": 10,
                "full_experiment_name": args.save,
                "print_stats": True,
                "truncate_grads": True,
                "grad_norm": 1.0,
                "schedule_type": "standard",
                "lr_schedule": "None",
                "kl_threshold": 0.008,
                "clip_value": True,
                "score_to_win": 1.0,
                "bound_loss_type": "regularisation",
                "torch_compile": False,
            },
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_lx.json")
    ap.add_argument("--n-envs", type=int, default=4096)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--min-support", type=float, default=0.9)
    ap.add_argument("--save", default="pallet_ppo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inst = load_task_lx(args.task)
    print(
        f"Task: {len(inst.items)} boxes "
        f"pallet={inst.pallet_length}x{inst.pallet_width}"
        f"x{inst.pallet_max_height}"
    )

    env = PalletVecEnv(
        instance=inst, num_envs=args.n_envs,
        min_support_ratio=args.min_support,
    )
    print(
        f"VecEnv: N={env.N} obs_dim={env.obs_dim} A={env.A} "
        f"device={env.device}"
    )

    params = build_params(env, args)
    runner = Runner()
    runner.load_config(params=params["params"])

    t0 = time.perf_counter()
    runner.run_train({"train": True, "play": False})
    dt = time.perf_counter() - t0
    total = args.epochs * args.n_envs * args.n_steps
    print(
        f"Trained {args.epochs} epochs = {total} transitions "
        f"in {dt:.1f}s ({total / dt:.0f} fps)"
    )


if __name__ == "__main__":
    main()
