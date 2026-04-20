**English** | [中文](README_CN.md) | [日本語](README_JP.md)

# Palletizing Planner

A 3D bin packing solver for robotic palletizing. Given a set of boxes with varying dimensions, it computes placement positions and orientations to maximize pallet space utilization.

## Solvers

### GA3D — GPU-Accelerated Genetic Algorithm
- Chromosome = permutation of box order + rotation flags
- Decoder = heightmap + bit-packed voxel, greedy best-fit placement with support constraint
- Population decoded in parallel on GPU (fused Triton scan kernel, one block per chromosome)
- Structured initialization: 2/3 of population sorted long-to-short (half rotates steps `[0, n/2)`, half rotates `[n/2, n)`), remaining 1/3 random permutation with alternating rotation

### RL3D — Masked PPO Reinforcement Learning
- GPU-native vectorized env (rl_games + PyTorch), all tensors stay on GPU end-to-end
- Discrete masked action space: `A = 2n` (box × rotation), infeasible actions masked each step
- Per-step greedy scan: `F.max_pool2d` over unique footprints at 10 mm grid (~12k env-steps/s at N=16384)
- MLP policy `[512, 256, 128]`, PPO with CategoricalMasked; inference uses **stochastic rollout + best-of-N with temperature**

### CP-SAT — Constraint Programming (fill3d)
- Exact solver using Google OR-Tools CP-SAT
- Probe-based support area constraints
- Optimal for smaller instances

### Layer-Based Planner (palletizing)
- Groups boxes by height into layers
- 2D packing per layer using CP-SAT
- Stacks layers with support validation

## Quick Start

### GA3D — Mixed-Type 3D Packing + Visualization
```bash
python visualize_ga3d.py 7      # task_70_seed7
```

### RL3D — Train / Evaluate / Visualize
```bash
# Train PPO on task_lx (30 epochs, ~24 min on RTX 5090 at N=16384)
PYTHONPATH=. python -m rl3d.train --n-envs 16384 --n-steps 32 \
    --batch-size 131072 --epochs 30

# Evaluate best-of-N stochastic rollout
PYTHONPATH=. python -m rl3d.eval \
    --model runs/pallet_ppo/nn/pallet_ppo.pth \
    --n-envs 16384 --temperature 1.5

# Visualize the best of N=16384 rollouts
python visualize_rl3d.py --temperature 1.5
```

### Layer-Based Planner — Single-Type Packing
```bash
python main.py \
  --pallet-length 1200 --pallet-width 1000 --pallet-max-height 1500 \
  --box-length 300 --box-width 200 --box-height 250 --count 70
```

### Task Generation
```bash
python generate_task.py --count 70 --seed 7
```

## Viewer Controls

- `Space`: Pause / Resume
- `Right Arrow`: Step forward one box
- `R`: Restart

## Project Structure

```
ga3d/           GPU-accelerated genetic algorithm solver
rl3d/           Masked PPO RL solver (rl_games + GPU VecEnv)
fill3d/         3D CP-SAT constraint solver
fill2d/         2D CP-SAT bin packing solver
palletizing/    Layer-based planner, task generator, box catalog
one/            OpenGL viewer (from [one](https://github.com/wanweiwei07/one))
tests/          Unit tests
```

## Benchmarks

### task_70 (70 mixed-type boxes, pop=1024, 30s)

| Seed | Packed | Volume Ratio |
|------|--------|-------------|
| 7    | 65/70  | 90.6%       |
| 42   | 67/70  | 87.0%       |
| 99   | 67/70  | 80.9%       |
| 123  | 70/70  | 85.3%       |

### task_lx (35 boxes, pallet 1000×1200×1600, RTX 5090, seed=42)

| pop  | 30s          | 60s          | s/gen |
|------|--------------|--------------|-------|
| 512  | 78.7% (88g)  | 78.7% (178g) | 0.34  |
| 1024 | 78.1% (60g)  | 78.1% (120g) | 0.50  |
| 2048 | 78.1% (35g)  | 78.1% (69g)  | 0.87  |
| 4096 | 78.1% (18g)  | 78.7% (36g)  | 1.69  |

All boxes placed in every configuration. Volume ratio clusters tightly at 78.1–78.7% — with the structured initialization, pop=512/30s already reaches the ceiling on this instance.

Volume ratio = packed volume / (pallet base area × max box top height)

### task_lx — GA3D vs RL3D (seed=42)

| Solver | Setting                           | Packed | Volume Ratio | Notes             |
|--------|-----------------------------------|--------|--------------|-------------------|
| GA3D   | pop=1024, 60 s                    | 35/35  | 78.1%        | plateau           |
| GA3D   | pop=4096, 60 s                    | 35/35  | 78.7%        | peak ~80%         |
| RL3D   | N=256, deterministic              | 35/35  | 74.6%        | single rollout    |
| RL3D   | N=16384, T=1.0, best-of-N         | 35/35  | 79.2%        | —                 |
| RL3D   | **N=16384, T=1.5, best-of-N**     | 35/35  | **80.3%**    | matches GA peak   |

The RL policy's final boost comes from stochastic sampling + temperature at inference, not more training — the classic neural combinatorial optimization pattern (POMO, Attention Model). N=16384 rollouts finish in ~3 s on RTX 5090 because the vectorized env runs ~12k env-steps/s.

## Tuning Guide

`GA3DConfig` parameters (see [ga3d/solver.py](ga3d/solver.py)):

| Param | Default | Effect |
|-------|---------|--------|
| `pop_size`           | 2048  | Diversity vs per-generation cost. Larger is slower per gen but explores more. Scaling is roughly linear in s/gen. |
| `time_limit_seconds` | 60.0  | Wall-clock budget. GA runs until elapsed ≥ this value. |
| `mutation_rate`      | 0.15  | Per-chromosome swap-mutation probability. Raise for more exploration. |
| `crossover_rate`     | 0.8   | OX1 crossover probability. |
| `elite_ratio`        | 0.1   | Fraction of each generation copied directly from top performers. |
| `tournament_k`       | 4     | Tournament selection size. |
| `min_support_ratio`  | 0.8   | Reject placements below this support fraction. Raise (0.9) for safer stacks, lower for tighter packing. |
| `grid_res_xy/z`      | 10 mm | Voxel resolution. Smaller is more precise but slower (memory + compute grow quadratically/cubically). |
| `use_triton`         | True  | Fused Triton scan kernel. Disable only for debugging. |
| `seed`               | None  | Set for reproducible runs. |

**Practical recommendations:**
- Small / well-structured tasks (≤ 50 boxes): `pop=512, time=30s` is usually enough.
- Medium tasks (50–100 boxes): `pop=1024–2048, time=30–60s`.
- Hard / highly mixed tasks: `pop=2048–4096, time=60–120s`.
- If per-gen time is the bottleneck, lower `pop_size`; if convergence stalls early, raise `pop_size` or `mutation_rate`.
- On task_lx the sweep is flat — indicating the decoder/objective ceiling, not a GA limitation. Further gains need decoder changes (void-fill, better support scoring) rather than GA tuning.
