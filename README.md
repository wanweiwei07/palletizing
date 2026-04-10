**English** | [中文](README_CN.md) | [日本語](README_JP.md)

# Palletizing Planner

A 3D bin packing solver for robotic palletizing. Given a set of boxes with varying dimensions, it computes placement positions and orientations to maximize pallet space utilization.

## Solvers

### GA3D — GPU-Accelerated Genetic Algorithm
- Chromosome = permutation of box order + rotation flags
- Decoder = heightmap-based greedy placement with support constraint
- Population decoded in parallel on GPU (CUDA)
- Best-fit scoring with side contact reward
- Height-grouped initialization for better convergence

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
python visualize_ga3d.py 7
```
Runs GA3D on `task_70_seed7_dims.json` and opens a 3D viewer.

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
fill3d/         3D CP-SAT constraint solver
fill2d/         2D CP-SAT bin packing solver
palletizing/    Layer-based planner, task generator, box catalog
one/            OpenGL viewer (from [one](https://github.com/wanweiwei07/one))
tests/          Unit tests
```

## Benchmark (70 mixed-type boxes, 30s, pop=1024)

| Seed | Packed | Volume Ratio |
|------|--------|-------------|
| 7    | 65/70  | 90.6%       |
| 42   | 67/70  | 87.0%       |
| 99   | 67/70  | 80.9%       |
| 123  | 70/70  | 85.3%       |

Volume ratio = packed volume / (pallet base area x max box top height)
