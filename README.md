[English](#english) | [中文](#中文) | [日本語](#日本語)

---

# English

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

---

# 中文

面向机器人码垛的三维装箱求解器。给定一组不同尺寸的箱子，计算放置位置和朝向，最大化托盘空间利用率。

## 求解器

### GA3D — GPU 加速遗传算法
- 染色体 = 箱子放置顺序排列 + 旋转标志
- 解码器 = 基于高度图的贪心放置 + 支撑约束
- 种群在 GPU (CUDA) 上并行解码
- Best-fit 评分，侧面贴合奖励
- 按高度分组初始化，加速收敛

### CP-SAT — 约束规划 (fill3d)
- 使用 Google OR-Tools CP-SAT 精确求解
- 基于探针的支撑面积约束
- 适合小规模实例

### 分层规划器 (palletizing)
- 按高度将箱子分组为层
- 每层使用 CP-SAT 做二维排列
- 层间支撑验证

## 快速开始

### GA3D — 多箱型三维装箱 + 可视化
```bash
python visualize_ga3d.py 7
```
对 `task_70_seed7_dims.json` 运行 GA3D 求解并打开三维可视化。

### 分层规划器 — 单箱型码垛
```bash
python main.py \
  --pallet-length 1200 --pallet-width 1000 --pallet-max-height 1500 \
  --box-length 300 --box-width 200 --box-height 250 --count 70
```

### 生成任务
```bash
python generate_task.py --count 70 --seed 7
```

## 查看器操作

- `Space`：暂停 / 继续
- `Right`：手动前进一步
- `R`：重新播放

## 项目结构

```
ga3d/           GPU 加速遗传算法求解器
fill3d/         三维 CP-SAT 约束求解器
fill2d/         二维 CP-SAT 装箱求解器
palletizing/    分层规划器、任务生成器、箱型目录
one/            OpenGL 可视化（复用 [one](https://github.com/wanweiwei07/one) 框架）
tests/          单元测试
```

## 性能测试 (70 个混合箱型, 30 秒, pop=1024)

| Seed | 装载 | 体积利用率 |
|------|------|-----------|
| 7    | 65/70  | 90.6%   |
| 42   | 67/70  | 87.0%   |
| 99   | 67/70  | 80.9%   |
| 123  | 70/70  | 85.3%   |

体积利用率 = 箱子总体积 / (托盘底面积 x 最高箱子顶部高度)

---

# 日本語

ロボットパレタイジング向けの三次元ビンパッキングソルバー。異なるサイズの箱が与えられた場合、配置位置と向きを計算し、パレット空間利用率を最大化します。

## ソルバー

### GA3D — GPU 高速化遺伝的アルゴリズム
- 染色体 = 箱の配置順序の順列 + 回転フラグ
- デコーダ = ハイトマップベースの貪欲配置 + 支持面制約
- GPU (CUDA) 上で集団を並列デコード
- Best-fit スコアリング、側面接触報酬
- 高さ別グループ初期化による収束改善

### CP-SAT — 制約プログラミング (fill3d)
- Google OR-Tools CP-SAT による厳密解法
- プローブベースの支持面積制約
- 小規模インスタンスに最適

### レイヤーベースプランナー (palletizing)
- 箱を高さ別にレイヤーにグループ化
- 各レイヤーで CP-SAT による二次元配置
- レイヤー間の支持面検証

## クイックスタート

### GA3D — 混合箱型 3D パッキング + 可視化
```bash
python visualize_ga3d.py 7
```
`task_70_seed7_dims.json` に対して GA3D を実行し、3D ビューアを開きます。

### レイヤープランナー — 単一箱型パッキング
```bash
python main.py \
  --pallet-length 1200 --pallet-width 1000 --pallet-max-height 1500 \
  --box-length 300 --box-width 200 --box-height 250 --count 70
```

### タスク生成
```bash
python generate_task.py --count 70 --seed 7
```

## ビューア操作

- `Space`：一時停止 / 再開
- `Right`：1ステップ進む
- `R`：最初から再生

## プロジェクト構成

```
ga3d/           GPU 高速化遺伝的アルゴリズムソルバー
fill3d/         3D CP-SAT 制約ソルバー
fill2d/         2D CP-SAT ビンパッキングソルバー
palletizing/    レイヤープランナー、タスク生成器、箱型カタログ
one/            OpenGL 可視化（[one](https://github.com/wanweiwei07/one) フレームワークを利用）
tests/          ユニットテスト
```

## ベンチマーク (混合箱型 70 個, 30 秒, pop=1024)

| Seed | 積載 | 体積利用率 |
|------|------|-----------|
| 7    | 65/70  | 90.6%   |
| 42   | 67/70  | 87.0%   |
| 99   | 67/70  | 80.9%   |
| 123  | 70/70  | 85.3%   |

体積利用率 = 箱総体積 / (パレット底面積 x 最上箱の頂部高さ)
