[English](README.md) | **中文** | [日本語](README_JP.md)

# 码垛规划器

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
