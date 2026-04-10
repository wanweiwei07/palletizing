# Palletizing Planner

一个最小可运行的码垛规划器，面向“给定一批相同尺寸箱子，在托盘上尽可能紧密码垛，并输出放置序列与姿态”的场景。

当前版本支持：

- 输入托盘长、宽、最大可用高度
- 输入箱子长、宽、高、数量
- 自动枚举箱子朝向，选择可放数量最多且底面利用率更高的方案
- 输出每个箱子的放置顺序和姿态
- 姿态包含箱子底面中心位置 `(x, y, z)` 与偏航角 `yaw`

## 快速运行

```bash
python main.py ^
  --pallet-length 1200 ^
  --pallet-width 1000 ^
  --pallet-max-height 1500 ^
  --box-length 300 ^
  --box-width 200 ^
  --box-height 250 ^
  --count 70
```

输出为 JSON，包含：

- `placements`: 每个箱子的序列和姿态
- `packed_count`: 实际成功规划的箱子数量
- `orientation`: 选中的箱体朝向
- `rotation`: 选中的离散旋转角
- `utilization_2d`: 托盘底面利用率
- `utilization_3d`: 托盘上方体积利用率

## 序列与姿态定义

- `sequence`: 第几个放置
- `x`, `y`, `z`: 箱体中心位置
- `yaw`: 只考虑 `0` 和 `pi/2`
- `size_x`, `size_y`, `size_z`: 规划后该箱子在托盘坐标系中的尺寸

当前规划策略是“规则网格 + 分层 + 蛇形序列”，适合作为第一版求解器或后续接入机器人执行系统的基础输出层。

## 多箱型任务生成

项目里现在也包含候选箱型 catalog，以及可重复箱型的 70 箱任务生成器。

直接生成一个任务：

```bash
python generate_task.py --count 70 --seed 7
```

输出包含：

- `summary`: 箱型统计、频率分组统计、总体积
- `boxes`: 70 个具体箱子实例

每个实例会带：

- `instance_id`
- `box_type_id`
- `length`, `width`, `height`
- `frequency_group`
- `allowed_yaws`

## Viewer 展示

如果想用项目里的 `one/viewer` 查看规划动画，可以直接运行：

```bash
python visualize_plan.py
```

脚本会：

- 生成托盘几何体
- 按放置序列一个一个生成箱子
- 根据 `yaw` 绕 `Z` 轴旋转箱子
- 用不同颜色区分不同层

常用控制：

- `Space`: 暂停 / 继续
- `Right`: 手动前进一步
- `R`: 从头重播

你也可以调动画速度：

```bash
python visualize_plan.py --step-interval 0.1
```

## 多箱型托盘展示

如果想把多箱型任务按真实托盘尺寸做一个 baseline 排布，再用 viewer 播放，可以运行：

```bash
python visualize_task_on_pallet.py --task-file task_70_seed7.json
```

当前使用的是一个启发式 baseline：

- 箱子按底面积和高度优先排序
- 逐层排布
- 每层采用货架式行布局
- 每个箱子只考虑 `yaw = 0` 或 `pi/2`
