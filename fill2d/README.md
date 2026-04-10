# Fill2D

一个单层二维填充的 CP-SAT 原型。

## 建模

给定托盘矩形和若干候选块 `i`：

- 使用变量 `u_i ∈ {0,1}` 表示块是否被选中
- 旋转变量 `r_i ∈ {0,1}` 表示是否旋转 90 度
- 位置变量 `x_i, y_i` 表示块左下角坐标

旋转后的尺寸：

```text
width_i  = (1-r_i) * w_i + r_i * h_i
height_i = (1-r_i) * h_i + r_i * w_i
```

边界约束：

```text
x_i + width_i  <= pallet_width
y_i + height_i <= pallet_height
x_i >= 0
y_i >= 0
```

不重叠约束对每对块 `(i, j)` 引入：

```text
L_ij, R_ij, B_ij, T_ij ∈ {0,1}
L_ij + R_ij + B_ij + T_ij >= u_i + u_j - 1
```

并用 big-M 写四种相对位置：

```text
x_i + width_i  <= x_j + M * (1 - L_ij)
x_j + width_j  <= x_i + M * (1 - R_ij)
y_i + height_i <= y_j + M * (1 - B_ij)
y_j + height_j <= y_i + M * (1 - T_ij)
```

目标函数：

```text
max Σ u_i * area_i
```

## 运行

先安装 `ortools`：

```bash
python -m pip install ortools
```

再运行示例：

```bash
python -m fill2d.demo
```
