[English](README.md) | [中文](README_CN.md) | **日本語**

# パレタイジングプランナー

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
