"""补齐两处证据：B9 的逐题差值标准差、以及在**同一条件下**的模型名次。

为什么重算名次
--------------
上一版脚本用表 5 的 A−B / B−D 列排名次，那是"跨条件差值"，不是
"某格内的模型名次"。审稿人问的是后者：在条件 A 里八个模型谁强谁弱
（表 8 的 Δ），在条件 D 里又是谁强谁弱（表 9 的 Δ）。两者不是一回事。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"

# ---------------------------------------------------------------- B9 细节
print("=" * 92)
print("B9 明细（逐题差值标准差 → 解释三个 Δ 的隐含 SE 差异）")
print("=" * 92)
boot = json.loads((RES / "llm_vs_trained_bootstrap.json").read_text(encoding="utf-8"))
print(f"  n_items={boot['n_items']}  n_boot={boot['n_boot']}  seed={boot['seed']}")
print(f"  训练侧 MAE（五种子均值）={boot['trained_mae_seedmean']:.3f}")
print(f"  训练侧 MAE（集成）      ={boot['trained_mae_ensemble']:.3f}")
print(f"  target = {boot['target']}")
mdl = boot["models"]
if isinstance(mdl, dict):
    for name, v in mdl.items():
        print(f"\n  [{name}]")
        for k, vv in v.items():
            if isinstance(vv, (list, dict)):
                print(f"    {k}: {vv if len(str(vv)) < 160 else str(vv)[:160] + '...'}")
            else:
                print(f"    {k}: {vv}")
else:
    print("  models 为列表：", mdl)

# ---------------------------------------------------------------- 名次
print()
print("=" * 92)
print("同一条件格内的模型名次（按表 8 / 表 9 的 Δ 升序，Δ 越小越好）")
print("=" * 92)
A = {"随机森林": 23.26, "多层感知机": 32.09, "岭回归": 42.51, "有序回归": 79.09,
     "核 SVR": 84.89, "融合模型": 98.95, "TextCNN": 292.74, "BERT": 306.90}
D = {"核 SVR": 13.69, "岭回归": 19.12, "融合模型": 22.27, "多层感知机": 26.45,
     "随机森林": 42.33, "有序回归": 47.86, "TextCNN": 66.74, "BERT": 76.11}
ra = {k: i + 1 for i, (k, _) in enumerate(sorted(A.items(), key=lambda x: x[1]))}
rd = {k: i + 1 for i, (k, _) in enumerate(sorted(D.items(), key=lambda x: x[1]))}
print(f"  {'模型':<12s}{'A格Δ':>9s}{'A名次':>7s}{'D格Δ':>9s}{'D名次':>7s}{'变化':>6s}")
for k in A:
    print(f"  {k:<12s}{A[k]:>9.2f}{ra[k]:>7d}{D[k]:>9.2f}{rd[k]:>7d}{rd[k]-ra[k]:>+6d}")

srt_a = sorted(A.values())
srt_d = sorted(D.values())
print()
print(f"  A 格第 1–6 名跨度：{srt_a[5] - srt_a[0]:.2f} 分")
print(f"  D 格第 1–6 名跨度：{srt_d[5] - srt_d[0]:.2f} 分")
print(f"  八模型极差：A 格 {srt_a[-1] - srt_a[0]:.2f} 分 → D 格 {srt_d[-1] - srt_d[0]:.2f} 分")
print(f"  两格名次的 Spearman：",
      f"{pd.Series(list(ra.values())).corr(pd.Series(list(rd.values())), method='spearman'):.3f}")

# 名次差的自助检验（是否需要"不是随机波动"这种断言）
rng = np.random.default_rng(42)
a_arr, d_arr = np.array(list(A.values())), np.array(list(D.values()))
diffs = []
for _ in range(10000):
    ia = rng.permutation(8)
    idx = rng.permutation(8)
    ra_b = pd.Series(a_arr[ia]).rank().values
    rd_b = pd.Series(d_arr[idx]).rank().values
    diffs.append(abs(ra_b - rd_b).mean())
obs = np.mean([abs(ra[k] - rd[k]) for k in A])
print(f"  平均名次变化的观测值 {obs:.3f}；独立重排下的分布 "
      f"均值 {np.mean(diffs):.3f}，95 分位 {np.percentile(diffs, 95):.3f}")
print(f"  → 观测值{'未超过' if obs <= np.percentile(diffs, 95) else '超过'}独立重排的 95 分位")