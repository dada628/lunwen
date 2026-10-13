"""从"主实验 baseline 跑了两轮"这一点，量出大模型侧的**运行间 MAE 波动**。

审稿意见④ N1 说："[83] 的'采样噪声基线'目前只报了一致率（0.666），
建议同一批重测数据顺手给出 MAE 层面的波动。"

ratings_*.jsonl 里 baseline 变体有 repeat=1 与 repeat=2 各 500 题，
这正是同一提示词、同一题集的两次独立运行，可直接算：
  - 两次运行的总体 MAE 之差
  - 逐题差值的分布（与 B9 的配对检验同构）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
LLM = ROOT / "outputs" / "llm"

key = pd.read_csv(LLM / "items_key.csv", dtype={"item_id": str})
id2y = dict(zip(key["item_id"], key["rating"]))
new_ids = [i for i in key["item_id"] if dict(zip(key["item_id"], key["is_new"]))[i] == 1]

MODELS = ["qwen3.8-flash", "DeepSeek-V4.1-Flash", "mimo-v2.6-flash"]
rows = []
for m in MODELS:
    r: dict[int, dict[str, float]] = {1: {}, 2: {}}
    for line in (LLM / f"ratings_{m}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("variant") == "baseline":
            r.setdefault(d.get("repeat", 1), {})[str(d["item_id"])] = float(d["rating"])
    common = sorted(set(r[1]) & set(r[2]))
    e1 = np.array([abs(r[1][i] - id2y[i]) for i in common])
    e2 = np.array([abs(r[2][i] - id2y[i]) for i in common])
    d = e1 - e2
    rows.append((m, len(common), e1.mean(), e2.mean(), abs(e1.mean() - e2.mean()),
                 d.std(ddof=1)))

print("=" * 88)
print("大模型侧运行间波动（同一提示词、同一 500 题、两次独立运行）")
print("=" * 88)
print(f"  {'模型':<22s}{'n':>5s}{'第 1 轮 MAE':>12s}{'第 2 轮 MAE':>12s}"
      f"{'差':>8s}{'逐题差 SD':>11s}")
for m, n, a, b, delta, sd in rows:
    print(f"  {m:<22s}{n:>5d}{a:>12.1f}{b:>12.1f}{delta:>8.1f}{sd:>11.1f}")
mx = max(r[4] for r in rows)
print(f"\n  两次运行 MAE 的最大差：{mx:.1f} 分")
print("  → 这是**同子集**的运行间波动，可用于说明 B9 的区间未纳入该项方差；")
print("    而消融基线与主实验的 27.5–70.5 分差源于**子集不同**（K0001–200 含 125 道旧题），")
print("    两者性质不同，不可混为一谈。")

print()
print("=" * 88)
print("子集构成核对（解释消融基线偏低）")
print("=" * 88)
first200 = [f"K{i:04d}" for i in range(1, 201)]
isnew = dict(zip(key["item_id"], key["is_new"]))
n_new = sum(isnew[i] for i in first200)
print(f"  K0001–K0200：新题 {n_new} + 旧题 {200 - n_new}")
print(f"  旧题更易（三家旧题 MAE 318.7/358.7/353.3 vs 新题 451.5/560.5/439.5），")
print(f"  占比 62.5% 的旧题把该子集的整体 MAE 拉低了约 30–70 分。")
for m, n, a, b, delta, sd in rows:
    pass