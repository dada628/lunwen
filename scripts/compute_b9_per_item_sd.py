"""算 B9 三个 Δ 的逐题差值标准差，解释隐含 SE 的差异（审稿意见② 5.2）。

审稿人的疑问
------------
从表 11 的区间反推标准误（区间宽 ÷ 3.92）：qwen 25.9、DeepSeek 30.1、mimo 38.5。
三个模型跑在同一批 200 道题上、配对单位相同，为什么 mimo 的 SE 大 49%？

要回答它，必须看**逐题差值** d_i = |pred_i − y_i| − |lgb_i − y_i| 的离散度，
而不是看 MAE 本身。本脚本把它算出来。
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
LLM = ROOT / "outputs" / "llm"
PRED = RES / "preds"


def load_ratings(name: str) -> dict[str, float]:
    """取 baseline 变体、repeat=1 的逐题评分。"""
    out: dict[str, float] = {}
    for line in (LLM / f"ratings_{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("variant") == "baseline" and d.get("repeat", 1) == 1:
            out[str(d["item_id"])] = float(d["rating"])
    return out


def main() -> None:
    key = pd.read_csv(LLM / "items_key.csv", dtype={"item_id": str})
    id2pk = dict(zip(key["item_id"], key["problem_key"]))
    id2y = dict(zip(key["item_id"], key["rating"]))
    new_ids = set(key.loc[key["is_new"] == 1, "item_id"])

    # 训练侧：D 格 LightGBM 的五种子预测，按 problem_key 对齐后取逐题均值
    per_item_pred: dict[str, list[float]] = {}
    for p in sorted(PRED.glob("D_lightgbm_seed*.npz")):
        z = np.load(p, allow_pickle=True)
        for k, v in zip(z["test_problem_key"], z["test_pred"]):
            per_item_pred.setdefault(str(k), []).append(float(v))
    lgb = {k: float(np.mean(v)) for k, v in per_item_pred.items()}

    llms = {n: load_ratings(n) for n in
            ["qwen3.8-flash", "DeepSeek-V4.1-Flash", "mimo-v2.6-flash"]}

    # 200 道新题 ∩ LLM 有评分 ∩ 训练侧有预测
    ids = [i for i in sorted(new_ids)
           if all(i in d for d in llms.values()) and id2pk[i] in lgb]
    print(f"配对样本：{len(ids)} 题（新题 ∩ 三家均有评分 ∩ D 格有预测）")

    y = np.array([id2y[i] for i in ids])
    tr = np.array([lgb[id2pk[i]] for i in ids])
    d_tr = np.abs(tr - y)

    print()
    print("=" * 88)
    print("逐题差值 d = |LLM 预测 − 真值| − |LightGBM 预测 − 真值|")
    print("=" * 88)
    print(f"  {'模型':<22s}{'MAE':>8s}{'Δ均值':>9s}{'d 的 SD':>10s}"
          f"{'理论 SE':>10s}{'区间反推':>10s}")
    ci = {"qwen3.8-flash": (-76.478, 25.093),
          "DeepSeek-V4.1-Flash": (-72.105, 46.031),
          "mimo-v2.6-flash": (18.618, 169.69)}
    rows = []
    for name, d in llms.items():
        p = np.array([d[i] for i in ids])
        dd = np.abs(p - y) - d_tr
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        lo, hi = ci[name]
        rows.append((name, np.abs(p - y).mean(), dd.mean(), dd.std(ddof=1), se,
                     (hi - lo) / 3.92))
        print(f"  {name:<22s}{np.abs(p-y).mean():>8.1f}{dd.mean():>9.2f}"
              f"{dd.std(ddof=1):>10.1f}{se:>10.2f}{(hi-lo)/3.92:>10.2f}")

    print()
    print("  LightGBM 侧逐题绝对误差的 SD：", f"{d_tr.std(ddof=1):.1f}")
    for name, d in llms.items():
        p = np.array([d[i] for i in ids])
        print(f"  {name:<22s} 逐题绝对误差 SD = {np.abs(p-y).std(ddof=1):.1f}，"
              f"平均偏差 = {(p-y).mean():+.1f}")

    out = {r[0]: {"mae": r[1], "delta_mean": r[2], "delta_sd": r[3],
                  "se_theory": r[4], "se_from_ci": r[5]} for r in rows}
    (RES / "b9_per_item_sd.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {(RES/'b9_per_item_sd.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()