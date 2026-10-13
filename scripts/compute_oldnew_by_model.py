"""算出"新旧题分组"四个列表的逐模型对应，避免正文与 B3.1 出现不具名数字。

审稿意见④ N2 指出：正文[81]的
  旧题 MAE 318.7、358.7、353.3；新题 MAE 451.5、560.5、439.5
  旧题偏差 +8.0、+24.0、−136.7；新题偏差 +170.5、+371.5
四个列表**都没写模型名**，且顺序与上一段的引入顺序（mimo、DeepSeek、qwen）不一致。
本脚本从逐题评分直接算出「模型 → 旧/新 MAE 与平均偏差」的确切对应。
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
id2new = dict(zip(key["item_id"], key["is_new"]))

MODELS = ["DeepSeek-V4.1-Flash", "mimo-v2.6-flash", "qwen3.8-flash"]
print("=" * 92)
print("新旧题分组的逐模型对应（baseline、repeat=1、全 500 题）")
print("=" * 92)
print(f"  {'模型':<22s}{'旧题 MAE':>10s}{'新题 MAE':>10s}{'旧题偏差':>10s}"
      f"{'新题偏差':>10s}{'全 500 MAE':>12s}")
out = {}
for m in MODELS:
    rec = {}
    for line in (LLM / f"ratings_{m}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("variant") == "baseline" and d.get("repeat", 1) == 1:
            rec[str(d["item_id"])] = float(d["rating"])
    old = [i for i in rec if not id2new[i]]
    new = [i for i in rec if id2new[i]]
    eo = np.array([abs(rec[i] - id2y[i]) for i in old])
    en = np.array([abs(rec[i] - id2y[i]) for i in new])
    bo = np.array([rec[i] - id2y[i] for i in old])
    bn = np.array([rec[i] - id2y[i] for i in new])
    allmae = np.mean([abs(rec[i] - id2y[i]) for i in rec])
    out[m] = dict(old_mae=eo.mean(), new_mae=en.mean(), old_bias=bo.mean(),
                  new_bias=bn.mean(), mae500=allmae, n_old=len(old), n_new=len(new))
    print(f"  {m:<22s}{eo.mean():>10.1f}{en.mean():>10.1f}{bo.mean():>+10.1f}"
          f"{bn.mean():>+10.1f}{allmae:>12.1f}")

print()
print("核对 500 题口径 = (n_old×旧 + n_new×新)/500：")
for m, v in out.items():
    calc = (v["n_old"] * v["old_mae"] + v["n_new"] * v["new_mae"]) / (v["n_old"] + v["n_new"])
    print(f"  {m:<22s} 复算 {calc:.2f}　直算 {v['mae500']:.2f}　差 {calc-v['mae500']:+.2f}")

print()
print("正文/附录现在报的是：旧题 318.7、358.7、353.3；新题 451.5、560.5、439.5")
print("偏差：旧 +8.0、+24.0、−136.7；新 +170.5、+371.5（另 qwen 新题 +53.5）")