"""同题同条件对照：把 LLM 与训练侧模型放在**完全相同的题目**上比较。

⚠ 为什么必须单独做这一步
------------------------
`analyze_llm.py` 里"零样本 LLM MAE 371.8 vs 训练侧 D 格最优 442.4"的对比**不公平**：

- LLM 的 500 题抽样是 **200 新题（year ≥ 2023）+ 300 旧题（< 2023）**；
- 而 D 格（时间划分 + 统计置空）的测试集**全部是新题**（≥2023，2967 题）。

旧题对 LLM 明显更容易（实测 DeepSeek 旧题 MAE 318.7、新题 451.5），
因此 LLM 的 371.8 被那 300 道旧题拉低了。拿它去比"纯新题测试集"上的 442.4，
等于让 LLM 打了一场含送分题的比赛。

正确做法：**取 LLM 抽中的那 200 道新题**，从训练侧 D 格的逐样本预测缓存里
把同一批题捞出来，逐题对齐后重算指标。本脚本只做这件事。

用法
----
    python scripts/compare_llm_same_items.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task2difficulty import config as C  # noqa: E402
from task2difficulty import experiments as E  # noqa: E402
from task2difficulty import metrics as M  # noqa: E402

import human_kappa as HK  # noqa: E402

LLM_DIR = ROOT / "outputs" / "llm"
OUT_DIR = C.RESULTS_DIR

TRAINED = ("lightgbm", "fusion", "bert", "mlp", "textcnn", "random_forest",
           "ridge", "svr", "ordinal_logit", "tag_mean")


def llm_predictions(variant: str = "baseline", repeat: int = 1) -> pd.DataFrame:
    frames = []
    for p in sorted(LLM_DIR.glob("ratings_*.jsonl")):
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    sub = df[(df["variant"] == variant) & (df["repeat"] == repeat)]
    return sub.pivot_table(index="item_id", columns="model", values="rating",
                           aggfunc="first").sort_index()


def trained_predictions(condition: str, model: str, seed: int = 42) -> pd.DataFrame | None:
    """从逐样本预测缓存里取训练侧模型在某条件测试集上的预测（带题目键）。"""
    d = E.load_predictions(condition, model, seed)
    if d is None or "test_problem_key" not in d:
        return None
    return pd.DataFrame({
        "problem_key": np.asarray(d["test_problem_key"], dtype=str),
        "pred": np.asarray(d["test_pred"], dtype=np.float64),
        "y": np.asarray(d["y_test"], dtype=np.float64),
    })


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    d = np.abs(pred - y)
    tb, pb = HK.rating_to_bin(y), HK.rating_to_bin(pred)
    return {
        "n": int(len(y)),
        "mae": float(d.mean()),
        "hit200": float((d <= 200).mean()),
        "spearman": float(M.spearman(y, pred)),
        "bin_exact": float((tb == pb).mean()),
        "bin_within1": float((np.abs(tb - pb) <= 1).mean()),
        "bias": float((pred - y).mean()),
    }


def main() -> int:
    key = pd.read_csv(LLM_DIR / "items_key.csv", encoding="utf-8-sig")
    key["item_id"] = key["item_id"].astype(str)
    key["problem_key"] = key["problem_key"].astype(str)
    key["rating"] = key["rating"].astype(float)

    new_keys = key[key["is_new"] == 1]
    old_keys = key[key["is_new"] == 0]
    print("=" * 82)
    print("同题同条件对照：LLM vs 训练侧模型")
    print("=" * 82)
    print(f"LLM 抽样 {len(key)} 题：新题 {len(new_keys)}（≥{C.TEMPORAL_TEST_START_YEAR}）"
          f" / 旧题 {len(old_keys)}")
    print(f"⚠ D 格测试集全部是新题，因此**只能在新题子集上做公平比较**。\n")

    lw = llm_predictions("baseline", 1)
    lw = lw.loc[[i for i in lw.index if i in set(key["item_id"])]]

    rows: list[dict] = []

    # ---------- 新题子集：LLM vs D 格训练模型 ----------
    new_ids = set(new_keys["item_id"])
    print("=" * 82)
    print(f"【A】新题子集（{len(new_ids)} 题）——与 D 格（时间划分 + 统计置空）逐题对齐")
    print("=" * 82)
    d_models: dict[str, pd.DataFrame] = {}
    for m in TRAINED:
        tp = trained_predictions("D", m, seed=42)
        if tp is None:
            continue
        d_models[m] = tp.set_index("problem_key")

    # 找出这 200 道新题里，有多少确实落在 D 格测试集内
    in_d = {m: [i for i in new_keys["problem_key"] if i in tp.index]
            for m, tp in d_models.items()}
    n_common = min((len(v) for v in in_d.values()), default=0)
    print(f"  LLM 新题中落在 D 格测试集内：{n_common} 题（各模型一致）")
    common_keys = [k for k in new_keys["problem_key"] if k in d_models["lightgbm"].index]
    kid2pk = dict(zip(key["item_id"], key["problem_key"]))
    pk2kid = {v: k for k, v in kid2pk.items()}
    common_ids = [pk2kid[k] for k in common_keys if k in pk2kid]
    print(f"  实际用于对比：{len(common_ids)} 题")

    # LLM 侧
    for model in lw.columns:
        pred = lw.loc[common_ids, model].to_numpy(dtype=np.float64)
        truth = new_keys.set_index("item_id").loc[common_ids, "rating"].to_numpy()
        m = evaluate(pred, truth)
        m.update({"side": "零样本 LLM", "model": model})
        rows.append(m)

    # 训练侧
    for m_, tp in d_models.items():
        sub = tp.loc[common_keys]
        mm = evaluate(sub["pred"].to_numpy(), sub["y"].to_numpy())
        mm.update({"side": "训练侧（D 格）", "model": m_})
        rows.append(mm)

    res = pd.DataFrame(rows)[["side", "model", "n", "mae", "hit200", "spearman",
                              "bin_exact", "bin_within1", "bias"]]
    res = res.sort_values("mae")
    show = res.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))
    res.to_csv(OUT_DIR / "llm_vs_trained_same_items.csv", index=False,
               encoding="utf-8-sig")

    # ---------- 旧题子集：LLM vs B 格训练模型（随机划分 + 统计置空） ----------
    old_ids = set(old_keys["item_id"])
    print("\n" + "=" * 82)
    print(f"【B】旧题子集（{len(old_ids)} 题）——与 B 格（随机划分 + 统计置空）逐题对齐")
    print("=" * 82)
    b_models: dict[str, pd.DataFrame] = {}
    for m in TRAINED:
        tp = trained_predictions("B", m, seed=42)
        if tp is not None:
            b_models[m] = tp.set_index("problem_key")
    if b_models:
        b_keys = [k for k in old_keys["problem_key"] if k in b_models["lightgbm"].index]
        b_ids = [pk2kid[k] for k in b_keys if k in pk2kid]
        print(f"  旧题中落在 B 格测试集内：{len(b_ids)} 题")
        rows_b: list[dict] = []
        for model in lw.columns:
            pred = lw.loc[b_ids, model].to_numpy(dtype=np.float64)
            truth = old_keys.set_index("item_id").loc[b_ids, "rating"].to_numpy()
            m = evaluate(pred, truth)
            m.update({"side": "零样本 LLM", "model": model})
            rows_b.append(m)
        for m_, tp in b_models.items():
            sub = tp.loc[b_keys]
            mm = evaluate(sub["pred"].to_numpy(), sub["y"].to_numpy())
            mm.update({"side": "训练侧（B 格）", "model": m_})
            rows_b.append(mm)
        rb = pd.DataFrame(rows_b).sort_values("mae")
        showb = rb.copy()
        for c in showb.columns:
            if showb[c].dtype.kind == "f":
                showb[c] = showb[c].round(3)
        print(showb.to_string(index=False))
        rb.to_csv(OUT_DIR / "llm_vs_trained_old_items.csv", index=False,
                  encoding="utf-8-sig")
        print("\n  ⚠ 旧题上训练侧占优是**必然**的：这些题就在 B 格的训练集里（随机划分），"
              "模型见过它们。此表仅用于说明'LLM 的旧题优势不等于超越训练模型'。")

    # ---------- 结论 ----------
    print("\n" + "=" * 82)
    print("【结论】公平比较（新题子集，同题同条件）")
    print("=" * 82)
    best_llm = res[res["side"] == "零样本 LLM"].nsmallest(1, "mae").iloc[0]
    best_tr = res[res["side"] != "零样本 LLM"].nsmallest(1, "mae").iloc[0]
    print(f"  零样本 LLM 最优：{best_llm['model']} MAE {best_llm['mae']:.1f}"
          f"（±200 {best_llm['hit200']:.3f}，ρ {best_llm['spearman']:.3f}）")
    print(f"  训练侧最优    ：{best_tr['model']} MAE {best_tr['mae']:.1f}"
          f"（±200 {best_tr['hit200']:.3f}，ρ {best_tr['spearman']:.3f}）")
    diff = best_llm["mae"] - best_tr["mae"]
    print(f"  → 差值 {diff:+.1f} 分（负=LLM 更优）")
    print(f"  提示：公平比较与'混入旧题'的口径相差很大，正文必须用本表数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
