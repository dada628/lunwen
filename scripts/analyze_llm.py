"""四·（五）LLM 零样本实验的分析：新旧题分组 / 多模型一致性 / 与真实 rating 对比 /
采样噪声基线 / 两组消融。

⚠ 口径要点（与大纲【八】写作红线一致）
--------------------------------------
1. **一致性只报"该协议下的可复现程度"**：25 题一块的批处理协议下，同一块内
   存在残余批内锚定，因此 Fleiss' κ / Cohen's κ **不得**写成"独立评分者之间的
   一致性"。三个评分者下 Fleiss' κ 的自由度有限，故同时给 bootstrap 置信区间。
2. **新旧题分组只作时间邻近性证据**：三家均未公布训练截止时间，**不得**写成
   "证实/证伪记忆污染"。
3. **不得出现**"人类标注水平""人类可达上界"。
4. 分档一律用 `human_kappa.rating_to_bin`（左开右闭），与训练模型/表 4 同口径。

用法
----
    python scripts/analyze_llm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task2difficulty import config as C  # noqa: E402
from task2difficulty import metrics as M  # noqa: E402

import human_kappa as HK  # noqa: E402

LLM_DIR = ROOT / "outputs" / "llm"
OUT_DIR = C.RESULTS_DIR

#: 训练侧 D 格模型（条件 D：时间划分 + 统计特征置空）的对照值，取自 runs.csv
D_CONDITION_MODELS = ("lightgbm", "fusion", "bert", "mlp", "textcnn")

RATING_STEP = 100.0


# ---------------------------------------------------------------- 读取


def load_ratings(suffix: str = "") -> pd.DataFrame:
    """读取所有 `ratings_<模型>.jsonl`，合并成长表。

    列：model / item_id / rating / variant / repeat。
    """
    frames = []
    for p in sorted(LLM_DIR.glob(f"ratings_*{suffix}.jsonl")):
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"{LLM_DIR} 下没有 ratings_*.jsonl；请先运行 scripts/run_llm_experiment.py")
    out = pd.concat(frames, ignore_index=True)
    # 从文件名兜底补 model 列（万一日志缺列）
    out["rating"] = out["rating"].astype(float)
    return out


def load_key() -> pd.DataFrame:
    k = pd.read_csv(LLM_DIR / "items_key.csv", encoding="utf-8-sig")
    k["item_id"] = k["item_id"].astype(str)
    k["rating"] = k["rating"].astype(float)
    return k


def wide(df: pd.DataFrame, variant: str = "baseline", repeat: int = 1) -> pd.DataFrame:
    """取某个变体/重复的宽表：index=item_id，列=模型，值=预测 rating。"""
    sub = df[(df["variant"] == variant) & (df["repeat"] == repeat)]
    w = sub.pivot_table(index="item_id", columns="model", values="rating",
                        aggfunc="first")
    return w.sort_index()


# ---------------------------------------------------------------- 指标


def metrics_vs_truth(pred: np.ndarray, truth: np.ndarray,
                     edges=None) -> dict[str, float]:
    """与真实 rating 对比：MAE / ±200 命中率 / Spearman ρ / 分档一致率。

    分档阈值用**训练侧同一套**（`config.DIFFICULTY_BINS`，左开右闭），
    不另行拟合——否则与表 3/表 4 不可比。
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    m = ~(np.isnan(p) | np.isnan(t))
    p, t = p[m], t[m]
    if len(p) == 0:
        return {}
    d = np.abs(p - t)
    tb = HK.rating_to_bin(t)
    pb = HK.rating_to_bin(p)
    return {
        "n": int(len(p)),
        "mae": float(d.mean()),
        "hit200": float((d <= 200).mean()),
        "spearman": float(M.spearman(t, p)),
        "bin_exact": float((tb == pb).mean()),
        "bin_within1": float((np.abs(tb - pb) <= 1).mean()),
        "bias": float((p - t).mean()),          # 正=高估，负=低估
        "mae_low": float(d[t <= 1200].mean()) if (t <= 1200).any() else float("nan"),
        "mae_high": float(d[t > 2400].mean()) if (t > 2400).any() else float("nan"),
    }


def fleiss_with_ci(bin_mat: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Fleiss' κ + bootstrap 95% CI。

    ⚠ 三个评分者下 κ 的抽样波动不小，只报点值会被质疑；故按题目重抽样给 CI。
    """
    k = HK.fleiss_kappa_manual(bin_mat)
    n = bin_mat.shape[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots.append(HK.fleiss_kappa_manual(bin_mat[idx]))
        except Exception:
            continue
    boots = [b for b in boots if np.isfinite(b)]
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots else (float("nan"), float("nan"))
    return {"fleiss_kappa": float(k), "ci_low": lo, "ci_high": hi,
            "n_boot": len(boots)}


def bin_count_matrix(w: pd.DataFrame) -> np.ndarray:
    """宽表 → Fleiss 计数矩阵（n_items × 5 档）。

    只保留**所有模型都给出有效 rating** 的题目，不做任何填补（大纲要求）。
    """
    complete = w.dropna(how="any")
    if complete.empty:
        return np.zeros((0, HK.n_bins()), dtype=np.float64)
    n_bins = HK.n_bins()
    counts = np.zeros((len(complete), n_bins), dtype=np.float64)
    for j, col in enumerate(complete.columns):
        b = HK.rating_to_bin(complete[col].to_numpy()) - 1
        for i, bi in enumerate(b):
            counts[i, int(bi)] += 1
    return counts


def per_model_metrics(df: pd.DataFrame, key: pd.DataFrame,
                      variant: str = "baseline", repeat: int = 1) -> pd.DataFrame:
    sub = df[(df["variant"] == variant) & (df["repeat"] == repeat)]
    j = sub.merge(key[["item_id", "rating", "year", "is_new"]], on="item_id",
                  how="inner", suffixes=("", "_true"))
    rows = []
    for model, g in j.groupby("model"):
        m = metrics_vs_truth(g["rating"].to_numpy(), g["rating_true"].to_numpy())
        m["model"] = model
        # 新旧题分组（仅时间邻近性证据）
        for era, lab in ((1, "new"), (0, "old")):
            gg = g[g["is_new"] == era]
            if len(gg):
                mm = metrics_vs_truth(gg["rating"].to_numpy(),
                                      gg["rating_true"].to_numpy())
                m[f"mae_{lab}"] = mm["mae"]
                m[f"hit200_{lab}"] = mm["hit200"]
                m[f"bias_{lab}"] = mm["bias"]
                m[f"n_{lab}"] = mm["n"]
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="四·（五）LLM 实验分析")
    ap.add_argument("--suffix", default="", help="结果文件后缀（验证用 _dryrun）")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = load_ratings(args.suffix)
    key = load_key()
    models = sorted(df["model"].unique())
    n_batches_done = df.groupby(["model", "variant", "repeat"])["batch_id"].nunique()

    print("=" * 78)
    print("四·（五）LLM 零样本实验分析")
    print("=" * 78)
    print(f"模型 {models}")
    print(f"评级记录 {len(df)} 条（{df['item_id'].nunique()} 题）")
    print(f"各模型各变体完成块数：\n{n_batches_done.to_string()}")

    # ---- 1. 与真实 rating 对比（主实验） ----
    print("\n" + "=" * 78)
    print("【1】与真实 rating 的对比（主实验，repeat=1）")
    print("=" * 78)
    mt = per_model_metrics(df, key, "baseline", 1)
    if not mt.empty:
        cols = ["model", "n", "mae", "hit200", "spearman", "bin_exact", "bin_within1",
                "bias", "mae_low", "mae_high", "n_new", "n_old", "mae_new", "mae_old",
                "bias_new", "bias_old"]
        cols = [c for c in cols if c in mt.columns]
        show = mt[cols].copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].round(3)
        print(show.to_string(index=False))
        mt.to_csv(OUT_DIR / "llm_main_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 2. 新旧题分组（时间邻近性） ----
    if not mt.empty and "mae_new" in mt.columns:
        print("\n【1b】新旧题分组（★ 仅时间邻近性证据，非记忆污染证据）")
        for _, r in mt.iterrows():
            d = r["mae_new"] - r["mae_old"]
            print(f"  {r['model']:<22s} 新题 MAE {r['mae_new']:7.1f}（n={int(r['n_new'])}）  "
                  f"旧题 {r['mae_old']:7.1f}（n={int(r['n_old'])}）  差 {d:+7.1f}  "
                  f"偏差 新 {r['bias_new']:+7.1f} / 旧 {r['bias_old']:+7.1f}")
        print("  ⚠ 三家均未公布训练截止时间，此处**不能**区分判断与回忆。")

    # ---- 3. 多模型一致性 ----
    print("\n" + "=" * 78)
    print("【2】多模型一致性（主实验）")
    print("=" * 78)
    w = wide(df, "baseline", 1)
    complete = w.dropna(how="any")
    print(f"宽表 {w.shape[1]} 个模型 × {len(w)} 题；全部模型都有有效评分的题 "
          f"{len(complete)} 题（有效样本量）")
    if len(complete) >= 10:
        counts = bin_count_matrix(w)
        fl = fleiss_with_ci(counts, n_boot=args.n_boot)
        print(f"  Fleiss' κ = {fl['fleiss_kappa']:.4f}  "
              f"95% CI [{fl['ci_low']:.4f}, {fl['ci_high']:.4f}]（{fl['n_boot']} 次重抽样）")
        # 原始一致率与 ±1 档一致率
        bmat = np.column_stack([HK.rating_to_bin(complete[c].to_numpy())
                                for c in complete.columns])
        exact = float(np.mean([len(set(row)) == 1 for row in bmat]))
        within1 = float(np.mean([(row.max() - row.min()) <= 1 for row in bmat]))
        print(f"  原始一致率（3 模型同档）= {exact:.3f}；±1 档一致率 = {within1:.3f}")
        # 两两 Cohen's κ / QWK
        print("  两两一致（Cohen's κ / 二次加权 κ）：")
        for i in range(bmat.shape[1]):
            for j2 in range(i + 1, bmat.shape[1]):
                a, b = bmat[:, i], bmat[:, j2]
                k = HK.safe_kappa(a, b)
                q = HK.safe_kappa(a, b, weights="quadratic")
                print(f"    {complete.columns[i]:<22s} vs {complete.columns[j2]:<22s} "
                      f"κ={k:.3f}  QWK={q:.3f}")
        json.dump({"fleiss": fl, "exact_agreement": exact, "within1": within1,
                   "n_complete": int(len(complete)), "models": list(complete.columns)},
                  (OUT_DIR / "llm_agreement.json").open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # ---- 4. 采样噪声基线 ----
    print("\n" + "=" * 78)
    print("【3】采样噪声基线（同块重跑两次的内部一致率）")
    print("=" * 78)
    w1, w2 = wide(df, "baseline", 1), wide(df, "baseline", 2)
    common = w1.index.intersection(w2.index)
    noise_rows = []
    for c in w1.columns:
        if c not in w2.columns:
            continue
        a = w1.loc[common, c]
        b = w2.loc[common, c]
        m = a.notna() & b.notna()
        if m.sum() == 0:
            continue
        ba, bb = HK.rating_to_bin(a[m].to_numpy()), HK.rating_to_bin(b[m].to_numpy())
        d = np.abs(a[m].to_numpy() - b[m].to_numpy())
        noise_rows.append({
            "model": c, "n": int(m.sum()),
            "mae_self": float(d.mean()),
            "exact": float((ba == bb).mean()),
            "within1": float((np.abs(ba - bb) <= 1).mean()),
            "same_rating": float((d < 1e-9).mean()),
        })
    if noise_rows:
        nd = pd.DataFrame(noise_rows)
        for c in nd.columns:
            if nd[c].dtype.kind == "f":
                nd[c] = nd[c].round(3)
        print(nd.to_string(index=False))
        nd.to_csv(OUT_DIR / "llm_noise_baseline.csv", index=False, encoding="utf-8-sig")
        print("  ⚠ 这是**同一模型**在同协议下的自我复现下限。跨模型一致率必须")
        print("     明显高于它，才说明多模型投票带来了超出采样波动的信息。")
    else:
        print("  （无 repeat=2 数据；请用 --noise-check 之外的完整运行）")

    # ---- 5. 消融 ----
    print("\n" + "=" * 78)
    print("【4】提示词消融")
    print("=" * 78)
    for variant, lab in (("no_tags", "消融①：不给算法标签"), ("role", "消融②：角色设定")):
        sub = df[df["variant"] == variant]
        if sub.empty:
            print(f"  {lab}: 无数据")
            continue
        j = sub.merge(key[["item_id", "rating"]], on="item_id", how="inner",
                      suffixes=("", "_true"))
        base_ids = set(sub["item_id"])
        base = df[(df["variant"] == "baseline") & (df["repeat"] == 1)
                  & (df["item_id"].isin(base_ids))]
        jb = base.merge(key[["item_id", "rating"]], on="item_id", how="inner",
                        suffixes=("", "_true"))
        print(f"\n  {lab}（同题 {len(base_ids)} 题，与主实验前 8 块同序）")
        for model in sorted(sub["model"].unique()):
            g = j[j["model"] == model]
            gb = jb[jb["model"] == model]
            m1 = metrics_vs_truth(g["rating"].to_numpy(), g["rating_true"].to_numpy())
            m0 = metrics_vs_truth(gb["rating"].to_numpy(), gb["rating_true"].to_numpy())
            if not m1 or not m0:
                continue
            print(f"    {model:<22s} MAE {m1['mae']:7.1f} vs 主实验 {m0['mae']:7.1f}  "
                  f"差 {m1['mae'] - m0['mae']:+7.1f}   "
                  f"ρ {m1['spearman']:.3f} vs {m0['spearman']:.3f}   "
                  f"偏差 {m1['bias']:+7.1f} vs {m0['bias']:+7.1f}")

    # ---- 6. 与训练侧 D 格对照 ----
    print("\n" + "=" * 78)
    print("【5】与训练侧 D 格模型的对照（条件 D：时间划分 + 统计特征置空）")
    print("=" * 78)
    try:
        runs = pd.read_csv(OUT_DIR / "runs.csv", encoding="utf-8-sig")
        d = runs[(runs["condition"] == "D") & (runs["model"].isin(D_CONDITION_MODELS))]
        agg = d.groupby("model")["test_mae"].mean().sort_values()
        print("  训练侧（测试集 MAE，均值）：")
        for m, v in agg.items():
            print(f"    {m:<22s} {v:7.1f}")
        if not mt.empty:
            best = mt.loc[mt["mae"].idxmin()]
            print(f"  零样本 LLM 最优：{best['model']} MAE {best['mae']:.1f}")
            print(f"  → 与训练侧最优（{agg.index[0]} {agg.iloc[0]:.1f}）相差 "
                  f"{best['mae'] - agg.iloc[0]:+.1f} 分")
    except FileNotFoundError:
        print("  （未找到 runs.csv，跳过）")

    print("\n产出：")
    for f in ("llm_main_metrics.csv", "llm_agreement.json", "llm_noise_baseline.csv"):
        p = OUT_DIR / f
        if p.exists():
            print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
