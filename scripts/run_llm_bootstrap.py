"""补做意见③ P0：零样本大模型 vs 训练侧 LightGBM 的配对自助检验（200 道新题子集）。

背景（修改意见③ 问题 1）
-------------------------
全文在四·（五）写明"样本量千级至万级时易给显著结论，故以区间宽度与效应幅度为准"，
并在附录 B2 对训练侧做了完整配对自助检验；但**唯一一处跨范式、样本量最小的
200 题比较上，只有点估计**（439.5 vs 465.1，差 25.6 分），却得出了
"零样本大模型比专门训练的最优模型低约 26 分"的结论。

这是全文证据强度最低、却最吸引眼球的一处。本脚本补做该检验：
  - 配对单位：题目（同一批 200 道新题）
  - 大模型侧：逐题评分（单次零样本）
  - 训练侧：LightGBM 在同一批题上的五种子均值预测（与正文口径一致）
  - 统计量：Δ = MAE(大模型) − MAE(LightGBM) 的配对自助 95% CI（10^4 次）

输出到 outputs/results/llm_vs_trained_bootstrap.json，供正文与摘要引用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LLM = ROOT / "outputs" / "llm"
RES = ROOT / "outputs" / "results"
N_BOOT = 10_000
SEED = 42


def log(m: str) -> None:
    print(m, flush=True)


def load_llm_ratings(model: str) -> dict[str, float]:
    """读某模型的逐题零样本评分（baseline 变体、首个 repeat）。"""
    out: dict[str, float] = {}
    p = LLM / f"ratings_{model}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("variant") not in (None, "baseline"):
            continue
        if d.get("repeat", 1) != 1:
            continue
        r = d.get("rating")
        if r is None:
            continue
        out[str(d["item_id"])] = float(r)
    return out


def main() -> int:
    import pandas as pd

    key = pd.read_csv(LLM / "items_key.csv", encoding="utf-8-sig")
    key["item_id"] = key["item_id"].astype(str)
    # ⚠ 两侧 ID 体系不同，必须经 problem_key 桥接：
    #   LLM 侧 item_id 形如 'K0001'；训练侧 test_problem_key 形如 '1768-A'。
    #   items_key.csv 同时含这两列，是唯一的对应关系来源。
    key["problem_key"] = key["problem_key"].astype(str)
    kk = key.set_index("item_id")
    kp = key.set_index("problem_key")
    new_ids = set(kk.index[kk["is_new"] == 1])
    log(f"答案键：{len(kk)} 题，其中新题 {len(new_ids)}")

    # ---- 训练侧：LightGBM 在 D 格的逐题预测（五种子均值） ----
    preds_dir = RES / "preds"
    files = sorted(preds_dir.glob("D_lightgbm_seed*.npz"))
    if not files:
        log("✗ 找不到 D_lightgbm_seed*.npz")
        return 1
    acc: dict[str, list[float]] = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        # ⚠ 键名是 test_problem_key（不是 test_ids）——形如 '1768-A'，
        # 与 LLM 侧 items_key.csv 的 problem_key 同构（实测确认）。
        ids = [str(x) for x in z["test_problem_key"]]
        pr = np.asarray(z["test_pred"], dtype=float)
        seed = int(z["seed"])
        for i, v in zip(ids, pr):
            acc.setdefault(i, {})[seed] = float(v)
    seeds = sorted({int(np.load(f, allow_pickle=True)["seed"]) for f in files})
    log(f"训练侧：{len(files)} 个种子 {seeds}，{len(acc)} 道题的逐题预测")

    # ---- 大模型侧 ----
    models = ["qwen3.8-flash", "DeepSeek-V4.1-Flash", "mimo-v2.6-flash"]
    llm = {m: load_llm_ratings(m) for m in models}

    # ---- 对齐到"200 道新题 ∩ 训练侧有预测" ----
    # 训练侧按 problem_key 索引；LLM 侧按 item_id。先取新题的 problem_key 集合，
    # 再用它去交集，最后回到 item_id 取评分。
    new_keys = set(kk.loc[list(new_ids), "problem_key"])
    llm_ids_common = set.intersection(*[set(llm[m]) for m in models])
    # kk 是 item_id 索引，这里用列取 problem_key（不能用 kp，它被 set_index 后
    # 同名 DataFrame 已被覆盖，列不再是 problem_key）
    llm_keys = set(key.loc[key["item_id"].isin(llm_ids_common), "problem_key"])
    common_keys = new_keys & set(acc) & llm_keys
    ids = sorted(key.loc[key["problem_key"].isin(common_keys), "item_id"])
    log(f"对齐后可比子集：{len(ids)} 题（新题 ∩ 三方均有预测）")
    if len(ids) < 50:
        log("✗ 可比样本过少，无法做检验")
        return 1

    id2pk = dict(zip(key["item_id"], key["problem_key"]))
    id2y = dict(zip(key["item_id"], key["rating"]))
    y = np.array([float(id2y[i]) for i in ids])
    tr = np.array([float(np.mean(list(acc[id2pk[i]].values()))) for i in ids])

    # ⚠ 两种"种子聚合"估计量必须分清（修改意见① 第 4 点）：
    #   ① 五次运行的 MAE 取平均 → 与正文/官方文件口径一致（表 3 的 465.08）
    #   ② 五次预测先平均再算 MAE → 是"集成预测器"，值略低
    # 正文主表用的是 ①，故此处也以 ① 为准做配对检验（② 仅作对照打印）。
    mae_seedmean = float(np.mean([
        np.mean(np.abs(np.array([acc[id2pk[i]][s] for i in ids]) - y))
        for s in seeds]))
    mae_ens = float(np.mean(np.abs(tr - y)))
    log(f"\n训练侧 LightGBM 在 {len(ids)} 道新题上：")
    log(f"  ① 五种子 MAE 的均值（正文口径）= {mae_seedmean:.2f}")
    log(f"  ② 五种子预测先平均再算 MAE（集成口径）= {mae_ens:.2f}")
    assert abs(mae_seedmean - 465.08) < 0.5, \
        f"与官方文件 465.08 不符（实算 {mae_seedmean:.2f}），请检查种子集合"

    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))

    results = {"n_items": len(ids), "n_boot": N_BOOT, "seed": SEED,
               "target": "Δ = MAE(LLM) − MAE(LightGBM)，正=大模型更差",
               "trained_mae_seedmean": round(mae_seedmean, 3),
               "trained_mae_ensemble": round(mae_ens, 3),
               "models": {}}

    mae_tr = mae_seedmean
    log(f"\n配对检验以口径 ①（{mae_tr:.2f}）为基准")

    # 训练侧：每个种子的逐题绝对残差（配对自助时对题目重采样）
    E_tr_seeds = np.array([[abs(acc[id2pk[i]][s] - y[j]) for j, i in enumerate(ids)]
                           for s in seeds])          # shape (n_seed, n_item)

    for m in models:
        p = np.array([llm[m][i] for i in ids])
        mae_llm = float(np.mean(np.abs(p - y)))
        d_point = mae_llm - mae_tr
        e_llm = np.abs(p - y)
        # 配对自助（向量化）：对题目重采样；训练侧同时对"种子"与"题目"两级重采样，
        # 以反映五种子均值本身的不确定性。
        d_llm_boot = e_llm[idx_boot].mean(axis=1)                # (N_BOOT,)
        s_pick = rng.integers(0, len(seeds), size=(N_BOOT, len(ids)))
        d_tr_boot = E_tr_seeds[s_pick, idx_boot].mean(axis=1)    # (N_BOOT,)
        diffs = d_llm_boot - d_tr_boot
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        cross = bool(lo <= 0 <= hi)
        results["models"][m] = {
            "mae": round(mae_llm, 3),
            "delta_vs_lightgbm": round(d_point, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_bootstrap": round(float(min(p_two, 1.0)), 4),
            "crosses_zero": cross,
        }
        verdict = "区间跨零 → 差异不显著" if cross else "区间不跨零 → 差异显著"
        log(f"  {m:<22s} MAE={mae_llm:7.2f}  Δ={d_point:+7.2f}  "
            f"95%CI=[{lo:+.2f}, {hi:+.2f}]  p={min(p_two, 1.0):.3f}  {verdict}")

    out = RES / "llm_vs_trained_bootstrap.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n-> {out.name}")

    # 结论摘要：供正文使用
    best = min(results["models"].items(), key=lambda kv: kv[1]["mae"])
    log("\n" + "=" * 74)
    log(f"最优零样本模型：{best[0]}（MAE {best[1]['mae']}）")
    if best[1]["crosses_zero"]:
        log("结论：该模型与训练侧 LightGBM 的差距**在 95% 水平上不显著**（区间跨零），")
        log("      正文应写『点估计接近，但差异不显著』，不得写成『优于/达到』。")
    else:
        log(f"结论：差距显著（Δ={best[1]['delta_vs_lightgbm']:+.2f}，"
            f"95%CI=[{best[1]['ci95'][0]:+.2f}, {best[1]['ci95'][1]:+.2f}]）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
