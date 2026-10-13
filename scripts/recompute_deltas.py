"""由逐样本预测重算表 5 的 Δ、核查 MLP 在 C 格的方差异常、并给出 RMSE/R²。

回应审稿意见②
--------------
5.1 表 5 的舍入差：现表由**表 3 的已舍入值相减**得到，产生 9 处 ±0.01 偏差。
    正确做法是用**全精度**的逐样本 MAE 相减，最后一次舍入。
5.2 B9 三个 Δ 的隐含标准误差异：需要逐题差值的标准差。
问题 D  表 3 的 MLP 在 C 格为 392.57 ± 99.32，标准差异常，全文无提示。

方法
----
preds/*.npz 保存了每个 (模型, 条件, 种子) 的 test_pred 与 y_test，以及
test_problem_key（题目键）。同划分的条件格（A/B 为随机划分、C/D 为时间
划分）测试集相同，可逐样本配对；跨划分的对比（B−D、A−C）则只能比均值。

⚠ 本脚本只读，不写稿；输出供人工确认后再改表。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "results" / "preds"

MODELS = ["ridge", "svr", "random_forest", "lightgbm", "ordinal_logit",
          "tag_mean", "mlp", "textcnn", "bert", "fusion"]
LABEL = {
    "ridge": "岭回归", "svr": "SVR", "random_forest": "随机森林",
    "lightgbm": "LightGBM", "ordinal_logit": "有序回归",
    "tag_mean": "标签均值基线", "mlp": "MLP", "textcnn": "TextCNN",
    "bert": "BERT", "fusion": "融合模型",
}
#: 划分方式：随机划分的格测试集相同，时间划分的格测试集相同
SPLIT = {"A": "random", "B": "random", "C": "time", "D": "time"}


def load(cond: str, model: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """返回 {seed: (keys, y, pred)}。"""
    out = {}
    for p in PRED.glob(f"{cond}_{model}_seed*.npz"):
        m = re.search(r"_seed(\d+)", p.name)
        if not m:
            continue
        z = np.load(p, allow_pickle=True)
        out[int(m.group(1))] = (z["test_problem_key"], z["y_test"], z["test_pred"])
    return out


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    data: dict[tuple[str, str], dict[int, tuple]] = {}
    for cond in "ABCD":
        for mod in MODELS:
            d = load(cond, mod)
            if d:
                data[(cond, mod)] = d

    # ---------------- 1. 全精度跨种子均值 MAE ----------------
    #: (条件, 模型) -> (五种子均值 MAE, 各种子 MAE 列表)
    agg: dict[tuple[str, str], tuple[float, list[float]]] = {}
    for (cond, mod), d in data.items():
        per = [mae(y, p) for _, y, p in d.values()]
        agg[(cond, mod)] = (float(np.mean(per)), per)

    print("=" * 92)
    print("1. 全精度 MAE（用于重算 Δ，消除表 3 舍入传播）")
    print("=" * 92)
    print(f"{'模型':<14s}" + "".join(f"{c:>12s}" for c in "ABCD"))
    for mod in MODELS:
        row = f"{LABEL[mod]:<14s}"
        for c in "ABCD":
            v = agg.get((c, mod))
            row += f"{v[0]:>12.4f}" if v else f"{'—':>12s}"
        print(row)

    # 与表 3 已发布的舍入值比较
    print("\n  与表 3 现有舍入值的差（>0.005 即说明舍入方向不一致）：")
    pub = {}
    for line in (ROOT / "outputs/results/table3_factors.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        name = parts[0]
        for c, cell in zip("ABCD", parts[1:5]):
            m = re.match(r"(-?\d+(?:\.\d+)?)", cell)
            if m:
                pub[(c, name)] = float(m.group(1))
    inv = {v: k for k, v in LABEL.items()}
    for (c, mod), (v, _) in sorted(agg.items()):
        nm = LABEL[mod]
        if (c, nm) in pub:
            d = v - pub[(c, nm)]
            if abs(d) > 0.005:
                print(f"    {nm:<12s} {c} 格 全精度 {v:.4f} vs 表 3 {pub[(c,nm)]:.2f}"
                      f"  → 差 {d:+.4f}")

    # ---------------- 2. 重算 Δ（全精度相减，最后舍入） ----------------
    DIFFS = [("A-B", "A", "B"), ("B-D", "B", "D"), ("C-D", "C", "D"),
             ("A-C", "A", "C")]
    print()
    print("=" * 92)
    print("2. 由全精度均值重算的 Δ（对照表 5 现值）")
    print("=" * 92)
    cur = {}
    for line in (ROOT / "outputs/results/table5_effects.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 6:
            cur[parts[0]] = parts[1:6]
    hdr = f"{'模型':<14s}" + "".join(f"{k:>13s}" for k, _, _ in DIFFS)
    print(hdr)
    fixed: dict[str, dict[str, float]] = defaultdict(dict)
    for mod in MODELS:
        nm = LABEL[mod]
        row = f"{nm:<14s}"
        for key, l, r in DIFFS:
            a, b = agg.get((l, mod)), agg.get((r, mod))
            if not a or not b:
                row += f"{'—':>13s}"
                continue
            d = a[0] - b[0]
            fixed[nm][key] = d
            row += f"{d:>13.4f}"
        print(row)

    print("\n  与表 5 现值的差（±0.01 级即为审稿人指出的舍入差）：")
    order = ["A−B", "B−D", "B′−D", "C−D", "A−C"]
    n_bad = 0
    for mod in MODELS:
        nm = LABEL[mod]
        if nm not in cur:
            continue
        for key, l, r in DIFFS:
            label = key.replace("-", "−")
            i = order.index(label) if label in order else None
            if i is None:
                continue
            try:
                oldv = float(cur[nm][i])
            except ValueError:
                continue
            newv = fixed[nm].get(key)
            if newv is None:
                continue
            if abs(round(newv, 2) - round(oldv, 2)) >= 0.005:
                n_bad += 1
                print(f"    {nm:<12s} {label:<6s} 表 5 {oldv:+.2f}"
                      f"  → 全精度 {newv:+.4f}（取两位 {round(newv,2):+.2f}）")
    print(f"\n  需要更正的格子：{n_bad} 处")

    # ---------------- 3. MLP 在 C 格的方差异常 ----------------
    print()
    print("=" * 92)
    print("3. MLP 在 C 格的方差异常核查（审稿意见 D）")
    print("=" * 92)
    for c in "ABCD":
        v = agg.get((c, "mlp"))
        if not v:
            continue
        mean, per = v
        print(f"  {c} 格：均值 {mean:8.3f}  标准差(ddof=1) {np.std(per, ddof=1):7.3f}"
              f"   各种子 {[round(x, 2) for x in per]}")
    print()
    for c in "ABCD":
        v = agg.get((c, "lightgbm"))
        if v:
            mean, per = v
            print(f"  [对照] LightGBM {c} 格：均值 {mean:8.3f}  "
                  f"标准差 {np.std(per, ddof=1):7.3f}")

    # ---------------- 4. RMSE 与 R²（审稿意见：A5 只给一条理由） ----------------
    print()
    print("=" * 92)
    print("4. 各格 RMSE 与 R²（用于回应「为何不报 RMSE/R²」）")
    print("=" * 92)
    print(f"{'模型':<14s}" + "".join(f"{c+'格 MAE':>11s}{c+'格 RMSE':>11s}" for c in "AD"))
    for mod in MODELS:
        row = f"{LABEL[mod]:<14s}"
        for c in "AD":
            d = data.get((c, mod))
            if not d:
                row += f"{'—':>22s}"
                continue
            ms = [mae(y, p) for _, y, p in d.values()]
            rs = [float(np.sqrt(np.mean((y - p) ** 2))) for _, y, p in d.values()]
            row += f"{np.mean(ms):>11.2f}{np.mean(rs):>11.2f}"
        print(row)

    print("\n  各格 R²（跨种子均值，逐样本预测）：")
    for c in "ABCD":
        d = data.get((c, "lightgbm"))
        if not d:
            continue
        r2s = []
        for _, y, p in d.values():
            ss_res = float(np.sum((y - p) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2s.append(1 - ss_res / ss_tot)
        print(f"    LightGBM {c} 格 R² = {np.mean(r2s):.4f}")

    # ---------------- 存盘 ----------------
    out = {
        "full_precision_mae": {f"{c}|{m}": agg[(c, m)][0] for c, m in agg},
        "recomputed_deltas": {k: v for k, v in fixed.items()},
        "mlp_seed_mae": {c: agg[(c, "mlp")][1] for c in "ABCD" if (c, "mlp") in agg},
    }
    dst = ROOT / "outputs" / "results" / "recomputed_deltas.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()