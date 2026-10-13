"""补做大纲承诺、但此前未执行的"弱基线自曝"：只用来源特征（题号+比赛等级+比赛编号）预测难度。

背景
----
大纲三处承诺了这条基线：
  - "须单独成组，并在四·（四）做 provenance 消融 + 弱基线自曝"
  - "**弱基线自曝**：只用'题号 + Div 类型'两个特征能到多少 macro-F1？若该基线显著高于随机，
     则所有深度模型的叙述都站不住"
  - 进度表："provenance 消融（结构+文本两侧）+ 弱基线"

但 `runs_ablation.csv` 里只有 provenance 与 template 两类变体，从未跑过弱基线。
本脚本补上它：**成本很低**（LightGBM + 2~3 个特征，单次训练秒级），
比在正文里删掉承诺更好——这条基线正是"捷径可用性"最直接的证据。

口径
----
- 条件 D（时间划分 + 统计特征置空）＝最贴近新题的场景；
- 只用来源特征（problem_index / div_type / contest_id），不看题面、不看标签；
- 与 provenance 消融同格同种子，便于并列。

用法
----
    python scripts/run_weak_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C      # noqa: E402
from task2difficulty import dataset as D      # noqa: E402
from task2difficulty import experiments as E  # noqa: E402

OUT = C.RESULTS_DIR / "runs_weak_baseline.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    issues = C.validate()
    if issues:
        for i in issues:
            log(f"[口径自检失败] {i}")
        return 1

    ds = D.load_dataset(C.DATASET_PATH)
    si = D.load_split_index(C.SPLIT_INDEX_PATH)
    cond = C.CONDITIONS["D"]
    log(f"条件 {cond.key}：{cond.positioning}")
    log(f"只用来源特征：{'、'.join(C.PROVENANCE_FEATURES)}")

    # keep_groups 只保留来源特征组（with_feature_condition 会据此裁剪 X）。
    # ⚠ 必须同时传 `drop_provenance=False`：`with_feature_condition` 先执行
    # drop_provenance（默认 True，会先把来源特征删掉），再执行 keep_groups；
    # 若只用 keep_groups=("provenance",)，来源列早已被删，结果是**0 特征**
    # （实测报 "Found array with 0 feature(s) (shape=(7284, 0))"）。
    pkg = E.build_data_package(ds, si, cond, need_tfidf=False,
                               drop_provenance=False,
                               keep_groups=("provenance",))
    n_feat = pkg.ds_train.X.shape[1]
    log(f"  特征维度 {n_feat}（应为 {len(C.PROVENANCE_FEATURES)}）")
    if n_feat != len(C.PROVENANCE_FEATURES):
        log(f"  [warn] 特征数与来源特征数不符，实际列：{list(pkg.ds_train.X.columns)}")

    rows: list[dict] = []
    for model in ("lightgbm", "random_forest", "ridge"):
        try:
            params = E.load_hyperparams(model)
        except FileNotFoundError:
            params = {}
        for seed in C.SEEDS:
            try:
                res = E.run_tabular(model, pkg, params=params, seed=seed,
                                    tag_columns=[], allow_test_repeat=True)
            except Exception as ex:
                log(f"  {model} seed={seed} 失败：{type(ex).__name__}: {ex}")
                continue
            m = res.test_metrics
            row = res.meta_row()
            row.update({"stage": "ablation_weak_baseline", "variant": "provenance_only",
                        "condition": "D"})
            rows.append(row)
            log(f"  {model:14s} seed={seed:<9d} test MAE={m['mae']:7.2f} "
                f"±200={m['hit200']:.3f} macroF1={m['macro_f1']:.3f} "
                f"QWK={m['qwk']:.3f} ρ={m['spearman']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    log(f"-> {OUT.name}（{len(df)} 行）")
    if df.empty:
        log("✗ 没有任何成功记录；请检查上面的失败原因")
        return 1

    # 与 provenance 消融的 with_provenance（也含来源特征，但同时含全部其他特征）对照
    abl = pd.read_csv(C.RESULTS_DIR / "runs_ablation.csv", encoding="utf-8-sig")
    wp = abl[abl["variant"] == "with_provenance"]
    print()
    log("=" * 74)
    log("弱基线 vs 含全部特征（同条件 D）")
    for model in ("lightgbm", "random_forest", "ridge"):
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        w = wp[wp["model"] == model]
        if w.empty:
            log(f"  {model:14s} 弱基线 MAE={sub['test_mae'].mean():7.2f}"
                f"   （无同模型全特征对照）")
            continue
        log(f"  {model:14s} 弱基线 MAE={sub['test_mae'].mean():7.2f}（ρ={sub['test_spearman'].mean():.3f}）"
            f"   全特征 MAE={w['test_mae'].mean():7.2f}（ρ={w['test_spearman'].mean():.3f}）"
            f"   差 {sub['test_mae'].mean() - w['test_mae'].mean():+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
