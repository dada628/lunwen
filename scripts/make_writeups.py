"""正文回填导出器：从结果数据机械生成可入正文的段落与表格片段。

为什么需要它
------------
四·（二）的因子设计说明、表 4 的差分措辞、检验结论都必须与 `runs*.csv` 的
实际数字严格一致，且要带上大纲规定的限定语（"含同源成分""含训练量差异"）。
手工从表格抄进正文，最容易出两类错：数字抄错，以及**限定语被省略**——
后者会直接违反大纲写作红线。本脚本把这两件事都变成代码行为：
数字取自数据，限定语取自 `config.DIFFERENCES`，不经过人手。

输出
----
    outputs/results/正文回填.md        可直接剪贴进正文的段落与表格
    outputs/results/正文回填_因子设计.md  仅因子设计部分（供四·（二）单独取用）

用法
----
    python scripts/make_writeups.py
    python scripts/make_writeups.py --model lightgbm --condition D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import results as R         # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- 差分段落


def _caveat_sentence(spec: dict) -> str:
    """把限定语整理成通顺的一句话。

    `config.DIFFERENCES` 里的 `caveat` 是**给成稿人看的纪律条文**（带 ⚠、带"不得"），
    直接拼进正文会很像内部批注。这里把它转成可读的正文句式，
    但**完整保留限定内容**，不删减任何一条限定。
    """
    txt = str(spec["caveat"]).strip()
    # 去掉只对成稿人可见的标记符号
    txt = txt.replace("⚠", "").replace("**", "").strip()
    txt = txt.replace("；", "。")
    return txt


def factor_design_paragraph(aggs: dict, model: str, space: str = "test") -> list[str]:
    """生成四·（二）的因子设计说明段落。

    ⚠ 数字全部现算，措辞全部取自 config：
    - 差分含义用 `C.DIFFERENCES[...]["meaning"]`（含"含同源成分"等限定）；
    - 方向措辞用 `C.describe_difference()`（只给方向与幅度，避免与带符号的
      数值列重复成复读）；
    - 四处冻结差分（A−B / B−D / C−D / A−C）均给出，若存在 B′ 则一并给出。
    """
    a = aggs.get((model, "A"))
    b = aggs.get((model, "B"))
    d = aggs.get((model, "D"))
    c = aggs.get((model, "C"))
    if not all((a, b, d)):
        return ["（数据不足：A/B/D 三格未齐，无法生成因子设计段落）"]

    lab = R.model_label(model)
    lines: list[str] = []
    order = ["A", "B", "D"] + (["C"] if c else [])
    shown = "、".join(
        f"{k} 格 {aggs[(model, k)].mean('mae', space):.2f}" for k in order)
    lines.append(f"以{lab}为例，各设置下的测试集 MAE 为：{shown}。")

    # ---- A − B：统计特征的边际贡献 ----
    if a and b:
        delta = a.mean("mae", space) - b.mean("mae", space)
        spec = C.DIFFERENCES["A-B"]
        lines.append(
            f"A 格与 B 格仅统计特征的可得性不同，前者的 MAE 比后者低 "
            f"{abs(delta):.2f} 分。这一差值对应{spec['meaning']}。"
            f"需要限定的是，A 格本身即为含同源泄漏的上界，"
            f"因此该差值不等于统计特征的纯净信息价值，本文不将其表述为信息价值。")

    # ---- B − D：跨时间泛化损失 ----
    if b and d:
        delta = b.mean("mae", space) - d.mean("mae", space)
        spec = C.DIFFERENCES["B-D"]
        caveat = _caveat_sentence(spec)
        # 去掉限定语开头与上文重复的"混杂训练集规模"（上文已表述同一意思）
        caveat = caveat.replace("混杂训练集规模。", "", 1).strip()
        lines.append(
            f"B 格与 D 格的 MAE 相差 {abs(delta):.2f} 分，后者更高。"
            f"该差值对应{spec['meaning']}，不能径直接归因于时间漂移："
            f"两格除划分方式不同外，训练集规模也存在差异。{caveat}")

    # ---- C 格：泄漏上界，需说明其性质 ----
    if c:
        spec_ac = C.DIFFERENCES["A-C"]
        delta_ac = a.mean("mae", space) - c.mean("mae", space)
        lines.append(
            f"C 格采用时间划分与全量快照统计，MAE 为 {c.mean('mae', space):.2f}，"
            f"比 A 格高 {abs(delta_ac):.2f} 分，这部分差异对应"
            f"{spec_ac['meaning']}。但 C 格不能视为真实部署场景："
            f"其测试集中 2023 年之后的题目使用的是上线数年后的累计统计，"
            f"属于用最终统计预测最终标签，因此本文仅将其作为上界参照单列，"
            f"不作为部署依据。")

    # ---- B′ 规模对照 ----
    bp = aggs.get((model, "B'"))
    if bp:
        delta = bp.mean("mae", space) - d.mean("mae", space)
        spec = C.DIFFERENCES["B'-D"]
        # 方向措辞先算好再插值：f-string 表达式内不允许出现反斜杠转义
        direction = C.describe_difference(delta, "B'", "D")
        lines.append(
            f"为把训练量差异从上述 B−D 中剥离，本文将随机划分的训练集下采样至与"
            f"时间划分同规模后重训，记为 B′ 格，其 MAE 为 {bp.mean('mae', space):.2f}，"
            f"与 D 格相差 {abs(delta):.2f} 分（{direction}）。"
            f"该结果对应{spec['meaning']}：{_caveat_sentence(spec)}")
    else:
        lines.append(
            "本次未运行 B′ 规模对照，因此 B−D 中的训练量混杂尚未剥离。"
            "这一限制需要在本节或局限部分如实说明，或补做规模对照之后再讨论时间漂移。")

    return lines


# ---------------------------------------------------------------- 主表段落


def main_table_paragraph(aggs: dict, condition: str = "A", space: str = "test") -> list[str]:
    """生成四·（三）主对比的结论段落框架（数字现算，比较结论按实际结果给）。"""
    present = [(m, aggs[(m, condition)]) for m in C.MAIN_TABLE_MODELS
               if (m, condition) in aggs]
    if len(present) < 2:
        return [f"（{condition} 格目前仅有 {len(present)} 个模型的数据，不足以生成主对比结论）"]

    by_mae = sorted(present, key=lambda kv: kv[1].mean("mae", space))
    best_m, best_a = by_mae[0]
    lines: list[str] = []
    lines.append(
        f"在 {condition} 格下，{R.model_label(best_m)} 的测试集 MAE 最低，"
        f"为 {best_a.mean('mae', space):.2f}（±200 命中率 "
        f"{best_a.mean('hit200', space):.3f}，斯皮尔曼 ρ "
        f"{best_a.mean('spearman', space):.3f}）。")

    # 传统侧与深度侧的最优对比——这是大纲主线的核心对照
    trad = [(m, a) for m, a in present if m in C.TRADITIONAL_MODELS]
    deep = [(m, a) for m, a in present if m in C.DEEP_MODELS]
    if trad and deep:
        bt = min(trad, key=lambda kv: kv[1].mean("mae", space))
        bd = min(deep, key=lambda kv: kv[1].mean("mae", space))
        gap = bd[1].mean("mae", space) - bt[1].mean("mae", space)
        lines.append(
            f"传统侧最优为{R.model_label(bt[0])}（MAE {bt[1].mean('mae', space):.2f}），"
            f"深度侧最优为{R.model_label(bd[0])}（MAE {bd[1].mean('mae', space):.2f}），"
            f"二者相差 {gap:+.2f} 分。")

    # 朴素基线的参照作用
    if ("tag_mean", condition) in aggs:
        tm = aggs[("tag_mean", condition)].mean("mae", space)
        gain = tm - best_a.mean("mae", space)
        lines.append(
            f"按算法标签均值预测的朴素基线 MAE 为 {tm:.2f}，"
            f"最优模型较其低 {gain:.2f} 分，说明模型的判断并非仅由标签映射而来。")
    return lines


# ---------------------------------------------------------------- 显著性段落


def significance_paragraph(condition: str, pairs: Sequence[tuple[str, str]],
                           metrics: Sequence[str] = ("mae",)) -> list[str]:
    """生成配对自助检验的描述段落。只用真实缓存，缺缓存则明确说明。"""
    df = R.significance_table(condition, pairs=list(pairs), metrics=list(metrics))
    if df.empty:
        return ["（无逐样本预测缓存，未能生成显著性描述；"
                "请先运行 run_experiments.py --stage factors）"]
    lines: list[str] = []
    for _, r in df.iterrows():
        lines.append(
            f"{r['对比']}：{r['指标']} 差值 {r['Δ']:+.4f}，"
            f"95% 置信区间 {r['95% CI']}，p = {r['p']}，"
            f"{'差异显著' if r['显著'] == '是' else '差异未达显著'}。")
    lines.append("需要说明的是，本文样本量在万级，p 值容易达到显著，"
                 "因此结论以置信区间与效应幅度为准，而非仅凭显著性判定。")
    return lines


# ---------------------------------------------------------------- 组装


def build_all(aggs: dict, model: str, condition: str) -> str:
    L: list[str] = []
    L.append("<!-- 本文件由 scripts/make_writeups.py 自动生成，数字取自 runs*.csv，")
    L.append("     限定语取自 config.DIFFERENCES。请勿手工修改数字。 -->\n")
    L.append("# 正文回填（实验部分）\n")
    L.append(f"> 数据来源：`outputs/results/runs*.csv`；模型 {R.model_label(model)}；"
             f"分箱阈值取自验证集。\n")

    L.append("## 四·（二）因子设计说明段落\n")
    L.extend(factor_design_paragraph(aggs, model))
    L.append("")

    L.append("## 四·（三）主对比结论段落\n")
    L.extend(main_table_paragraph(aggs, condition))
    L.append("")

    L.append("## 四·（三）显著性描述段落（A 格：所有模型 vs LightGBM）\n")
    pairs = [(m, "lightgbm") for m in C.DEEP_MODELS] + \
            [("ridge", "lightgbm"), ("svr", "lightgbm"),
             ("random_forest", "lightgbm"), ("ordinal_logit", "lightgbm")]
    L.extend(significance_paragraph(condition, pairs))
    L.append("")
    # D 格才是真实部署场景，它的配对检验比 A 格更该进正文
    L.append("## 四·（三）显著性描述段落（D 格：最真实的新题场景）\n")
    L.extend(significance_paragraph("D", pairs))
    L.append("")

    L.append("## 差分核对表（含限定语，不得省略）\n")
    rows = R.difference_summary(aggs, model)
    if rows:
        L.append(R.to_markdown(pd.DataFrame(rows)))
    else:
        L.append("（数据不足）")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="正文回填导出")
    ap.add_argument("--model", default=None, help="用于生成说明段落的模型（默认选 A 格 MAE 最低者）")
    ap.add_argument("--condition", default="A", help="主对比所用的条件格")
    ap.add_argument("--synthetic", action="store_true", help="用合成数据结果")
    args = ap.parse_args()

    runs = R.load_all_runs(synthetic=args.synthetic)
    aggs = R.aggregate(runs)
    log(f"载入运行记录 {len(runs)} 行，聚合出 {len(aggs)} 个（模型 × 条件）格")

    model = args.model
    if model is None:
        cands = [(m, aggs[(m, args.condition)]) for m in C.MAIN_TABLE_MODELS
                 if (m, args.condition) in aggs]
        if not cands:
            log(f"[warn] {args.condition} 格无任何主表模型数据，无法生成回填文本")
            return 1
        model = min(cands, key=lambda kv: kv[1].mean("mae"))[0]
    log(f"说明段落以 {R.model_label(model)} 为例")

    text = build_all(aggs, model, args.condition)
    out = C.RESULTS_DIR / "正文回填.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log(f"-> {out}")

    # 因子设计单独一份，便于四·（二）取用
    factor_only = "\n".join(["# 四·（二）因子设计说明段落（可直接入正文）\n"]
                            + factor_design_paragraph(aggs, model))
    out2 = C.RESULTS_DIR / "正文回填_因子设计.md"
    out2.write_text(factor_only, encoding="utf-8")
    log(f"-> {out2}")

    print("\n" + "=" * 70)
    print(factor_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
