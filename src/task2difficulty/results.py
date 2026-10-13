"""结果汇总与论文表格生成（大纲四·（三）表 3、四·（二）表 4）。

本模块只做"从原始运行记录到论文表格"的转换，不产生任何新数字。
所有口径（指标集合、差分定义、泄漏标注）都从 `config` 取，避免与正文口径漂移。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M


# ---------------------------------------------------------------- 模型显示名

MODEL_LABELS: dict[str, str] = {
    "ridge": "岭回归",
    "svr": "SVR",
    "random_forest": "随机森林",
    "lightgbm": "LightGBM",
    "ordinal_logit": "有序回归",
    "tag_mean": "标签均值基线",
    "mlp": "MLP",
    "textcnn": "TextCNN",
    "bert": "BERT",
    "fusion": "融合模型",
    "bilstm_attn": "BiLSTM+Attention",
}

#: 主表中的呈现顺序（与传统侧/深度侧的分组一致）
MAIN_TABLE_ORDER: tuple[str, ...] = C.MAIN_TABLE_MODELS


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


# ---------------------------------------------------------------- 载入


def load_runs(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"未找到运行记录 {p}")
    return pd.read_csv(p, encoding="utf-8-sig")


#: 哪些 stage 的记录可以参与"模型 × 条件"聚合。
#:
#: ⚠ 只有 factor 类 stage 可以。`ablation_*` 记录**复用了条件键 D**
#: （with/without provenance、clean/stripped 模板句），`efficiency` 记录复用了同样的条件键，
#: 它们是**不同实验配置**下的同条件结果。若不过滤就 concat 进聚合，会把消融结果混进
#: D 格均值，**静默污染表 4 与 B−D 差分**——而表 4 正是本文核心表。
#: `size_control` 记录用条件键 `B'`（独立键，不会与 B 冲突），故一并放行。
AGGREGATABLE_STAGES: tuple[str, ...] = ("factors", "size_control")


def load_all_runs(results_dir: Path = C.RESULTS_DIR, synthetic: bool = False,
                  aggregatable_only: bool = True) -> pd.DataFrame:
    """合并 factors / bprime / ablation 三个阶段的原始记录。

    Parameters
    ----------
    aggregatable_only:
        默认 True，**剔除消融与数据效率等复用了条件键的记录**，只保留可用于
        "模型 × 条件"均值聚合的 factor 类记录。设 False 可拿到全部记录
        （做消融分析时用），但此时**不可**直接喂给 `aggregate()`。
    """
    suf = "_synthetic" if synthetic else ""
    frames: list[pd.DataFrame] = []
    for name in (f"runs{suf}.csv", f"runs_bprime{suf}.csv", f"runs_ablation{suf}.csv",
                 f"runs_efficiency{suf}.csv"):
        p = results_dir / name
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["source_file"] = name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"{results_dir} 下没有任何 runs*.csv；请先运行 "
            f"scripts/run_experiments.py --stage factors"
        )
    out = pd.concat(frames, ignore_index=True)
    if aggregatable_only and "stage" in out.columns:
        # 只保留 stage ∈ AGGREGATABLE_STAGES 的记录。
        # 消融/数据效率等 stage 复用了条件键，混入会静默污染条件均值。
        stage = out["stage"].astype(str)
        out = out[stage.isin(AGGREGATABLE_STAGES)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------- 聚合


@dataclass
class Agg:
    """一个（模型 × 条件）的多种子聚合结果。"""

    model: str
    condition: str
    n_seeds: int
    stats: dict[str, dict[str, float]]     # metric -> {mean, std}
    is_leaky: bool = False
    in_main_conclusion: bool = True
    n_test: int = 0
    n_train: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def mean(self, metric: str, space: str = "test") -> float:
        return self.stats[f"{space}_{metric}"]["mean"]

    def std(self, metric: str, space: str = "test") -> float:
        return self.stats[f"{space}_{metric}"]["std"]

    def cell(self, metric: str, space: str = "test", digits: int = 3) -> str:
        s = self.stats[f"{space}_{metric}"]
        if s["n_seeds"] <= 1:
            return f"{s['mean']:.{digits}f}"
        return f"{s['mean']:.{digits}f}±{s['std']:.{digits}f}"


def aggregate(runs: pd.DataFrame, space: str = "test") -> dict[tuple[str, str], Agg]:
    """按（模型, 条件）聚合多个种子为均值±标准差。"""
    out: dict[tuple[str, str], Agg] = {}
    if "condition" not in runs.columns:
        return out
    for (model, cond), g in runs.groupby(["model", "condition"], sort=False):
        stats: dict[str, dict[str, float]] = {}
        for m in C.METRICS:
            col = f"{space}_{m}"
            if col not in g.columns:
                continue
            vals = g[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            stats[col] = {
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": float(len(vals)),
            }
        out[(model, cond)] = Agg(
            model=model, condition=cond, n_seeds=int(len(g)),
            stats=stats,
            is_leaky=bool(g["is_leaky"].iloc[0]) if "is_leaky" in g.columns else False,
            in_main_conclusion=(bool(g["in_main_conclusion"].iloc[0])
                                if "in_main_conclusion" in g.columns else True),
            n_test=int(g["n_test"].iloc[0]) if "n_test" in g.columns else 0,
            n_train=int(g["n_train"].iloc[0]) if "n_train" in g.columns else 0,
            extra={k: g[k].iloc[0] for k in g.columns
                   if k not in ("model", "condition") and g[k].nunique() == 1},
        )
    return out


# ---------------------------------------------------------------- 表 3


def main_table(aggs: Mapping[tuple[str, str], Agg], condition: str = "A",
               space: str = "test") -> pd.DataFrame:
    """表 3：模型 × 5 指标的均值±标准差。

    按大纲四·（三）：主表共 10 行（传统侧 6 + 深度侧 4）。

    ⚠ **不丢弃任何主表模型行**：缺少数据的模型保留行、单元格填「—」。
    早期实现是 `if a is None: continue`，会静默把主表缩成 1 行——
    那样"主表 10 行"这条大纲口径就无从核对，也容易误把不完整的表当定稿（实测踩过）。
    """
    rows: list[dict[str, Any]] = []
    for model in MAIN_TABLE_ORDER:
        a = aggs.get((model, condition))
        row: dict[str, Any] = {"模型": model_label(model), "model": model}
        if a is None:
            for m in C.METRICS:
                row[C.METRIC_LABELS[m]] = "—"
            row["训练/测试样本数"] = "—"
        else:
            for m in C.METRICS:
                key = f"{space}_{m}"
                row[C.METRIC_LABELS[m]] = (
                    a.cell(m, space, digits=2 if m == "mae" else 3)
                    if key in a.stats else "—"
                )
            # ⚠ 列名如实反映聚合到的字段：Agg 只聚合了 n_train / n_test（未聚合 n_valid），
            # 因此不能写成"训练/验证/测试"，否则正文会引用一个并不存在的验证集规模。
            row["训练/测试样本数"] = f"{a.n_train}/{a.n_test}"
        rows.append(row)

    df = pd.DataFrame(rows)
    # 有数据的按 MAE 升序排在前，缺数据的保持在末尾（顺序稳定、可复现）
    df["_has_data"] = [0 if (m, condition) in aggs else 1 for m in df["model"]]
    df["_mae_mean"] = [
        aggs[(m, condition)].mean("mae", space) if (m, condition) in aggs else np.nan
        for m in df["model"]
    ]
    df = df.sort_values(["_has_data", "_mae_mean"], na_position="last").reset_index(drop=True)
    return df.drop(columns=["_has_data", "_mae_mean"])


def mark_best(table: pd.DataFrame, metric_label: str, lower_is_better: bool
              ) -> pd.DataFrame:
    """给指定指标列的最优值加粗、次优下划线（该刊图表规范）。"""
    if metric_label not in table.columns:
        return table
    vals = []
    for s in table[metric_label].astype(str):
        try:
            vals.append(float(str(s).split("±")[0]))
        except ValueError:
            vals.append(np.nan)
    arr = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(arr).any():
        return table
    order = np.argsort(arr if lower_is_better else -arr)
    order = [i for i in order if np.isfinite(arr[i])]
    out = table.copy()
    for rank, i in enumerate(order[:2]):
        txt = str(out.at[i, metric_label])
        out.at[i, metric_label] = f"**{txt}**" if rank == 0 else f"_{txt}_"
    return out


# ---------------------------------------------------------------- 表 4


def factor_table(aggs: Mapping[tuple[str, str], Agg],
                 models: Sequence[str] | None = None,
                 space: str = "test") -> pd.DataFrame:
    """表 4：A/B/C/D 并列 + A−B、B−D 差分行（核心表）。"""
    models = list(models or [m for m in MAIN_TABLE_ORDER if (m, "A") in aggs])
    rows: list[dict[str, Any]] = []

    for cond_key in ("A", "B", "C", "D", "S1"):
        cond = C.CONDITIONS.get(cond_key)
        if cond is None:
            continue
        present = [(m, cond_key) for m in models if (m, cond_key) in aggs]
        if not present:
            continue
        row: dict[str, Any] = {
            "设置": cond_key,
            "划分": "随机" if cond.split == C.SPLIT_RANDOM else "时间",
            "统计特征": C.STAT_CONDITION_LABELS[cond.stat],
            "定位": cond.positioning,
        }
        for m in models:
            a = aggs.get((m, cond_key))
            row[model_label(m)] = a.cell("mae", space, digits=2) if a else "—"
        rows.append(row)

    # 差分行（口径取自 config.DIFFERENCES，不在此处重写表述）
    #
    # ⚠ 差分列的口径必须写明"左 − 右"，并**另给一列方向化措辞**。
    # 原因：MAE 越小越好，A−B 为负表示"A 更优"（即统计特征有价值），
    # 但负号极容易被读成"统计特征贡献为负"。实测 A−B = −297.48，
    # 若只给数字，审稿人会直接误读。方向化措辞可直接照抄进正文。
    #
    # ⚠ 注意：只有**差分行**的模型列才是差值，A/B/C/D 行的模型列是绝对 MAE。
    # 因此差值列的语义标记放在"设置"列文字里（如"A − B（左 − 右）"），
    # 而不是去改模型列的表头——改表头会让两种语义不同的数值共用一个列名。
    for diff_key in ("A-B", "B-D", "C-D", "A-C", "B'-D"):
        spec = C.DIFFERENCES.get(diff_key)
        if not spec:
            continue
        k1, k2 = diff_key.split("-")
        row: dict[str, Any] = {
            "设置": str(spec["label"]),
            "划分": "—",
            "统计特征": "—",
            "定位": str(spec["meaning"]),
        }
        any_val = False
        first_delta: float | None = None
        for m in models:
            a1, a2 = aggs.get((m, k1)), aggs.get((m, k2))
            if a1 and a2:
                d = a1.mean("mae", space) - a2.mean("mae", space)
                row[model_label(m)] = f"{d:+.2f}"
                if first_delta is None:
                    first_delta = d
                any_val = True
            else:
                row[model_label(m)] = "—"
        if not any_val:
            continue
        # 把方向化措辞并入"设置"列：既让符号自明，又不额外占一列（该刊要求少用图表）。
        # 例："A − B → A 更优，领先 297.48 分（MAE）"。
        assert first_delta is not None
        row["设置"] = (f"{spec['label']} → "
                       f"{C.describe_difference(first_delta, k1, k2)}")
        rows.append(row)

    return pd.DataFrame(rows)


def difference_summary(aggs: Mapping[tuple[str, str], Agg],
                       model: str, space: str = "test") -> list[dict[str, Any]]:
    """把大纲四·（二）的差分表连**限定语**与**方向化措辞**一起输出。

    防止成稿时漏掉限定语，也防止把"MAE 差值为负"误读成"贡献为负"。
    """
    out: list[dict[str, Any]] = []
    for key, spec in C.DIFFERENCES.items():
        k1, k2 = key.split("-")
        a1, a2 = aggs.get((model, k1)), aggs.get((model, k2))
        if not (a1 and a2):
            continue
        d = a1.mean("mae", space) - a2.mean("mae", space)
        out.append({
            "对比": str(spec["label"]),
            "含义（正文必须照此表述）": str(spec["meaning"]),
            "MAE 差值（左 − 右）": round(d, 2),
            "方向（照此写正文）": C.describe_difference(d, k1, k2),
            "是否干净": "是" if spec["clean"] else "否",
            "限定语（不得省略）": str(spec["caveat"]),
        })
    return out


# ---------------------------------------------------------------- 显著性


def resolve_seed(condition: str, model: str, seed: int | None = None,
                 suffix: str = "") -> int | None:
    """找出该 (条件, 模型) **实际落盘**的种子。

    ⚠ 不能像早期实现那样把 seed 写死为 0：真实实验的种子取自 `config.SEEDS`
    （42 起），而合成数据的冒烟运行用过 seed=0 并把缓存写进同一个目录。
    写死 0 的结果是"显著性表整表为空"（真实缓存一个都找不到），
    更糟的情况是会读到合成数据的缓存，把两套数据混在一起做配对检验。
    这里按 `config.SEEDS` 的顺序探测，探测不到就返回 None。
    """
    from . import experiments as E

    cands = [int(seed)] if seed is not None else list(C.SEEDS)
    for s in cands:
        if E.load_predictions(condition, model, s, suffix) is not None:
            return s
    return None


def pairwise_significance(condition: str, model_a: str, model_b: str,
                          metric: str = "mae", seed: int | None = None,
                          n_boot: int = C.BOOTSTRAP_N) -> M.BootstrapResult | None:
    """用缓存的逐样本预测做**配对** bootstrap（同一批测试样本）。

    没有预测缓存时返回 None——不做"用聚合指标近似"的假检验。
    种子由 `resolve_seed` 从实际缓存里解析；两模型即使落在不同种子也仍可配对，
    因为种子只作用于训练，测试集样本与顺序完全相同。
    """
    from . import experiments as E

    sa = resolve_seed(condition, model_a, seed)
    sb = resolve_seed(condition, model_b, seed)
    if sa is None or sb is None:
        return None
    pa = E.load_predictions(condition, model_a, sa)
    pb = E.load_predictions(condition, model_b, sb)
    if pa is None or pb is None:
        return None
    y = pa["y_test"]
    # 题目键必须一致，否则"配对"是两批不同样本上的差值，方差会被严重低估
    ka, kb = pa.get("test_problem_key"), pb.get("test_problem_key")
    if ka is not None and kb is not None and not np.array_equal(ka, kb):
        return None
    if metric in ("macro_f1", "qwk"):
        thr = M.BinThresholds(edges=tuple(pa["width_edges"]), scheme="equal_width")
        return M.bootstrap_binned_metrics(y, pa["test_pred"], pb["test_pred"],
                                         thr, metric=metric, n_boot=n_boot)
    return M.paired_bootstrap(y, pa["test_pred"], pb["test_pred"],
                             metric=metric, n_boot=n_boot)


def significance_table(condition: str = "A",
                       pairs: Sequence[tuple[str, str]] | None = None,
                       metrics: Sequence[str] = ("mae",),
                       models: Sequence[str] | None = None) -> pd.DataFrame:
    """模型两两配对的显著性表（每个候选模型 vs 主表最优模型）。"""
    from . import experiments as E

    rows: list[dict[str, Any]] = []
    # 默认：每个深度模型 vs LightGBM
    pairs = list(pairs or [])
    if not pairs:
        base = "lightgbm"
        pairs = [(m, base) for m in models or C.DEEP_MODELS]
    for a, b in pairs:
        for metric in metrics:
            r = pairwise_significance(condition, a, b, metric=metric)
            if r is None:
                continue
            rows.append({
                "对比": f"{model_label(a)} vs {model_label(b)}",
                "条件": condition,
                "指标": C.METRIC_LABELS.get(metric, metric),
                "Δ": round(r.delta, 4),
                "95% CI": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                "p": round(r.p_value, 4),
                "显著": "是" if r.significant else "否",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 导出


def to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """把 DataFrame 渲染成 Markdown 表格。

    不依赖 `tabulate`（pandas.to_markdown 的可选依赖，本机未安装）——
    自实现一份，避免为一个纯格式化功能引入新依赖。
    表格内容按该刊要求：表题在上（由调用方负责）、黑色线条、阿拉伯数字编号。
    """
    if df is None or df.empty:
        return "（无数据）"

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:            # NaN
                return "—"
            return f"{v:.4g}"
        s = str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    cols = ([str(df.index.name or "")] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, (_, row) in enumerate(df.iterrows()):
        cells = ([str(df.index[i])] if index else []) + [fmt(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, path: Path | str) -> Path:
    """同时存 CSV（供 Excel 排版）与 Markdown（供直接粘贴进正文）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    p.with_suffix(".md").write_text(to_markdown(df), encoding="utf-8")
    return p


def dump_aggs(aggs: Mapping[tuple[str, str], Agg], path: Path | str) -> Path:
    """把聚合结果存 json，供绘图脚本复用，避免重复聚合产生口径差异。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "model": a.model, "condition": a.condition, "n_seeds": a.n_seeds,
            "stats": a.stats, "is_leaky": a.is_leaky,
            "in_main_conclusion": a.in_main_conclusion,
            "n_test": a.n_test, "n_train": a.n_train,
        }
        for a in aggs.values()
    ]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p
