"""论文表格一键生成：结果文件 → 表 1～表 4（同时输出 .csv 与 .md）。

    python scripts/make_tables.py                    # 真实数据（默认）
    python scripts/make_tables.py --synthetic        # 合成数据自测结果
    python scripts/make_tables.py --out-dir outputs/results

为什么要把"结果 → 表格"固化成脚本
----------------------------------
论文里 4 张表的每个数字都必须能追溯到 `outputs/results/runs*.csv` 与
`data/processed/*.csv`。手工从 CSV 里抄数必然出错，而且一旦重跑实验，
正文与表格就会静默不一致。本脚本把这一步变成一条命令，**不产生任何新数字**：
所有口径（指标名、模型清单、分档切点、差分表述）一律从 `config` 读。

该刊排版要求（本项目已冻结）：表题在上、黑色线条、阿拉伯数字编号、中文标注。
表题与"数据来源与口径说明"写在 .md 里，正文直接粘贴即可。

⚠ 已知口径陷阱（踩过，务必保留处理）
------------------------------------
1. `results.load_all_runs` 会把 `runs_ablation*.csv` 一并并入，而消融阶段复用了
   条件键 `D`（"含/去来源特征"、"含/去模板句"两版）。若直接按 (模型, 条件) 聚合，
   D 格的均值会被消融记录污染——所以这里按 `stage` 列显式剔除消融阶段。
2. `results.main_table` 只输出**有数据**的模型行。大纲要求主表恒为 10 行
   （传统 6 + 深度 4），故本脚本补齐缺行并填"—"，绝不悄悄删行。
3. 差分行表述一个字都不能改写：A−B 与 B−D 的 `meaning` 从 `config.DIFFERENCES` 取，
   连 `caveat` 一起落在表下说明里——这是审稿必查的口径。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import dataset as D         # noqa: E402
from task2difficulty import results as R         # noqa: E402
from task2difficulty import vectorizers as V     # noqa: E402


# ---------------------------------------------------------------- 基础工具


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def suffix_of(synthetic: bool) -> str:
    """合成自测的产物加后缀，避免覆盖真实数据的论文产物（与 run_experiments.py 一致）。"""
    return "_synthetic" if synthetic else ""


def dataset_path(synthetic: bool) -> Path:
    return C.PROCESSED_DIR / ("dataset_synthetic.csv" if synthetic else "dataset.csv")


def split_index_path(synthetic: bool) -> Path:
    return C.PROCESSED_DIR / ("split_index_synthetic.csv" if synthetic else "split_index.csv")


def synthetic_warning(synthetic: bool) -> list[str]:
    """合成数据产物**不得**当作论文结果——直接把警告写进表下说明，防止误粘。"""
    if not synthetic:
        return []
    return ["⚠ 本表由合成数据自测集（dataset_synthetic.csv / runs_synthetic.csv）生成，"
            "仅用于流程自检，**不得**作为论文结果引用"]


def bin_labels(edges: Sequence[float] = C.DIFFICULTY_BINS) -> list[str]:
    """难度分档标签，口径与 figures.figure_error_analysis 的 (b) 面板完全一致。

    刻意不用 "(1200, 1600]" 那种数学写法：中文刊物里"（(1200, 1600]）"会套成双层括号，
    排出来很难看；改用 "1200~1600"，与分难度段命中率图的横轴标签逐字一致。
    """
    bounds = [-np.inf] + [float(e) for e in edges] + [np.inf]
    out: list[str] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if not np.isfinite(a):
            out.append(f"≤{b:g}")
        elif not np.isfinite(b):
            out.append(f">{a:g}")
        else:
            out.append(f"{a:g}~{b:g}")
    return out


def write_table(df: pd.DataFrame, path: Path, title: str, subtitle: str,
                notes: Sequence[str]) -> list[Path]:
    """按该刊体例落盘一张表：表题在上（Markdown 标题 + 说明行），表下附口径说明。

    CSV 与 Markdown 都写：CSV 供 Excel/Word 里重排版式，Markdown 供直接粘进正文。
    这里复用 `results.save_table`（项目统一的 csv+md 写出），再补上表题与说明行——
    这样 csv 内容永远与 md 表格一致，不会出现"两处数字不一样"。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    R.save_table(df, path)                       # 写 .csv 与纯表格 .md
    lines = [f"## {title}", "", f"（{subtitle}）", "", R.to_markdown(df), ""]
    if notes:
        # 该刊"表题在上"：表题与说明行在表格之前，口径说明在表格之后（一行）
        lines.append("> 说明：" + "；".join(notes) + "。")
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return [path.with_suffix(".csv"), path.with_suffix(".md")]


# ---------------------------------------------------------------- 运行记录


def load_runs_for_tables(results_dir: Path, synthetic: bool) -> pd.DataFrame:
    """载入原始运行记录，并剔除消融阶段（见模块 docstring 的陷阱 1）。"""
    runs = R.load_all_runs(results_dir, synthetic=synthetic)
    if "stage" in runs.columns:
        stage = runs["stage"].astype(str)
        keep = stage.str.startswith("factors") | stage.eq("size_control")
        dropped = int((~keep).sum())
        if dropped:
            log(f"  剔除消融阶段记录 {dropped} 行（它们复用条件键 D，会污染 D 格聚合）")
        runs = runs[keep].copy()
    return runs


# ---------------------------------------------------------------- 表 1


def build_table1(synthetic: bool) -> tuple[pd.DataFrame, str, list[str]]:
    """表 1：数据集统计信息。素材来自 dataset_summary + text_length_stats + 划分索引。"""
    p = dataset_path(synthetic)
    if not p.exists():
        raise FileNotFoundError(f"未找到数据集 {p}；表 1 需要 data/processed/dataset.csv")

    ds = D.load_dataset(p)
    s = D.dataset_summary(ds)
    tl = V.text_length_stats(list(ds.text))
    y = np.asarray(ds.y, dtype=np.float64)
    n = int(s["n_problems"])

    rows: list[list[str]] = []

    def add(item: str, value: str, note: str = "") -> None:
        rows.append([item, value, note])

    add("题目数（道）", f"{n}", "清洗后进入实验的题目数")
    add("数据快照日期", str(s["snapshot_date"]), "Codeforces 官方 API + 题面快照")
    if "year" in ds.meta.columns and len(ds.meta):
        add("比赛年份范围", f"{int(ds.meta['year'].min())}—{int(ds.meta['year'].max())}",
            "决定时间划分的可用性")
    add("rating 最小值 / 最大值", f"{s['rating_min']:.0f} / {s['rating_max']:.0f}",
        f"rating 以 {C.RATING_STEP} 为步长")
    add("rating 均值 ± 标准差", f"{s['rating_mean']:.1f} ± {s['rating_std']:.1f}", "")
    add("rating 中位数", f"{np.median(y):.0f}", "")
    add(f"rating = {C.RATING_FLOOR} 下钳位占比",
        f"{s['rating_floor_share']:.2%}（{int((y <= C.RATING_FLOOR).sum())} 道）",
        "最低档内部噪声最大，构成回归 MAE 的系统性下限")

    # 分档人数：与图 4(b)、附图 1 的切点完全一致，避免正文与图表两套档位
    for i, lab in enumerate(bin_labels()):
        lo = -np.inf if i == 0 else float(C.DIFFICULTY_BINS[i - 1])
        hi = np.inf if i == len(C.DIFFICULTY_BINS) else float(C.DIFFICULTY_BINS[i])
        cnt = int(((y > lo) & (y <= hi)).sum())
        add(f"难度分档 {i + 1}（{lab}）", f"{cnt} 道（{cnt / n:.1%}）",
            f"等宽 {len(C.DIFFICULTY_BINS) + 1} 档" if i == 0 else "")

    add("题面字符数（均值 / 中位数 / P95）",
        f"{tl['char_mean']:.0f} / {tl['char_median']:.0f} / {tl['char_p95']:.0f}", "清洗后题面")
    add("题面词数（均值 / 中位数 / P95）",
        f"{tl['word_mean']:.0f} / {tl['word_median']:.0f} / {tl['word_p95']:.0f}", "按空白切分")
    add("算法标签维度", f"{s['n_tags']} 维 + 标签数量 1 维",
        f"multi-hot 列前缀 {C.TAG_MULTIHOT_PREFIX}，另含 {C.TAG_COUNT_FEATURE}")
    add("算法标签 Top10（题数）",
        "、".join(f"{k}（{v}）" for k, v in s["tag_top10"].items()),
        "一题可有多个标签，故合计大于题目数")

    sp = split_index_path(synthetic)
    if sp.exists():
        si = D.load_split_index(sp)
        for kind, lab in ((C.SPLIT_RANDOM, "随机划分"), (C.SPLIT_TEMPORAL, "时间划分")):
            r = si.ratios(kind)
            cnt = {k: int(round(v * n)) for k, v in r.items()}
            rule = (f"训练 ≤{C.TEMPORAL_TRAIN_END_YEAR} / 验证 {C.TEMPORAL_VALID_YEAR} / "
                    f"测试 ≥{C.TEMPORAL_TEST_START_YEAR}"
                    if kind == C.SPLIT_TEMPORAL else
                    f"按 {C.RANDOM_SPLIT[0]:.0%}:{C.RANDOM_SPLIT[1]:.0%}:"
                    f"{C.RANDOM_SPLIT[2]:.0%} 固定种子划分")
            add(f"{lab}三段规模（训练 / 验证 / 测试）",
                f"{cnt['train']} / {cnt['valid']} / {cnt['test']}"
                f"（{r['train']:.1%} / {r['valid']:.1%} / {r['test']:.1%}）", rule)

    df = pd.DataFrame(rows, columns=["项目", "数值", "说明"])
    subtitle = (f"Codeforces 公开题目抽样，快照日期 {s['snapshot_date']}，"
                f"共 {n} 道有 rating 的题目")
    notes = [
        f"数据来源：Codeforces 官方 API（problemset.problems / contest.list）与题面快照，"
        f"快照日期 {s['snapshot_date']}",
        f"rating 取值 {s['rating_min']:.0f}～{s['rating_max']:.0f}，步长 {C.RATING_STEP}，"
        f"{C.RATING_FLOOR} 为下钳位（该档内部噪声最大，构成 MAE 系统性下限）",
        f"难度分档为等宽 {len(C.DIFFICULTY_BINS) + 1} 档，切点 "
        f"{'/'.join(f'{e:g}' for e in C.DIFFICULTY_BINS)}（左开右闭）；"
        f"等频口径的敏感性见正文",
        f"两套划分共用同一索引文件，时间划分三段式为 训练 ≤{C.TEMPORAL_TRAIN_END_YEAR} / "
        f"验证 = {C.TEMPORAL_VALID_YEAR} / 测试 ≥{C.TEMPORAL_TEST_START_YEAR}",
    ] + synthetic_warning(synthetic)
    return df, subtitle, notes


# ---------------------------------------------------------------- 表 2

#: 特征组的输入通道 token（取自 config.MODEL_INPUT_CHANNEL 的取值分解）。
#: 这里只做"组 → 通道"的映射，模型清单仍然从 config 动态取，不在脚本里写死模型名。
_GROUP_CHANNEL_TOKENS: dict[str, tuple[str, ...]] = {
    "text": ("tfidf", "text"),
    "constraint": ("struct",),
    "tag": ("struct", "tag_only"),
    "stat": ("struct",),
    "provenance": ("struct",),
}

#: 文本特征在训练时由 vectorizers 现场构造，不是 CSV 列，无法从数据集枚举，故在此描述。
_TEXT_GROUP_FEATURES = "TF-IDF 向量（传统侧）/ 预训练词嵌入（深度侧）"


def _consumers_cell(group: str) -> str:
    """表 2 的"输入给谁"列：从 config.MODEL_INPUT_CHANNEL 反查消费该特征组的模型。"""
    tokens = _GROUP_CHANNEL_TOKENS[group]
    trad: list[str] = []
    deep: list[str] = []
    app: list[str] = []
    for m, ch in C.MODEL_INPUT_CHANNEL.items():
        if not any(t in ch.split("+") for t in tokens):
            continue
        if m in C.APPENDIX_MODELS:
            app.append(R.model_label(m))
        elif m in C.DEEP_MODELS:
            deep.append(R.model_label(m))
        else:
            trad.append(R.model_label(m))
    if not (trad or deep or app):
        return "—"
    if trad and (deep or app):
        head = "两侧通用"
    elif trad:
        head = "传统侧"
    else:
        head = "深度侧"
    names = trad + deep + app
    cell = f"{head}（{len(names)} 项）：" + "、".join(names)
    if group == "provenance":
        cell += "；仅 provenance 消融时保留，主实验恒剔除"
    return cell


def build_table2(synthetic: bool) -> tuple[pd.DataFrame, str, list[str]]:
    """表 2：特征体系汇总（静态表）。

    这张表描述的是**口径**而不是某次运行的结果，因此完全由 config 生成：
    特征组名、新题可得性、各组特征清单都取自冻结常量。数据规模变了它也不变，
    正文里可以放心引用。
    """
    feats: dict[str, str] = {
        "text": _TEXT_GROUP_FEATURES,
        "constraint": "、".join(C.CONSTRAINT_FEATURES),
        "tag": f"{C.TAG_MULTIHOT_PREFIX}* multi-hot（算法标签）、{C.TAG_COUNT_FEATURE}（标签数量）",
        "stat": "、".join(C.STAT_FEATURES),
        "provenance": "、".join(C.PROVENANCE_FEATURES),
    }
    remarks: dict[str, str] = {
        "text": "训练时由 vectorizers 现场构造；词表/tokenizer 只在训练集上拟合。"
                "核 SVR 的文本通道先经训练集上拟合的随机化截断 SVD 降至 "
                f"{C.TEXT_SVD_DIMS} 维（高维稀疏下 RBF 核的距离集中效应会让全维核矩阵失去分辨力，"
                "且 libsvm 在该规模下不可解），其余传统模型使用全维稀疏表示",
        "constraint": "冷启动场景的核心特征：题面里可解析，新题完全可得",
        "tag": "冷启动下仍完全可得（与约束规模特征共同构成残余增益的来源）",
        "stat": f"B/D 格整组置空＝缺失（不填 0）；除 "
                f"{'、'.join(C.NAN_NATIVE_MODELS)} 外的模型按 "
                f"{C.COLD_START_IMPUTATION} 处理（随机森林因稀疏输入限制亦在此列，"
                f"理由见三·（六）），避免用训练集中位数把统计信息带回冷启动",
        "provenance": (
            "⚠ 来源特征警告：" + "、".join(C.PROVENANCE_FEATURES) +
            " 是 rating 的强代理（题号 A 必简单、Div.1 必偏难），模型无需读懂题面即可猜得八九不离十；"
            "**不得作为普通特征混入**，必须单独成组，并做 provenance 消融 + 弱基线自曝"
        ),
    }

    rows: list[dict[str, str]] = []
    for g, gname in C.FEATURE_GROUP_LABELS.items():
        rows.append({
            "特征组": gname,
            "具体特征": feats.get(g, "—"),
            "输入给谁": _consumers_cell(g),
            "新题可得性": C.FEATURE_GROUP_AVAILABILITY[g],
            "口径说明": remarks.get(g, ""),
        })
    df = pd.DataFrame(rows, columns=["特征组", "具体特征", "输入给谁", "新题可得性", "口径说明"])

    subtitle = "特征组 / 具体特征 / 输入通道 / 新题可得性（含来源特征危害）"
    cold = "、".join(C.FEATURE_GROUP_LABELS[g] for g in C.COLD_START_STRUCT_GROUPS)
    notes = [
        "本表为特征体系口径表（与数据规模无关），特征组名与新题可得性取自实验配置的冻结口径",
        f"纯冷启动（B/D 格）下仍完全可得的结构化特征组：{cold}",
        "来源特征（" + "、".join(C.PROVENANCE_FEATURES) + "）是 rating 的强代理，"
        "须单独成组并做 provenance 消融与弱基线自曝，不得作为普通特征混入",
        f"统计特征置空后除 {'、'.join(C.NAN_NATIVE_MODELS)} 外的模型统一采用 "
        f"{C.COLD_START_IMPUTATION}（含随机森林——sklearn 的随机森林仅在稠密输入下支持 NaN，"
        f"而本文特征矩阵为省内存必须保持稀疏），不使用训练集中位数填补"
        f"（后者会把统计信息偷偷带回冷启动格）",
    ] + synthetic_warning(synthetic)
    return df, subtitle, notes


# ---------------------------------------------------------------- 表 3


def _side_of(model: str) -> str:
    """传统侧 / 深度侧的归属也按 config 判定，不在脚本里写死。"""
    if model in C.TRADITIONAL_MODELS:
        return "传统机器学习"
    if model in C.DEEP_MODELS:
        return "深度学习"
    if model in C.APPENDIX_MODELS:
        return "附表项"
    return "其他"


def build_table3(aggs: dict, condition: str,
                 synthetic: bool = False) -> tuple[pd.DataFrame, str, list[str]]:
    """表 3：主对比结果（模型 × 5 指标，均值±标准差）。

    ⚠ 主表恒为 10 行（传统 6 + 深度 4）：`results.main_table` 只输出有数据的模型，
    缺数据的行在这里补齐并填"—"，绝不悄悄删行——审稿人一眼就能看出哪些模型还没跑。
    """
    tbl = R.main_table(aggs, condition=condition)
    present: dict[str, dict[str, Any]] = {}
    if tbl is not None and not tbl.empty:
        for _, row in tbl.iterrows():
            present[str(row["model"])] = row.to_dict()

    metric_labels = [C.METRIC_LABELS[m] for m in C.METRICS]
    sample_col = "训练/测试样本数"
    rows: list[dict[str, Any]] = []
    for m in C.MAIN_TABLE_MODELS:
        if m in present:
            r = dict(present[m])
            # main_table 的列名写作"训练/验证/测试"，但 Agg 只带 n_train/n_test，
            # 验证集规模并未进入聚合——按实改名，免得正文写出一个不存在的数。
            r[sample_col] = r.pop("训练/验证/测试", "—")
        else:
            r = {"模型": R.model_label(m), "model": m}
            r.update({lab: "—" for lab in metric_labels})
            r[sample_col] = "—"
        r["类别"] = _side_of(m)
        rows.append(r)

    lead = ["模型", "类别", "model"] + metric_labels + [sample_col]
    df = pd.DataFrame(rows)
    for c in lead:
        if c not in df.columns:
            df[c] = "—"
    extra = [c for c in df.columns if c not in lead]
    df = df[lead + extra]

    # 该刊要求"加粗最优、下划线次优"：按 MAE 与 macro-F1 两列标注
    df = R.mark_best(df, C.METRIC_LABELS["mae"], lower_is_better=True)
    df = R.mark_best(df, C.METRIC_LABELS["macro_f1"], lower_is_better=False)

    cond = C.CONDITIONS.get(condition)
    n_missing = sum(1 for m in C.MAIN_TABLE_MODELS if m not in present)
    sub = (f"条件 {condition} 格；" + (f"{cond.positioning}；" if cond else "")
           + f"已跑 {len(present)}/{len(C.MAIN_TABLE_MODELS)} 个主表模型")
    notes = [
        f"数据来源：outputs/results/runs*.csv 的 {condition} 格，跨随机种子均值±标准差；"
        f"加粗＝该列最优、下划线＝次优（按 MAE 与 macro-F1 判定）",
        f"表中数值为测试集指标，测试集只碰一次；主表构成固定为传统侧 "
        f"{len(C.TRADITIONAL_MODELS)} 项 + 深度侧 {len(C.DEEP_MODELS)} 项＝"
        f"{len(C.MAIN_TABLE_MODELS)} 行，未运行的模型保留空行并以「—」占位",
    ]
    # ⚠ 不再用 leaky 标志拼"含同源泄漏的乐观上界"这类措辞：那会把 A 格
    # （随机划分，统计与标签同快照、**不含未来信息**）错按成 C 格的问题。
    # 各格的定位一律取自 config.Conditions[*].positioning，一处改、处处一致。
    # 两种成因的区分见 config.Condition 的 docstring。
    if cond is not None:
        notes.append(f"{condition} 格定位：{cond.positioning}")
    # 大纲四·（三）：规模对照 B′ 如做，以附注形式进主表
    bprime = [(m, aggs[(m, C.SIZE_CONTROL_KEY)]) for m in C.MAIN_TABLE_MODELS
              if (m, C.SIZE_CONTROL_KEY) in aggs]
    if bprime:
        notes.append("规模对照 " + C.SIZE_CONTROL_KEY + "（" + C.SIZE_CONTROL_NOTE + "）："
                     + "、".join(f"{R.model_label(m)} test MAE={a.mean('mae'):.2f}"
                                 for m, a in bprime))
    if n_missing:
        notes.append(f"⚠ 当前数据尚缺 {n_missing} 个主表模型（"
                     + "、".join(R.model_label(m) for m in C.MAIN_TABLE_MODELS if m not in present)
                     + "），需先运行 scripts/run_experiments.py 补齐后再定稿")
    notes += synthetic_warning(synthetic)
    return df, sub, notes


# ---------------------------------------------------------------- 表 4


def _leak_mark(cond: C.Condition) -> str:
    """泄漏标注列：由 config 的 leaky / in_main_conclusion 判定，不在脚本里写死格名。"""
    if not cond.leaky:
        return "无泄漏（主结论格）" if cond.in_main_conclusion else "无泄漏"
    if cond.in_main_conclusion:
        return "乐观上界（含同源泄漏），不可作部署依据"
    return "泄漏乐观上界，不可作部署依据"


def _diff_row(diff_key: str, models: Sequence[str], aggs: dict, space: str) -> dict[str, Any] | None:
    """构造一行差分（含义与限定语一律取自 config.DIFFERENCES，一个字都不改写）。

    ⚠ 差值为「左格 MAE − 右格 MAE」。MAE 越小越好，所以**负值表示左侧更优**——
    这与"贡献为正"的直觉相反，实测极易误读（A−B = −297.48 会被读成
    "统计特征有负贡献"，实际是"缺统计特征后劣化 297.48 点"）。
    因此把方向化措辞（`config.describe_difference`）并入"设置"列：
    例如 "A − B → A 更优，领先 297.48 分（MAE）"。这样符号在表内自明，
    也不必为方向另开一列（该刊要求少用图表）。
    """
    spec = C.DIFFERENCES.get(diff_key)
    if not spec:
        return None
    k1, k2 = diff_key.split("-")
    row: dict[str, Any] = {
        "设置": str(spec["label"]), "划分": "—", "统计特征": "—",
        "泄漏标注": "—", "定位": str(spec["meaning"]),
    }
    any_val = False
    first_delta: float | None = None
    for m in models:
        a1, a2 = aggs.get((m, k1)), aggs.get((m, k2))
        if a1 is not None and a2 is not None:
            d = a1.mean("mae", space) - a2.mean("mae", space)
            row[R.model_label(m)] = f"{d:+.2f}"
            if first_delta is None:
                first_delta = d
            any_val = True
        else:
            row[R.model_label(m)] = "—"
    if not any_val:
        return None
    assert first_delta is not None
    row["设置"] = f"{spec['label']} → {C.describe_difference(first_delta, k1, k2)}"
    return row


def build_table4(aggs: dict, synthetic: bool) -> tuple[pd.DataFrame, str, list[str]]:
    """表 4（★ 核心表）：A/B/C/D 并列 + A−B、B−D 差分行（+ B′ 若有）。

    `results.factor_table` 已经实现了主体；本函数补三件事：
      1. 增列"泄漏标注"——C 行必须写明"泄漏乐观上界，不可作部署依据"；
      2. 条件行补空缺（factor_table 会整行丢弃无数据的条件，核心表不能缺格）；
      3. 追加 B′ 行与 B′−D 差分行，并按大纲加"两套划分三段规模"行便于核对 B−D 的规模混杂。
    """
    models = [m for m in R.MAIN_TABLE_ORDER
              if any((m, k) in aggs for k in list(C.CONDITIONS) + [C.SIZE_CONTROL_KEY])]
    ft = R.factor_table(aggs, models=models) if models else pd.DataFrame()

    rows: dict[str, dict[str, Any]] = {}
    if ft is not None and not ft.empty:
        for _, r in ft.iterrows():
            rows[str(r["设置"])] = r.to_dict()

    # ---- 条件行：按 config 顺序补齐（非 optional 的格子必须有行）----
    ordered: list[dict[str, Any]] = []
    for key, cond in C.CONDITIONS.items():
        if key in rows:
            r = rows.pop(key)
        elif cond.optional or not models:
            continue
        else:
            r = {"设置": key, "划分": "随机" if cond.split == C.SPLIT_RANDOM else "时间",
                 "统计特征": C.STAT_CONDITION_LABELS[cond.stat], "定位": cond.positioning,
                 **{R.model_label(m): "—" for m in models}}
        r["泄漏标注"] = _leak_mark(cond)
        ordered.append(r)

    # ---- B′ 规模对照：大纲四·（二）"强烈建议"，做了就必须进表 ----
    bprime_rows: list[dict[str, Any]] = []
    if any(k == C.SIZE_CONTROL_KEY for _, k in aggs.keys()):
        # B′ 是 B 格的训练量对照（同划分、同特征条件，只把训练集下采样），故行列取自 B 格
        base = C.CONDITIONS["B"]
        r = {"设置": C.SIZE_CONTROL_KEY, "划分": "随机",
             "统计特征": C.STAT_CONDITION_LABELS[base.stat],
             "泄漏标注": _leak_mark(base), "定位": C.SIZE_CONTROL_NOTE}
        for m in models:
            a = aggs.get((m, C.SIZE_CONTROL_KEY))
            r[R.model_label(m)] = a.cell("mae", digits=2) if a is not None else "—"
        bprime_rows.append(r)

    # ---- 差分行：A−B、B−D 必做；B′−D、C−D、A−C 有数据才出 ----
    diff_order = ["A-B", "B-D", "B'-D", "C-D", "A-C"]
    diff_rows: dict[str, dict[str, Any]] = {}
    for key in diff_order:
        r = _diff_row(key, models, aggs, "test")
        if r is not None:
            diff_rows[key] = r
    # factor_table 已算好的 C−D / A−C 行沿用其数值口径（同样的 mean 相减），无需重复构造

    # ---- 两套划分三段规模行（大纲四·（二）：B−D 的规模混杂必须可核对）----
    scale_rows: list[dict[str, Any]] = []
    measured: dict[str, dict[str, float]] = {}
    sp = split_index_path(synthetic)
    if sp.exists():
        si = D.load_split_index(sp)
        for kind, lab in ((C.SPLIT_RANDOM, "随机划分"), (C.SPLIT_TEMPORAL, "时间划分")):
            r_ = si.ratios(kind)
            measured[kind] = r_
            scale_rows.append({
                "设置": f"{lab}三段规模", "划分": "随机" if kind == C.SPLIT_RANDOM else "时间",
                "统计特征": "—", "泄漏标注": "—",
                "定位": (f"训练/验证/测试 = {r_['train']:.1%} / {r_['valid']:.1%} / "
                         f"{r_['test']:.1%}"),
                **{R.model_label(m): "—" for m in models},
            })

    # 最终行序：A/B/C/D（+S1'）→ B′ → 差分行 → 规模行
    final = ordered + bprime_rows
    for key in diff_order:
        if key in diff_rows:
            final.append(diff_rows[key])
    final += scale_rows

    lead = ["设置", "划分", "统计特征", "泄漏标注", "定位"] + [R.model_label(m) for m in models]
    df = pd.DataFrame(final)
    for c in lead:
        if c not in df.columns:
            df[c] = "—"
    df = df[lead + [c for c in df.columns if c not in lead]]

    sub = ("因子设计＝划分方式 × 统计特征可得性；"
           "表内数值为测试集 MAE（rating 点，越小越好）")
    notes = [
        "数据来源：outputs/results/runs*.csv（含规模对照 runs_bprime*.csv），跨随机种子均值",
        "**差值列的口径＝左格 MAE − 右格 MAE**（差值行的模型列是差值，"
        "A/B/C/D 行的模型列是绝对 MAE，二者不可混看）；"
        "MAE 越小越好，故**差值为负表示左侧更优**；"
        "为免误读，差值行的「设置」列已直接写出方向（如「A − B → A 更优，领先 297.48 分」），"
        "正文可直接采用该措辞",
        "C 格为泄漏乐观上界，不可作部署依据（测试集新题使用了上线后的最终统计量）；"
        "主结论只建立在 A/B/D 三格上",
    ]
    for key in diff_order:
        spec = C.DIFFERENCES.get(key)
        if spec is None or key not in diff_rows:
            continue
        notes.append(f"{spec['label']} = {spec['meaning']}（"
                     + ("对比方法干净" if spec["clean"] else "对比不干净")
                     + f"）；限定：{spec['caveat']}")
    # ⚠ B−D 的 caveat 里写的是大纲【十】**全量数据**的占比（时间划分训练 65.5%），
    # 而本次实验实际跑的是 391 题抽样，实测占比会偏离。若不在表下说明，表内规模行与
    # caveat 就会自相矛盾（前者 70.6%，后者 65.5%）。这里把实测值一并报出，正文以实测为准。
    if C.SPLIT_RANDOM in measured and C.SPLIT_TEMPORAL in measured:
        gap_pp = (measured[C.SPLIT_RANDOM]["train"]
                  - measured[C.SPLIT_TEMPORAL]["train"]) * 100
        exp = C.EXPECTED_TEMPORAL_RATIOS["train"]
        dev_pp = abs(measured[C.SPLIT_TEMPORAL]["train"] - exp) * 100
        note = (f"本次实测训练量差 {gap_pp:.1f} pp"
                f"（随机 {measured[C.SPLIT_RANDOM]['train']:.1%} vs 时间 "
                f"{measured[C.SPLIT_TEMPORAL]['train']:.1%}），以本表规模行为准")
        if dev_pp > C.RATIO_TOLERANCE_PP:
            note += (f"；时间划分实测占比与大纲【十】给出的 {exp:.1%} 相差 {dev_pp:.1f} pp"
                     f"（超 {C.RATIO_TOLERANCE_PP:g} pp 容差，原因是本次为抽样数据而非全量），"
                     f"B−D 的规模混杂须按实测值重新计算")
        notes.append(note)
    if C.SIZE_CONTROL_KEY not in [r["设置"] for r in bprime_rows]:
        notes.append(f"未做规模对照 {C.SIZE_CONTROL_KEY}（随机划分训练集下采样至时间划分同规模），"
                     f"故 B−D 的训练量混杂"
                     + (f"（随机 {measured[C.SPLIT_RANDOM]['train']:.1%} vs 时间 "
                        f"{measured[C.SPLIT_TEMPORAL]['train']:.1%}）" if measured else "")
                     + "暂无法剔除，须先运行 scripts/run_experiments.py --stage size_control")
    notes += synthetic_warning(synthetic)
    return df, sub, notes


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成论文表 1～表 4（.csv + .md）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="产出目录默认 outputs/results；--synthetic 的产物带 _synthetic 后缀，不会覆盖真实数据产物。",
    )
    ap.add_argument("--synthetic", action="store_true",
                    help="用合成数据自测结果（dataset_synthetic.csv / runs_synthetic.csv）")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认 outputs/results")
    ap.add_argument("--results-dir", default=None, help="结果输入目录，默认 outputs/results")
    ap.add_argument("--main-condition", default=None, choices=sorted(C.CONDITIONS),
                    help="表 3 使用的条件格，默认 A 格（调参格）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else C.RESULTS_DIR
    results_dir = Path(args.results_dir) if args.results_dir else C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = suffix_of(args.synthetic)
    cond_key = args.main_condition or ("A" if "A" in C.CONDITIONS else
                                      next(k for k, c in C.CONDITIONS.items()
                                           if c.in_main_conclusion))

    issues = C.validate()
    for i in issues:
        log(f"[口径自检失败] {i}")
    if issues:
        return 1

    log(f"模式：{'合成数据自测' if args.synthetic else '真实数据'}；"
        f"结果目录 {results_dir}；产出目录 {out_dir}")
    produced: list[Path] = []
    skipped: list[str] = []

    # ---- 表 1 ----
    log("表 1：数据集统计信息")
    try:
        df1, sub1, notes1 = build_table1(args.synthetic)
        produced += write_table(df1, out_dir / f"table1_dataset{suf}",
                                "表 1  数据集统计信息", sub1, notes1)
        log(f"  {len(df1)} 行 -> table1_dataset{suf}.csv / .md")
    except Exception as ex:
        skipped.append(f"表 1（{type(ex).__name__}: {ex}）")
        log(f"  [skip] 表 1 生成失败：{type(ex).__name__}: {ex}")

    # ---- 表 2（静态表，不依赖任何运行结果）----
    log("表 2：特征体系汇总（静态口径表）")
    df2, sub2, notes2 = build_table2(args.synthetic)
    produced += write_table(df2, out_dir / f"table2_features{suf}",
                            "表 2  特征体系汇总", sub2, notes2)
    log(f"  {len(df2)} 行 -> table2_features{suf}.csv / .md")

    # ---- 表 3 / 表 4 需要运行记录 ----
    aggs: dict = {}
    try:
        runs = load_runs_for_tables(results_dir, args.synthetic)
        aggs = R.aggregate(runs)
        cells = "、".join(f"{m}×{k}" for (m, k) in sorted(aggs))
        log(f"运行记录 {len(runs)} 行，聚合出 {len(aggs)} 个（模型 × 条件）格：{cells}")
    except Exception as ex:
        skipped.append(f"表 3 / 表 4（{type(ex).__name__}: {ex}）")
        log(f"  [skip] 无法载入运行记录：{type(ex).__name__}: {ex}")
        log("         需先运行 scripts/run_experiments.py --stage factors（或 --smoke 自测）")

    if aggs:
        n_models = len({m for m, _ in aggs})
        if n_models < len(C.MAIN_TABLE_MODELS):
            log(f"  [warn] 当前只有 {n_models}/{len(C.MAIN_TABLE_MODELS)} 个主表模型的数据，"
                f"表 3 将保留缺失行并填「—」；数据不足以支撑完整的主对比结论，"
                f"请先跑更多模型（python scripts/run_experiments.py --stage factors --models ...）")

        # ⚠ 表号必须按**正文首次引用顺序**编（学报体例）。
        # 正文里四·（二）因子设计先于四·（三）主对比，因此因子设计是表 3、
        # 主对比是表 4。早期版本沿用"主对比＝表 3、因子设计＝表 4"的旧编号，
        # 结果是表 4 在表 3 之前被引用——审稿人会直接指出。
        log(f"表 3：因子设计结果（条件 A/B/C/D + 差分行）")
        df3, sub3, notes3 = build_table4(aggs, args.synthetic)
        produced += write_table(df3, out_dir / f"table3_factors{suf}",
                                "表 3  因子设计结果（划分方式 × 统计特征可得性）",
                                sub3, notes3)
        log(f"  {len(df3)} 行 -> table3_factors{suf}.csv / .md")

        log(f"表 4：主对比结果（条件 {cond_key}）")
        df4, sub4, notes4 = build_table3(aggs, cond_key, synthetic=args.synthetic)
        produced += write_table(df4, out_dir / f"table4_main{suf}",
                                "表 4  主对比结果（模型 × 5 指标）", sub4, notes4)
        log(f"  {len(df4)} 行 -> table4_main{suf}.csv / .md")

        if not any(k == "B'" for _, k in aggs):
            log("  [warn] 未发现 B′ 规模对照记录，表 3 未含 B′ 行；"
                "如需剔除 B−D 的训练量混杂，请运行 "
                "python scripts/run_experiments.py --stage size_control")

    # ---- 产出清单 ----
    print("\n===== 产出清单（表） =====")
    if produced:
        for p in produced:
            print(p.resolve())
    else:
        print("（无）")
    if skipped:
        print("\n===== 跳过项 =====")
        for s in skipped:
            print(f"- {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
