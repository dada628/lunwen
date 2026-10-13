#!/usr/bin/env python
"""人类标注一致性（human κ）实验工具链（大纲四·（五）实验 5）。

为什么需要这个脚本
------------------
论文要回答"模型是否已达到人类标注水平"。这个结论**不能**只靠模型指标自证，
必须有一个外部参照：请有竞赛经验的教师/研究生对同一批题独立定档，
算人类之间的一致性（Cohen's κ / Fleiss' κ / QWK），再与模型一致率对比。

四个子命令（前三个对应大纲【九】第 3 周后半的"人类 κ 实验"）::

    python scripts/human_kappa.py sample    # 分层抽 100 题 + 生成 3 份独立评题表
    python scripts/human_kappa.py agree     # 回收后算 κ（Fleiss/Cohen/QWK/α）
    python scripts/human_kappa.py compare   # 人类共识 vs 模型（**只能对 D 格**）
    python scripts/human_kappa.py selftest  # 开发自检：临时目录造模拟填答跑通 agree/compare

三条口径红线（写进代码而不是写进提醒邮件，否则一定会漂移）
--------------------------------------------------------
1. **评题表必须"盲"**：只给题面 + 算法标签，绝不出现 rating / solved_count /
   days_since_publish / contest_id / problem_index / div_type / year / name 等任何
   提示难度或题目身份的列。理由：人类只能靠题面+标签判断，与 D 格（时间划分 +
   统计特征置空）的特征条件对齐；给了提交统计就等于给人类开了模型没有的外挂，
   一致率立刻不可比。见 `assert_blind_sheet()`。
2. **分层抽样**：按 rating 段分层（低/中/高）+ 新题旧题混合。若纯随机抽，
   样本会被 1200~2000 的中等难度主导，κ 只代表中间段共识，
   而本文恰恰关心两端误差大——分层后才与四·（八）的分段分析口径一致。
3. **评委身份分开**：本实验要的是**有竞赛经验者**（估计任务的人类可达上界），
   与五·（一）教学阈值重校准所需的**熟悉高职学生水平的任课教师**是两拨人，
   样本与结论必须分开报告，不得混用（见 instructions.md 里的身份说明）。

依赖：只用 numpy/pandas/sklearn/statsmodels（均已安装）；predictions 读取走
`experiments.load_predictions`，因此 compare 会间接导入 torch（已安装）。
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import dataset as D          # noqa: E402
from task2difficulty import fetchers as F         # noqa: E402
from task2difficulty import metrics as M          # noqa: E402

# ---------------------------------------------------------------- 常量

#: 人类 κ 实验的所有产物统一放这里（与 results/ 分开：这里是人填的表，不是程序输出）
KAPPA_DIR: Path = C.OUTPUTS_DIR / "human_kappa"
KAPPA_MD: str = "kappa_results.md"
KAPPA_JSON: str = "kappa_results.json"
SHEET_PATTERN: str = "rating_sheet_rater*.csv"
KEY_FILE: str = "key.csv"
ITEMS_FILE: str = "items.csv"
INSTRUCTIONS_FILE: str = "instructions.md"
META_FILE: str = "sample_meta.json"

#: 评题表**只允许**有这几列。任何新增列都必须先回答"它会不会提示难度或身份"。
SHEET_COLUMNS: tuple[str, ...] = ("item_id", "statement", "tags", "difficulty_1to5", "comment")

#: 明确禁止出现在评题表里的列名（大纲四·（五）口径 1）。
#: 分两类：精确列名 + 子串模式。子串模式是为了拦住将来数据管线新增的派生列
#: （例如 solved_count_log、rating_bin 之类），只靠精确名单拦不住。
FORBIDDEN_SHEET_COLUMNS: tuple[str, ...] = (
    "rating", "rating_bin", "solved_count", "days_since_publish", "contest_id",
    "contest_id_feat", "problem_index", "problem_key", "div_type", "contest_type",
    "year", "name", "start_time", "snapshot_date", "template_hit",
    "stratum", "is_new", "statement_len", "tag_count", "sample_seed",
    "constraint_n_upper", "multi_test_count", "constraint_has_n",
    "constraint_n_imputed", "constraint_hits", "constraint_max_upper",
    "is_interactive",
)
FORBIDDEN_SHEET_SUBSTRINGS: tuple[str, ...] = (
    "rating", "solved", "year", "div_", "contest", "problem", "index",
    "name", "days", "snapshot", "stratum",
)

#: 报告里 splice 用的标记：`agree` 重写报告正文时必须原样保留"人类 vs 模型"一节，
#: 否则先跑 compare 再跑 agree 会把模型对比结果悄悄删掉。
BLOCK_BEGIN: str = "<!-- HUMAN_VS_MODEL:BEGIN -->"
BLOCK_END: str = "<!-- HUMAN_VS_MODEL:END -->"

#: 视为"留空"的填答内容（评题人可能写 NA / 不确定 / 问号等）。
_BLANK_MARKERS: frozenset[str] = frozenset({
    "", "na", "n/a", "nan", "none", "null", "-", "--", "?", "？", "无", "不确定", "不会",
})

#: 各档的直觉描述（只描述档位本身，不含任何具体题目信息，不构成提示）。
#: 长度必须等于分档数，配置改了要立刻报错而不是静默错位。
_BIN_HINTS: tuple[str, ...] = (
    "入门：基本语法、模拟、简单数学即可",
    "基础：贪心/排序/简单实现，会基本 STL",
    "中等：常见算法（二分、DP、图的基本应用）",
    "较难：需组合多个知识点或较细的观察",
    "困难：需较深的算法设计或较强的洞察",
)

_STRATUM_NAMES: tuple[str, ...] = ("低", "中", "高")


# ---------------------------------------------------------------- 小工具


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_k(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def jsonable(obj: Any) -> Any:
    """把 numpy/pandas 类型递归转成可 json 序列化的 Python 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------- 分档与分层


def bin_thresholds() -> M.BinThresholds:
    """按 config 的等宽口径构造分档阈值（与模型评估**同一条代码路径**）。

    ⚠ 这里刻意调用 `metrics.BinThresholds.apply` 而不是自己写 np.digitize：
    人类档位与模型档位必须由同一函数产出，否则边界处（rating 恰好等于
    1200/1600/2000/2400 的题）归属可能不一致，一致率就成了两套口径相减的伪结果。

    该函数的语义是**左开右闭**（大纲二·（五）的"≤1200 / ≤1600 / …"）：
    `apply(1200) == 0`（第 1 档）。这一点在实现阶段被修正过一次——
    早期 `np.digitize` 默认的 `right=False` 会让 1200 落进第 2 档，
    与大纲措辞相反；本任务 16.1% 的题正压在切点上，故该差异影响显著。
    """
    edges = tuple([-np.inf] + [float(e) for e in C.DIFFICULTY_BINS] + [np.inf])
    return M.BinThresholds(edges=edges, scheme="equal_width", fitted_on="config")


def n_bins() -> int:
    return len(C.DIFFICULTY_BINS) + 1


def rating_to_bin(rating: np.ndarray | float) -> np.ndarray:
    """rating -> 1..5 档（1 最易…5 最难），与模型评估口径完全一致。"""
    return bin_thresholds().apply(np.asarray(rating, dtype=np.float64)) + 1


def bin_interval_text(i: int) -> str:
    """第 i 档（0-based）的区间文字，按 `apply()` 的**左开右闭**语义生成。

    左开右闭 = rating 恰等于切点时归入**较低**那一档，与大纲的"≤1200"一致。
    因此最低档写作 `rating ≤ 1200`、其余写作 `1200 < rating ≤ 1600`。
    """
    bounds = [-np.inf] + [float(e) for e in C.DIFFICULTY_BINS] + [np.inf]
    lo, hi = bounds[i], bounds[i + 1]
    if not np.isfinite(lo):
        return f"rating ≤ {hi:g}"
    if not np.isfinite(hi):
        return f"rating > {lo:g}"
    return f"{lo:g} < rating ≤ {hi:g}"


def stratum_bounds() -> list[float]:
    """低/中/高三层的 rating 边界，从 `C.DIFFICULTY_BINS` 与 `C.HUMAN_KAPPA_STRATA` 派生。

    做法：把 `len(DIFFICULTY_BINS)+1` 个等宽档合并成 `HUMAN_KAPPA_STRATA` 层，
    边界取 DIFFICULTY_BINS 里**等距**的几个切点（含首尾）。
    默认配置下 DIFFICULTY_BINS=(1200,1600,2000,2400)、STRATA=3
    → 边界 1200 与 2400 → 低=（最易档）/ 中=（中间三档）/ 高=（最难档）。

    为什么两端要单独成层：本文的核心误差发现就是"两端误差大、中间准"，
    若把两端与中间混在一起抽，κ 会只反映中间段的共识（大纲四·（五）口径 3）。
    """
    edges = [float(e) for e in C.DIFFICULTY_BINS]
    k = int(C.HUMAN_KAPPA_STRATA)
    if k < 2:
        raise ValueError(f"C.HUMAN_KAPPA_STRATA={k} 非法：分层至少要 2 层")
    if k > len(edges) + 1:
        raise ValueError(
            f"C.HUMAN_KAPPA_STRATA={k} 超过可用档数 {len(edges) + 1}：分层不能细于等宽档"
        )
    idx = np.unique(np.linspace(0, len(edges) - 1, k - 1).round().astype(int))
    bounds = [edges[i] for i in idx]
    if len(bounds) != k - 1:
        raise ValueError(f"无法从 DIFFICULTY_BINS={edges} 派生 {k} 层的边界")
    return bounds


def stratum_labels() -> list[str]:
    """三层的显示名，含 rating 区间，例如 `中(1200~2400)`。"""
    k = int(C.HUMAN_KAPPA_STRATA)
    if len(_STRATUM_NAMES) != k:
        raise ValueError(
            f"_STRATUM_NAMES 有 {len(_STRATUM_NAMES)} 项，与 C.HUMAN_KAPPA_STRATA={k} 不符"
        )
    edges = [-np.inf] + stratum_bounds() + [np.inf]
    out: list[str] = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        lo_s = "−∞" if not np.isfinite(lo) else f"{lo:g}"
        hi_s = "∞" if not np.isfinite(hi) else f"{hi:g}"
        out.append(f"{_STRATUM_NAMES[i]}({lo_s}~{hi_s})")
    return out


def strata_of(rating: np.ndarray) -> np.ndarray:
    """rating -> 0..K-1 层号。

    ★ 必须与 `metrics.BinThresholds.apply` 用**同一分档约定**（左开右闭，
    即大纲二·（五）的"≤1200 / ≤1600 / …"），否则四·（五）"人类一致率与模型一致率
    对比"就不成立：rating 恰为 1200 的题会被人类实验归入"中层"、而被模型评估归入"第 1 档"。

    ⚠ 实测这个不一致是真会发生的：本任务 rating 以 100 为步长、分层边界（1200/2400）
    又是 100 的整数倍，391 题里有 63 题（16.1%）正压在切点上。
    早期本函数用 `right=False`，导致"低层"只有 88 题、而"第 1 档"有 105 题。
    """
    bounds = stratum_bounds()
    return np.digitize(np.asarray(rating, dtype=np.float64),
                       bounds, right=True).astype(int)


# ---------------------------------------------------------------- 评题表"盲"校验


def _is_forbidden_sheet_column(col: str) -> bool:
    c = str(col).strip().lower()
    if c in FORBIDDEN_SHEET_COLUMNS:
        return True
    return any(p in c for p in FORBIDDEN_SHEET_SUBSTRINGS)


def assert_no_leaky_columns(columns: Iterable[str], where: str) -> None:
    """强制"评题表不含任何难度/身份提示列"（大纲四·（五）口径 1）。

    为什么用断言而不是靠 code review：这张表一旦发出去就无法回收重来，
    而泄漏是**不可逆**的——评题人只要看见 rating，整批数据就作废。
    因此把校验放在写到磁盘之前的必经路径上，并在写盘之后**再读回来校验一次**
    （防止列名被 to_csv/Excel 悄悄改动）。
    """
    bad = [str(c) for c in columns if _is_forbidden_sheet_column(str(c))]
    if bad:
        raise AssertionError(
            f"{where} 出现禁止列 {bad}：评题人只能看到 题面+标签（与 D 格特征条件对齐），"
            f"任何难度提示或题目身份列都不允许出现在评题表里。"
        )


def assert_blind_sheet(columns: Iterable[str], where: str) -> None:
    """更严格的一档：列集合必须**恰好**等于 SHEET_COLUMNS（我们生成表时用）。"""
    assert_no_leaky_columns(columns, where)
    cols = [str(c) for c in columns]
    extra = [c for c in cols if c not in SHEET_COLUMNS]
    missing = [c for c in SHEET_COLUMNS if c not in cols]
    if extra or missing:
        raise AssertionError(
            f"{where} 的列集合与 SHEET_COLUMNS 不符：多出 {extra}，缺少 {missing}。"
            f"新增列前必须先确认它不提示难度、不提示题目身份。"
        )


# ---------------------------------------------------------------- 数据集读取


def load_working_frame(dataset_path: Path | str) -> pd.DataFrame:
    """读入数据集并派生本实验需要的列（复用项目的 load_dataset，不另起一套读取逻辑）。

    返回列：problem_key / year / rating / statement / tags / bin / stratum / is_new。
    档位与分层全部由 config 派生，脚本里不出现任何硬编码的档位边界。
    快照日期放在 `frame.attrs["snapshot_date"]`（正文必须注明数据快照日期）。

    ★ `statement` 列给的是**可读题面**，不是 `dataset.csv` 里给模型用的清洗文本。
    这是本实验的一处关键口径，原因如下：

    - `dataset.csv` 的 `statement` 是模型输入，经过小写化与 LaTeX 归一化，
      成分形如 `sum_ i=1 ^ n max (a_1, ..., a_i)`。人读这种文本比读原题面吃力得多，
      会让"人类定档"这一侧被无谓地削弱，从而**低估人类一致性**——
      偏偏这个方向的偏差会让"模型已达人类标注水平"的结论更容易成立，
      属于对本文有利的偏差，最应当避免。
    - 正确的做法是给评委**原始题面的可读版本**：保留大小写与 LaTeX 记法，
      但仍然删掉题号标题（A 题必简单，标题就是泄漏）与任何提交统计。
      即"信息条件与 D 格对齐，但呈现形式保持人类可读"。
    - 原始 HTML 由 `build_dataset.py` 缓存在 `data/raw/statements.jsonl`，
      因此这里可以随时重建可读题面，无需重新抓取。取不到时**回退到清洗文本并告警**，
      而不是静默使用——静默使用会让偏差无从察觉。
    """
    ds = D.load_dataset(dataset_path)
    tag_cols = [c for c in ds.group_columns("tag") if c.startswith(C.TAG_MULTIHOT_PREFIX)]
    if not tag_cols:
        raise ValueError(
            f"数据集中没有以 {C.TAG_MULTIHOT_PREFIX} 开头的算法标签列，"
            f"评题表就没有标签可给，实验口径不成立"
        )
    tag_names = [c[len(C.TAG_MULTIHOT_PREFIX):].replace("_", " ") for c in tag_cols]
    tag_matrix = ds.X[tag_cols].to_numpy(dtype=np.float64)
    tags = ["|".join(n for n, v in zip(tag_names, row) if v > 0) for row in tag_matrix]

    problem_keys = ds.meta["problem_key"].astype(str).to_numpy()
    statements, n_readable = readable_statements(problem_keys, ds.text.to_numpy())

    frame = pd.DataFrame({
        "problem_key": problem_keys,
        "year": ds.meta["year"].to_numpy(),
        "rating": np.asarray(ds.y, dtype=np.float64),
        "statement": statements,
        "tags": tags,
    })
    if frame["rating"].isna().any():
        raise ValueError("存在缺失 rating 的题目，人类定档实验无法给出对照标签")
    frame["bin"] = rating_to_bin(frame["rating"].to_numpy())
    frame["stratum"] = strata_of(frame["rating"].to_numpy())
    frame["is_new"] = (frame["year"].to_numpy() >= C.TEMPORAL_TEST_START_YEAR).astype(int)
    frame.attrs["snapshot_date"] = ds.snapshot_date
    frame.attrs["n_readable_statement"] = n_readable

    log(f"数据集 {len(frame)} 题；等宽档分布 "
        f"{np.bincount(frame['bin'].to_numpy(), minlength=n_bins())[1:].tolist()}；"
        f"新题（year ≥ {C.TEMPORAL_TEST_START_YEAR}）{int(frame['is_new'].sum())} 题")
    log(f"  可读题面 {n_readable}/{len(frame)} 题"
        + ("" if n_readable == len(frame)
           else "（其余回退为模型的清洗文本，需在正文说明）"))
    for i, lab in enumerate(stratum_labels()):
        sub = frame[frame["stratum"] == i]
        log(f"  分层 {lab}: 共 {len(sub)} 题（新 {int(sub['is_new'].sum())} / "
            f"旧 {int((1 - sub['is_new']).sum())}）")
    return frame


def humanize_math(text: str) -> str:
    """把 LaTeX 源码转成人类可读的数学记号。

    原始题面里的公式是 LaTeX 源码（`\\sum_{i=1}^{n}`、`\\le`、`a_i`）。
    评委读源码既慢又容易看错下标，会无谓地压低人类一致性——
    而"人类一致性偏低"恰好会支持"模型未达人类水平"或"任务本身模糊"两种论述中的后者，
    任何方向的无谓偏差都应消除。因此做一次**只面向展示、不改变语义**的替换：

    - `_{...}` / `^{...}` → 下标/上标用 Unicode 或直接展平（本表是 CSV，无法排版真上下标）
    - 常用希腊字母与运算符 → Unicode 符号
    - 多余的转义空格与花括号清理

    ⚠ 注意这与 `vectorizers.clean_statement` 的 LaTeX 归一化**目的不同**：
    后者是给模型的（追求一致性与去噪），本函数是给人类的（追求可读）。
    二者不可互相替代，也不应共用实现。

    ⚠⚠ **替换必须按"整条命令"匹配，不能按子串**（实测踩过）：
    早期实现是 `for k, v in repl.items(): s = s.replace(k, v)`，
    按字典顺序逐个做**子串**替换，于是短的命令会吃掉长的命令的前缀——
    `\\leq` 变成 `≤q`、`\\geq` 变成 `≥q`、`\\neq` 变成 `≠q`、
    `\\infty` 变成 `∈fty`、`\\int` 变成 `∈t`、`\\texttt{...}` 变成 `tt{...}`。
    这些错误不会留下反斜杠，所以"残留 LaTeX 0%"的自检**照样通过**，
    题面看起来只是"有一点点怪"，很难发现。
    现改为用 `\\[A-Za-z]+` 一次性切出整条命令再查表，未登记的整条命令转空格。
    """
    if not isinstance(text, str) or not text:
        return text or ""
    s = text

    # ⚠ 先处理**转义字面量**（`\{` `\}` `\%` `\$` `\_` `\|` `\&` `\#`）。
    # 它们不是命令，`\[A-Za-z]+` 匹配不到，若不先还原，后面剥花括号时会留下
    # 半个反斜杠——实测题面里 `\in \{L, R\}` 会渲染成 `∈ , R\;`，很难看。
    s = re.sub(r"\\([{}|%$&_#])", r"\1", s)

    # 下标/上标：CSV 里没有富文本，展平成 a_i 这种 A 界可读形式
    s = re.sub(r"([A-Za-z0-9\)\]])_\{([^{}]{1,24})\}", r"\1_\2", s)
    s = re.sub(r"([A-Za-z0-9\)\]])_([A-Za-z0-9])", r"\1_\2", s)
    s = re.sub(r"([A-Za-z0-9\)\]])\^\{([^{}]{1,24})\}", r"\1^\2", s)
    s = re.sub(r"([A-Za-z0-9\)\]])\^([A-Za-z0-9])", r"\1^\2", s)

    # 常见运算符与关系符号 → Unicode（人类一眼可读）
    repl = {
        r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
        r"\lt": "<", r"\gt": ">", r"\neq": "≠", r"\ne": "≠",
        r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±",
        r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯",
        r"\rightarrow": "→", r"\to": "→", r"\Rightarrow": "⇒",
        r"\infty": "∞", r"\in": "∈", r"\notin": "∉",
        r"\subset": "⊂", r"\subseteq": "⊆", r"\cup": "∪", r"\cap": "∩",
        r"\forall": "∀", r"\exists": "∃", r"\sum": "Σ", r"\prod": "Π",
        r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
        r"\epsilon": "ε", r"\varepsilon": "ε", r"\theta": "θ", r"\lambda": "λ",
        r"\mu": "μ", r"\pi": "π", r"\sigma": "σ", r"\phi": "φ", r"\omega": "ω",
        r"\Delta": "Δ", r"\Omega": "Ω", r"\Sigma": "Σ",
        r"\operatorname{max}": "max", r"\operatorname{min}": "min",
        r"\operatorname{gcd}": "gcd", r"\operatorname{lcm}": "lcm",
        r"\max": "max", r"\min": "min", r"\gcd": "gcd",
        r"\bmod": "mod", r"\pmod": "mod", r"\texttt": "", r"\text": "",
        # 第二批：实测残留最多的命令（评委读到 \dagger、\frac 这类会卡住）
        r"\dagger": "†", r"\ddagger": "‡", r"\oplus": "⊕", r"\otimes": "⊗",
        r"\mathbf": "", r"\mathit": "", r"\mathrm": "", r"\mathtt": "",
        r"\mathcal": "", r"\operatorname": "", r"\limits": "",
        r"\color": "", r"\colorbox": "", r"\hspace": " ", r"\quad": " ",
        r"\qquad": "  ", r"\left": "", r"\right": "", r"\big": "", r"\Big": "",
        r"\displaystyle": "", r"\overline": "", r"\underline": "",
        r"\pi": "π", r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
        r"\iota": "ι", r"\kappa": "κ", r"\nu": "ν", r"\xi": "ξ",
        r"\rho": "ρ", r"\tau": "τ", r"\upsilon": "υ", r"\chi": "χ",
        r"\psi": "ψ", r"\Gamma": "Γ", r"\Theta": "Θ", r"\Lambda": "Λ",
        r"\Xi": "Ξ", r"\Pi": "Π", r"\Phi": "Φ", r"\Psi": "Ψ",
        # 积分/对数等：不加会被"整条未登记命令转空格"删成一个空档
        r"\int": "∫", r"\iint": "∬", r"\oint": "∮", r"\log": "log",
        r"\ln": "ln", r"\exp": "exp", r"\sin": "sin", r"\cos": "cos",
        r"\tan": "tan", r"\min": "min", r"\max": "max",
        r"\equiv": "≡", r"\approx": "≈", r"\sim": "~", r"\propto": "∝",
        r"\mid": "|", r"\nmid": "∤", r"\ast": "*", r"\star": "*",
        r"\mathrm": "", r"\textbf": "", r"\textit": "",
    }
    # 先处理带花括号参数的命令（这些不是"纯命令"，必须在整条命令正则之前做完）
    for k in (r"\operatorname{max}", r"\operatorname{min}",
              r"\operatorname{gcd}", r"\operatorname{lcm}"):
        s = s.replace(k, repl[k])

    # `\frac{a}{b}` 这类分数：CSV 无法排版真分数，转成 a/b 形式。
    # ⚠ 必须排在"整条命令替换"之前：否则 `\frac` 会先被当成未登记命令删掉，
    # 只剩 `{a}{b}`，再去掉花括号就变成 `ab`（实测踩过）。
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])", r"\1/\2", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"√\1", s)

    # 再按**整条命令**替换：`\[A-Za-z]+` 一次吃下 \leq 而不是先吃下 \le
    s = re.sub(r"\\[A-Za-z]+", lambda m: repl.get(m.group(0), " "), s)
    # LaTeX 的间距命令（`\,` `\;` `\!` `\ `）与漏网的符号命令一并转空格
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\[^A-Za-z0-9\s]", " ", s)

    # 清理残余：花括号、`$$$` 公式定界符、多余空格
    s = s.replace("$$$", " ")
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)      # 仍未识别的命令直接去掉，避免评委看到源码
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def readable_statements(problem_keys: Sequence[str],
                        fallback: Sequence[str]) -> tuple[list[str], int]:
    """由原始 HTML 重建**人类可读**题面；取不到时回退到清洗文本。

    可读 = 保留大小写、公式转成人类可读记号，但删掉题号标题、样例与提示段。
    这些由 `fetchers.extract_statement_parts` 负责——它按 CSS 类逐节切片，
    已经剔除了 header（标题/时限/内存限制）、sample-tests 与 note。

    ⚠ **输出规范（output_spec）必须一并给出**：早先只拼了 preamble + input_spec，
    而训练侧模型读到的 `raw_html` 是包含输出规范的完整题面。
    少一段不只是"少点信息"，而是让"大模型 vs 训练模型"的比较在输入上就不对等，
    属于对本节结论有实质影响的口径错误（实测校正）。
    """
    raw = _load_raw_html_index()
    out: list[str] = []
    n_ok = 0
    for i, k in enumerate(problem_keys):
        html = raw.get(str(k))
        if html:
            try:
                parts = F.extract_statement_parts(html)
                text = "\n".join(
                    x for x in (parts["preamble"], parts["input_spec"],
                                parts["output_spec"]) if x
                )
                if text.strip():
                    out.append(humanize_math(text))
                    n_ok += 1
                    continue
            except Exception:
                pass
        out.append(str(fallback[i]))
    return out, n_ok


#: 原始 HTML 索引缓存（进程内，避免重复读 500 MB 的 jsonl）
_RAW_HTML_CACHE: dict[str, str] = {}


def _load_raw_html_index() -> dict[str, str]:
    global _RAW_HTML_CACHE
    if _RAW_HTML_CACHE:
        return _RAW_HTML_CACHE
    path = C.RAW_DIR / "statements.jsonl"
    if not path.exists():
        log(f"  [warn] 未找到原始题面缓存 {path}，将回退为模型的清洗文本；"
            f"这会让评题人读到小写化文本，请先运行 build_dataset.py")
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok") and r.get("raw_html"):
                _RAW_HTML_CACHE[r["problem_key"]] = r["raw_html"]
    return _RAW_HTML_CACHE


# ---------------------------------------------------------------- 配额分配


def allocate_even(n_total: int, capacities: Sequence[int]) -> list[int]:
    """把 n_total 尽可能**均匀**地分到容量受限的各格（水填式，触顶的格让给其他格）。

    为什么不用按总体比例的配额：分层抽样的目的就是**不**按总体比例，
    而是让低/中/高各占约 1/3，否则 κ 会被中间难度段主导（大纲四·（五）口径 3）。
    """
    caps = [int(c) for c in capacities]
    alloc = [0] * len(caps)
    remaining = int(n_total)
    if remaining > sum(caps):
        raise ValueError(f"可用题量 {sum(caps)} 少于需要的 {remaining}")
    while remaining > 0:
        active = [i for i in range(len(caps)) if alloc[i] < caps[i]]
        if not active:
            raise ValueError("配额分配失败：所有格都已触顶")
        base, extra = divmod(remaining, len(active))
        if base == 0:
            for i in active[:remaining]:
                alloc[i] += 1
            break
        for k, i in enumerate(active):
            alloc[i] += min(base + (1 if k < extra else 0), caps[i] - alloc[i])
        new_remaining = n_total - sum(alloc)
        if new_remaining == remaining:      # 兜底防死循环（理论上不可达）
            break
        remaining = new_remaining
    return alloc


def split_new_old(quota: int, n_new: int, n_old: int, new_frac: float) -> tuple[int, int]:
    """把一个层的配额在"新题/旧题"之间分配。

    为什么要显式控制新旧比例而不按总体比例（本数据集中新题只占约 1/4）：
    ① 大纲要求"新题与旧题混合"，两个群体的占比必须足以分别报告；
    ② 只有 year ≥ TEMPORAL_TEST_START_YEAR 的题落在 D 格的测试集里（新题 = 测试集），
       新题抽得太少，"人类 vs 模型"的可比样本就不够（compare 子命令会吃亏）。
    实际比例会写进报告，绝不偷偷加权。
    """
    if not 0.0 <= new_frac <= 1.0:
        raise ValueError(f"--new-frac={new_frac} 非法，应在 [0, 1] 内")
    want_new = int(round(quota * new_frac))
    want_new = max(0, min(want_new, n_new, quota))
    want_old = quota - want_new
    if want_old > n_old:                      # 旧题不够，缺口让给新题
        want_old = n_old
        want_new = min(quota - want_old, n_new)
    if want_new + want_old != quota:
        raise ValueError(f"该层配额 {quota} 无法在 新{n_new}/旧{n_old} 之间满足")
    return want_new, want_old


# ---------------------------------------------------------------- sample


def _sheets_have_data(paths: Sequence[Path]) -> list[Path]:
    """检查评题表里是否已经有人填过（防止重跑 sample 覆盖已回收的答卷）。"""
    filled: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        except Exception:
            filled.append(p)
            continue
        if "difficulty_1to5" in df.columns and df["difficulty_1to5"].notna().any():
            filled.append(p)
    return filled


def build_instructions(n_raters: int, seed: int, n_items: int) -> str:
    """评题人操作说明。**只含档位定义与操作规则，不含任何题目信息**。"""
    if len(_BIN_HINTS) != n_bins():
        raise ValueError(
            f"_BIN_HINTS 有 {len(_BIN_HINTS)} 项，与分档数 {n_bins()} 不符——"
            f"config.DIFFICULTY_BINS 改了以后必须同步更新档位描述"
        )
    rows = "\n".join(
        f"| {i + 1} | {bin_interval_text(i)} | {_BIN_HINTS[i]} |" for i in range(n_bins())
    )
    return f"""# 算法题难度定档 · 评题人操作说明

> 本说明随评题表一并发出。请先读完再开始填表，全程约需 60～90 分钟。

## 一、这是什么实验

我们在研究"能否用题面和算法标签自动预测一道算法题的难度"。要判断模型是否
**已经达到人类标注水平**，就必须先知道**人类自己**在同一任务上能一致到什么程度。
因此请您**独立**对 {n_items} 道题各定一个 1～5 的难度档。

## 二、对评题人的要求（请先确认自己符合）

- **必须有算法竞赛经历**（ACM/ICPC、NOI/NOIP、Codeforces、洛谷等均可），
  因为本实验估计的是该任务的"**人类可达上界**"，需要熟悉竞赛难度体系的判断者；
- ⚠ 这与论文另一处"面向高职学生的教学阈值重校准"所需的评委（**熟悉高职学生
  水平的任课教师**）是**两拨人、两套样本、两个结论**，不可互相替代、不得混用。
  本表请您按**竞赛难度体系**定档，**不要**按您所教学生的水平来折算。

## 三、档位定义（请严格按此标准）

rating 指 Codeforces 官方给该题标注的难度分值（步长 100，最低 800）：

| 档位 | 对应 rating 区间 | 难度感觉 |
|---|---|---|
{rows}

> 区间为**左开右闭**（例如 rating 恰好 1200 的题属于第 1 档、恰好 1600 的属于第 2 档），
> 这与模型评估使用的分箱口径完全一致。请严格按此标准定档——
> rating 以 100 为步长，本就常有题卡在切点上，口径对不齐会让后续一致性对比失去意义。

## 四、你会看到什么、不会看到什么

评题表只有四类信息：

1. `item_id`：打乱后的编号（**与题目真实身份无关**，不要试图推断题目来源）；
2. `statement`：题面全文；
3. `tags`：该题的算法标签（如 `dp|graphs`）；
4. `difficulty_1to5` 与 `comment`：留给你填写。

⚠ **我们刻意不提供任何提交统计**（通过人数、通过率、上线时间等），也不要自己去
Codeforces / 洛谷等网站查这些题。原因：模型是在"**没有提交统计**"的条件下做预测的
（新题上线时本来就没有这些数据），如果人能看到提交统计，人就拿到了模型没有的信息，
两边的一致率就不可比了，整个实验也就失去意义。**请在不查询的状态下完成。**

## 五、操作步骤

1. 打开发给你的那份 `rating_sheet_rater*.csv`（用 Excel/WPS 或任何表格软件；
   填完直接另存为 CSV，**不要改动列名**）；
2. 逐行填写 `difficulty_1to5`：只能填 **1、2、3、4、5** 这五个整数；
3. `comment` 列可选填（例如"题面太短无法判断"），不影响统计；
4. **确实无法判断的题可以留空**——留空会被如实计入有效样本量，不会被当作 0 或用
   众数填补；但请把留空控制在少数（建议 ≤ 10 题），否则有效样本量不足；
5. 填完后把整份 CSV 原样发回，**不要合并成汇总表**。

## 六、独立性要求（关键）

- 请**独立完成，不要与其他评题人讨论**任何一道题的档位；
- 几位评题人的评题表题目编号完全一致，但**内容互不可见**；
- 请一次性连续完成，中途不要参考其他人已填的结果；
- 不要打开 `items.csv` / `key.csv`（那是研究者留档与答案钥匙，
  打开会使本实验作废）。

## 七、节奏建议

建议每道题 30～60 秒，先快速过一遍、再回头修正明显不合理的地方。

---

> 抽样信息（供复核，与题目内容无关）：总题数 {n_items}，评题人 {n_raters} 人，
> 按 rating 分层 + 新题旧题混合抽取，随机种子 {seed}（可复现）。
"""


def cmd_sample(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    n_items = int(args.n_items)
    n_raters = int(args.n_raters)

    sheet_paths = [out_dir / f"rating_sheet_rater{i + 1}.csv" for i in range(n_raters)]
    filled = _sheets_have_data(sheet_paths)
    if filled and not args.force:
        raise SystemExit(
            "检测到已有填写内容的评题表：\n  "
            + "\n  ".join(str(p) for p in filled)
            + "\n重跑 sample 会覆盖已回收的答卷。确认要重抽请加 --force。"
        )

    frame = load_working_frame(args.dataset)
    if n_items > len(frame):
        raise SystemExit(f"需要 {n_items} 题，数据集只有 {len(frame)} 题")

    # ---- 1) 分层配额：低/中/高尽可能等分，容量不足的层让给其他层
    labels = stratum_labels()
    caps = [int((frame["stratum"] == i).sum()) for i in range(len(labels))]
    quotas = allocate_even(n_items, caps)
    log(f"分层抽样：目标 {n_items} 题；各层容量 {dict(zip(labels, caps))} → "
        f"配额 {dict(zip(labels, quotas))}")

    # ---- 2) 层内再按新/旧分配，然后在该格内无放回随机抽。
    # 每格用 SeedSequence([seed, stratum, era]) 派生独立随机流：即使将来改变层的
    # 遍历顺序，各格抽中的题目也不会连带变化（可复现性更稳）。
    picks: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    for s in range(len(labels)):
        sub = frame[frame["stratum"] == s]
        n_new = int(sub["is_new"].sum())
        n_old = int(len(sub) - n_new)
        want_new, want_old = split_new_old(int(quotas[s]), n_new, n_old, float(args.new_frac))
        for era_i, (want, era_name) in enumerate(((want_new, "新"), (want_old, "旧"))):
            if want == 0:
                continue
            pool = sub[sub["is_new"] == (1 - era_i)]
            if len(pool) < want:
                raise SystemExit(
                    f"分层 {labels[s]} 的{era_name}题只有 {len(pool)} 道，少于配额 {want}"
                )
            rng = np.random.default_rng(np.random.SeedSequence([seed, s, era_i]))
            idx = np.sort(rng.choice(len(pool), size=want, replace=False))
            chunk = pool.iloc[idx].copy()
            chunk["stratum_label"] = labels[s]
            picks.append(chunk)
            cell_rows.append({
                "stratum": labels[s], "stratum_id": s, "era": era_name,
                "capacity": int(len(pool)), "quota": int(want),
            })
            log(f"  层 {labels[s]} · {era_name}题：可用 {len(pool)} → 抽 {want}")

    sample = pd.concat(picks, ignore_index=True)
    if len(sample) != n_items:
        raise SystemExit(f"抽样数 {len(sample)} 与目标 {n_items} 不一致")
    if sample["problem_key"].duplicated().any():
        raise SystemExit("抽样出现重复题目，抽样逻辑有误")

    # ---- 3) 打乱 item_id：评题人看到的顺序与难度无关，避免"越往后越难"的锚定效应。
    # 用独立的随机流（种子加常量偏移），保证与抽取过程解耦：改这里不会改抽中的题。
    order = np.random.default_rng(np.random.SeedSequence([seed, 0x9E3779B9])).permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample["item_id"] = [f"K{i + 1:03d}" for i in range(len(sample))]

    # ---- 4) 写 items.csv（研究者留档：含 rating 与题目身份，**不得分发**）
    items = sample[[
        "item_id", "problem_key", "rating", "year", "stratum", "stratum_label", "is_new",
        "statement", "tags",
    ]].rename(columns={"stratum": "stratum_id", "stratum_label": "stratum"})
    items = items.sort_values(["stratum_id", "rating", "problem_key"]).drop(
        columns=["stratum_id"]).reset_index(drop=True)
    items["sample_seed"] = seed
    items_path = out_dir / ITEMS_FILE
    items.to_csv(items_path, index=False, encoding="utf-8-sig")

    # ---- 5) 写 key.csv（研究者专用钥匙，带醒目警告注释）
    key = sample[["item_id", "problem_key", "rating", "year", "stratum_label", "is_new"]].rename(
        columns={"stratum_label": "stratum"}
    ).sort_values("item_id").reset_index(drop=True)
    key_path = out_dir / KEY_FILE
    with key_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(
            "# ⚠⚠⚠ 答案钥匙（研究者专用）——绝对不能交给评题人 ⚠⚠⚠\n"
            "# 本文件把 item_id 映射回真实题目身份与真实 rating。\n"
            "# 一旦评题人看到它，“只能看题面+标签”的实验口径立即失效，整批标注作废。\n"
            "# 请勿随评题表一起分发；等评题表全部回收后再打开本文件计算一致性。\n"
            f"# 抽样种子 seed={seed}（items.csv 的 sample_seed 列同值，可复现抽样）。\n"
        )
        key.to_csv(fh, index=False)

    # ---- 6) 写每位评题人的独立评题表（只有 item_id/statement/tags + 两个空列）
    by_id = sample.set_index("item_id")
    sheet_records: list[dict[str, Any]] = []
    for r, path in enumerate(sheet_paths, start=1):
        sheet = pd.DataFrame({
            "item_id": key["item_id"].to_numpy(),
            "statement": by_id.loc[key["item_id"], "statement"].to_numpy(),
            "tags": by_id.loc[key["item_id"], "tags"].to_numpy(),
            "difficulty_1to5": "",
            "comment": "",
        })
        assert_blind_sheet(sheet.columns, f"评题表 rater{r}")          # 写盘前校验
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        # 写盘后再从磁盘读回表头校验一次：列名若被中间环节改动则立刻失败
        header = [str(c) for c in pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns]
        assert_blind_sheet(header, f"评题表 rater{r}（回读）")
        sheet_records.append({
            "rater": f"rater{r}", "path": str(path), "n_items": int(len(sheet)),
            "columns": header,
        })
        log(f"  评题表 rater{r} -> {path.name}（{len(sheet)} 行，列：{header}）")

    # ---- 7) 评题说明 + 抽样元信息
    instr_path = out_dir / INSTRUCTIONS_FILE
    instr_path.write_text(build_instructions(n_raters, seed, n_items), encoding="utf-8")

    meta = {
        "generated_at": now_iso(),
        "seed": seed,
        "n_items": n_items,
        "n_raters": n_raters,
        "new_frac_requested": float(args.new_frac),
        "dataset_path": str(Path(args.dataset)),
        "snapshot_date": frame.attrs.get("snapshot_date"),
        "stratum_bounds": stratum_bounds(),
        "stratum_labels": labels,
        "bin_edges": list(C.DIFFICULTY_BINS),
        "bin_interval_text": [bin_interval_text(i) for i in range(n_bins())],
        "temporal_test_start_year": C.TEMPORAL_TEST_START_YEAR,
        "cells": cell_rows,
        "composition": {
            "by_stratum": {lab: int((items["stratum"] == lab).sum()) for lab in labels},
            "by_era": {"新题": int(items["is_new"].sum()),
                       "旧题": int((1 - items["is_new"]).sum())},
            "by_stratum_era": {
                f"{lab}|{'新' if n == 1 else '旧'}":
                    int(((items["stratum"] == lab) & (items["is_new"] == n)).sum())
                for lab in labels for n in (1, 0)
            },
            "by_rating_bin": {
                f"{i + 1}档({bin_interval_text(i)})":
                    int((rating_to_bin(items["rating"].to_numpy()) == i + 1).sum())
                for i in range(n_bins())
            },
        },
        "sheet_columns": list(SHEET_COLUMNS),
        "forbidden_sheet_columns": list(FORBIDDEN_SHEET_COLUMNS),
        "files": {"items": str(items_path), "key": str(key_path),
                  "instructions": str(instr_path), "sheets": sheet_records},
    }
    (out_dir / META_FILE).write_text(
        json.dumps(jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 8) 抽样构成汇报（正文方法一节的数字来源）
    new_n = int(sample["is_new"].sum())
    log(f"抽样完成：{len(sample)} 题，种子 {seed}")
    log("  三层分布：" + "，".join(
        f"{lab} {int((sample['stratum'] == i).sum())}" for i, lab in enumerate(labels)))
    log(f"  新题（year ≥ {C.TEMPORAL_TEST_START_YEAR}）{new_n} 题（{new_n / len(sample):.1%}）"
        f" / 旧题 {len(sample) - new_n} 题（{(len(sample) - new_n) / len(sample):.1%}）")
    log("  等宽档分布：" + "，".join(
        f"{i + 1}档 {int((rating_to_bin(sample['rating'].to_numpy()) == i + 1).sum())}"
        for i in range(n_bins())))
    log(f"  输出目录：{out_dir}")
    log(f"  ⚠ {items_path.name} 与 {key_path.name} 含真实 rating/题目身份，"
        f"只能发给评题人的是 {SHEET_PATTERN}")
    for rec in sheet_records:
        log(f"  列校验通过（不含任何难度/身份提示列）：{rec['rater']} -> {rec['columns']}")
    return 0


# ---------------------------------------------------------------- κ 计算


def fleiss_kappa_manual(counts: np.ndarray) -> float:
    """自实现 Fleiss' κ（用于与 statsmodels 交叉验证）。

    P_i = (Σ_j n_ij² − n) / (n(n−1))：第 i 题上评题人两两一致的比例；
    P̄ = mean(P_i)；p_j = 第 j 档的总体占比；P_e = Σ_j p_j²；
    κ = (P̄ − P_e) / (1 − P_e)。

    ⚠ 前提是每个被试的评分人数相同（n 固定）。本脚本只在"所有评题人都评过"
    的完整子集上调用，因此前提成立；有人缺评就先把该题整题剔除，
    而不是把缺评当 0 或众数填补（大纲：不得用众数填补后当作完整数据）。
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts 必须是 n_items × n_categories 的二维数组")
    n_items = counts.shape[0]
    if n_items == 0:
        raise ValueError("有效样本量为 0，无法计算 Fleiss κ")
    n_per_item = counts.sum(axis=1)
    if not np.allclose(n_per_item, n_per_item[0]):
        raise ValueError(
            f"Fleiss κ 要求每题评分人数相同，实际为 {sorted(set(n_per_item.tolist()))}"
        )
    n = float(n_per_item[0])
    if n < 2:
        raise ValueError(f"每题至少要有 2 个评分才能算一致性，实际 {n}")
    p_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1.0))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = float(np.sum(p_j ** 2))
    if abs(1.0 - p_e) < 1e-12:
        # 所有人只用了同一档 → 期望一致率 = 1，κ 定义上退化；此时一致即 1
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def fleiss_kappa_statsmodels(counts: np.ndarray) -> float:
    """statsmodels 版本，用于与自实现交叉验证（两者必须一致，不一致要报错）。"""
    from statsmodels.stats.inter_rater import fleiss_kappa
    return float(fleiss_kappa(np.asarray(counts, dtype=np.float64), method="fleiss"))


def krippendorff_alpha(mat: np.ndarray, level: str = "ordinal",
                       n_cats: int | None = None) -> float:
    """Krippendorff's α（允许缺失值），有序等级用 ordinal 差异函数。

    与 κ 的区别：α 用 coincidence matrix 对每题实际评分数 m_u 做 1/(m_u−1) 加权，
    因此天然支持"评分人数不齐/有缺失"的情形，可作为 κ（完整子集口径）的稳健性佐证。
    ⚠ α 与 Cohen 的 QWK **不相等**（期望不一致的边际构造不同：α 用合并边际且带
    n(n−1) 修正），只在两评题人边际分布相同且样本量大时接近，报告中不得写成同一指标。
    """
    mat = np.asarray(mat, dtype=np.float64)
    valid = mat[~np.isnan(mat)]
    if valid.size == 0:
        return float("nan")
    if n_cats is None:
        n_cats = int(np.nanmax(mat))
    if not np.allclose(valid, np.round(valid)):
        raise ValueError("Krippendorff α 需要整数档位")
    if valid.min() < 1 or valid.max() > n_cats:
        raise ValueError(f"档位超出 1..{n_cats} 范围")

    # 一致性矩阵：对每题内部的**有序位置对**(i≠j) 各计 1/(m_u−1)，
    # 因此对角线也会被填充（同一档的成对重合也算一致性）。
    o = np.zeros((n_cats, n_cats), dtype=np.float64)
    for row in mat:
        vals = row[~np.isnan(row)]
        m_u = len(vals)
        if m_u < 2:
            continue
        w = 1.0 / (m_u - 1.0)
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                o[int(vals[i]) - 1, int(vals[j]) - 1] += w
    n_total = float(o.sum())
    if n_total <= 1.0:
        return float("nan")
    n_c = o.sum(axis=1)          # 各档边际（含对角）

    if level == "nominal":
        delta2 = np.ones((n_cats, n_cats)) - np.eye(n_cats)
    elif level == "interval":
        idx = np.arange(n_cats, dtype=np.float64)
        delta2 = (idx[:, None] - idx[None, :]) ** 2
    elif level == "ordinal":
        delta2 = np.zeros((n_cats, n_cats), dtype=np.float64)
        for c in range(n_cats):
            for k in range(n_cats):
                lo, hi = min(c, k), max(c, k)
                s = float(n_c[lo:hi + 1].sum()) - (n_c[lo] + n_c[hi]) / 2.0
                delta2[c, k] = s ** 2
    else:
        raise ValueError(f"未知差异口径 {level!r}")

    d_o = float((o * delta2).sum() / n_total)
    d_e = float((np.outer(n_c, n_c) * delta2).sum() / (n_total * (n_total - 1.0)))
    if abs(d_e) < 1e-15:
        return 1.0 if abs(d_o) < 1e-15 else float("nan")
    return 1.0 - d_o / d_e


def safe_kappa(a: np.ndarray, b: np.ndarray, weights: str | None = None) -> float:
    """`sklearn.cohen_kappa_score` 的退化情形包装。

    为什么必须包装：sklearn 1.9 起对两类退化输入返回 **NaN + warning**，
    直接拿来当数字写进论文会静默变成"缺失值"——
      ① 双方在共同样本上只有一个取值（例如 3 人全给 3 档）：
         κ 在定义上应为 1.0（完全一致），不是缺失；
      ② 双方标签集合完全不相交（例如人类全给 1~2 档、模型全给 4~5 档）：
         κ 确实无定义，如实返回 NaN，绝不能随手填一个数。
    """
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(np.unique(np.concatenate([a, b]))) == 1:
        return 1.0
    if len(np.intersect1d(np.unique(a), np.unique(b))) == 0:
        return float("nan")
    return float(cohen_kappa_score(a, b, weights=weights))


def pairwise_kappa(mat: pd.DataFrame) -> list[dict[str, Any]]:
    """两两配对：Cohen κ（未加权）+ 二次加权 κ（QWK）+ 原始/±1 一致率。

    为什么 QWK 必须与未加权 κ 并列：难度 1~5 是**有序**等级，未加权 κ 把
    "1 vs 2"与"1 vs 5"当成同等程度的错误，会系统性低估人类之间的真实共识；
    本文正文的分类指标同样是 QWK，口径必须一致。
    配对计算只用两位评题人**都评过**的题（成对删除），有效 N 逐对报告。
    """
    out: list[dict[str, Any]] = []
    raters = list(mat.columns)
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a = mat[raters[i]].to_numpy(dtype=np.float64)
            b = mat[raters[j]].to_numpy(dtype=np.float64)
            m = ~(np.isnan(a) | np.isnan(b))
            n = int(m.sum())
            rec: dict[str, Any] = {
                "pair": f"{raters[i]}-{raters[j]}", "n": n, "cohen_kappa": None,
                "qwk": None, "raw_agreement": None, "within1_agreement": None,
            }
            if n >= 2:
                rec["cohen_kappa"] = safe_kappa(a[m], b[m])
                rec["qwk"] = safe_kappa(a[m], b[m], weights="quadratic")
                rec["raw_agreement"] = float(np.mean(a[m] == b[m]))
                rec["within1_agreement"] = float(np.mean(np.abs(a[m] - b[m]) <= 1))
            out.append(rec)
    return out


def consensus_of(mat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """人类共识档：多数投票；出现并列（无唯一众数）时记为分歧。

    返回 (consensus, is_tie, n_tie)。并列题在 compare 里必须**排除**并如实报告
    ——把并列硬指派给某一档会凭空抬高或压低模型一致率。
    """
    vals = mat.to_numpy(dtype=np.float64)
    n = vals.shape[0]
    cons = np.full(n, np.nan)
    tie = np.zeros(n, dtype=bool)
    for i in range(n):
        row = vals[i][~np.isnan(vals[i])]
        if row.size == 0:
            continue
        cnt = Counter(row.tolist())
        top = max(cnt.values())
        winners = [v for v, c in cnt.items() if c == top]
        if len(winners) == 1:
            cons[i] = winners[0]
        else:
            tie[i] = True
    return cons, tie, int(tie.sum())


# ---------------------------------------------------------------- 评题表读取


def load_key(path: Path) -> pd.DataFrame:
    """读 key.csv（跳过 `#` 注释行），并校验必需列齐全。"""
    if not Path(path).exists():
        raise SystemExit(f"未找到 {path}；请先运行：python scripts/human_kappa.py sample")
    key = pd.read_csv(path, encoding="utf-8-sig", comment="#")
    need = ["item_id", "problem_key", "rating", "year"]
    missing = [c for c in need if c not in key.columns]
    if missing:
        raise SystemExit(f"{path} 缺列 {missing}")
    if key["item_id"].duplicated().any():
        raise SystemExit(f"{path} 的 item_id 有重复")
    return key


def resolve_sheets(patterns: Sequence[str] | None, out_dir: Path) -> list[Path]:
    """确定评题表路径：默认在输出目录里找 `rating_sheet_rater*.csv`。"""
    if not patterns:
        found = sorted(out_dir.glob(SHEET_PATTERN))
    else:
        found = []
        for pat in patterns:
            hits = (sorted(Path(p) for p in _glob.glob(pat))
                    if any(ch in pat for ch in "*?[") else [Path(pat)])
            for h in hits:
                if not h.exists():
                    raise SystemExit(f"评题表不存在：{h}")
                found.append(h)
        found = sorted(dict.fromkeys(found))
    if not found:
        raise SystemExit(
            f"没有找到任何评题表（{out_dir / SHEET_PATTERN}）。"
            f"请先运行 sample 并把表发给评题人填写。"
        )
    return found


def _parse_difficulty(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """解析 difficulty_1to5：返回 (数值, 非法标记)。

    非法 = 非空但既不是整数也不是我们认可的留空写法（如"不确定"）。
    这类值必须报错而不是静默当缺失：静默处理会让有效样本量虚高，
    而"有效样本量"是本实验正文必须报告的数字。
    """
    raw = series.astype("string").str.strip()
    blank = raw.isna() | raw.str.lower().isin(_BLANK_MARKERS)
    num = pd.to_numeric(raw.where(~blank), errors="coerce")
    return num.astype(np.float64), (~blank) & num.isna()


def load_rating_matrix(sheet_paths: Sequence[Path], item_ids: Sequence[str]
                       ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """读回填好的评题表 → items × raters 的档位矩阵（未填 = NaN）。

    缺失值处理口径（大纲明确要求）：**只对所有人都评了的题计算**，
    并报告有效样本量；绝不用众数/均值填补后当作完整数据。
    """
    mat = pd.DataFrame(index=[str(i) for i in item_ids])
    infos: list[dict[str, Any]] = []
    for p in sheet_paths:
        df = pd.read_csv(p, encoding="utf-8-sig", comment="#", dtype=str)
        # 回收回来的表同样要查泄漏列：若评题人交回的是 items.csv（含 rating），
        # 必须立刻失败而不是照常算出一个假的 κ
        assert_no_leaky_columns(df.columns, f"回收的评题表 {p.name}")
        extra = [str(c) for c in df.columns if str(c) not in SHEET_COLUMNS]
        if extra:
            log(f"  [warn] {p.name} 含未登记列 {extra}（已忽略，不影响统计）")
        if "item_id" not in df.columns or "difficulty_1to5" not in df.columns:
            raise SystemExit(f"{p.name} 缺少 item_id / difficulty_1to5 列，无法回收")
        ids = df["item_id"].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].tolist()[:5]
            raise SystemExit(f"{p.name} 的 item_id 有重复：{dup}…")
        num, bad = _parse_difficulty(df["difficulty_1to5"])
        if bad.any():
            rows = df.loc[bad, ["item_id", "difficulty_1to5"]].head(10).to_dict("records")
            raise SystemExit(
                f"{p.name} 的 difficulty_1to5 含无法解析的内容：{rows}…；"
                f"只允许 1~{n_bins()} 的整数，或留空（也可写 NA/不确定）"
            )
        out_of_range = num.notna() & (~num.isin(list(range(1, n_bins() + 1))))
        if out_of_range.any():
            rows = df.loc[out_of_range, ["item_id", "difficulty_1to5"]].head(10).to_dict("records")
            raise SystemExit(f"{p.name} 存在超出 1~{n_bins()} 的取值：{rows}…")

        label = p.stem.replace("rating_sheet_", "")
        s = pd.Series(num.to_numpy(), index=ids.to_numpy()).reindex(mat.index)
        unknown = sorted(set(ids) - set(mat.index))
        if unknown:
            log(f"  [warn] {p.name} 含 {len(unknown)} 个不在 key.csv 中的 item_id："
                f"{unknown[:5]}…（已忽略）")
        mat[label] = s.to_numpy(dtype=np.float64)
        infos.append({
            "rater": label, "path": str(p), "n_items_in_file": int(len(df)),
            "n_rated": int(num.notna().sum()), "n_blank": int(num.isna().sum()),
            "n_missing_vs_key": int(s.isna().sum()),
            "counts_by_level": {str(k): int((num == k).sum())
                                for k in range(1, n_bins() + 1)},
        })
    if mat.shape[1] < 2:
        raise SystemExit(f"至少需要 2 份评题表才能算一致性，实际 {mat.shape[1]} 份")
    return mat, infos


# ---------------------------------------------------------------- 报告工具


def _replace_marked_block(text: str, block: str) -> str:
    """把 text 里的 HUMAN_VS_MODEL 块替换成 block（没有就追加到末尾）。"""
    if BLOCK_BEGIN in text and BLOCK_END in text:
        head, rest = text.split(BLOCK_BEGIN, 1)
        _, tail = rest.split(BLOCK_END, 1)
        return f"{head}{block}{tail}"
    sep = "" if (not text or text.endswith("\n\n")) else "\n\n"
    return f"{text}{sep}{block}"


def _keep_marked_block(path: Path) -> str:
    """取出已存在的 HUMAN_VS_MODEL 块（含标记），没有则返回空串。"""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if BLOCK_BEGIN in text and BLOCK_END in text:
        return BLOCK_BEGIN + text.split(BLOCK_BEGIN, 1)[1].split(BLOCK_END, 1)[0] + BLOCK_END
    return ""


def _update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """合并式更新 json：agree 与 compare 各自只改自己的键，互不覆盖。"""
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  [warn] {path.name} 解析失败，将重建")
            data = {}
    data.update(jsonable(updates))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def conclusion_rule_rows() -> list[tuple[str, str]]:
    """大纲四·（五）的结论分支判定规则（**先定规则，再看结果**，防止事后挑口径）。"""
    return [
        ("Fleiss κ ≥ 0.600 且 平均 QWK ≥ 0.600",
         "可按“模型已达人类标注水平，剩余误差主要来自任务固有的边界模糊性”论述；"
         "仍须报告分档混淆矩阵作为佐证"),
        ("0.400 ≤ Fleiss κ < 0.600",
         "人类共识为中等强度：只能写“模型接近人类水平”，并如实说明等级体系在部分"
         "档位上本就模糊，不得声称达到人类水平"),
        ("Fleiss κ < 0.400",
         "应如实讨论等级体系本身在本题样本上的可分辨性（含 800 下钳位与相邻档互混），"
         "不得声称模型达到人类标注水平"),
    ]


def branch_for(fleiss: float | None, qwk: float | None) -> tuple[str, str]:
    """按上面的规则给出**机械判定**的分支（规则先行，不看结果挑口径）。"""
    if fleiss is None or (isinstance(fleiss, float) and math.isnan(fleiss)):
        return "无法判定", "缺少有效的 Fleiss κ（评题表未填写或有效样本量不足）"
    if fleiss >= 0.6 and (qwk is None or math.isnan(qwk) or qwk >= 0.6):
        return "分支 A（接近）", "Fleiss κ ≥ 0.600 且平均 QWK ≥ 0.600"
    if fleiss >= 0.4:
        return "分支 B（中等）", f"0.400 ≤ Fleiss κ = {fleiss:.3f} < 0.600"
    return "分支 C（偏低）", f"Fleiss κ = {fleiss:.3f} < 0.400"


# ---------------------------------------------------------------- agree


def cmd_agree(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / KAPPA_MD
    json_path = out_dir / KAPPA_JSON
    key = load_key(args.key)
    sheets = resolve_sheets(args.sheets, out_dir)
    log(f"读入 {len(sheets)} 份评题表：{[p.name for p in sheets]}")
    mat, infos = load_rating_matrix(sheets, key["item_id"].astype(str).tolist())
    for info in infos:
        log(f"  {info['rater']}: 评了 {info['n_rated']} / 留空 {info['n_blank']}；"
            f"各档计数 {info['counts_by_level']}")

    # ---- 有效样本量：所有评题人都评过的题（完整子集），缺失题整题剔除
    complete = mat.notna().all(axis=1).to_numpy()
    n_effective = int(complete.sum())
    n_tie_total = consensus_of(mat)[2]
    if mat.shape[1] == 2:
        log("  [warn] 只有 2 名评题人：多数投票无法化解分歧（1:1 即并列），"
            "compare 的可比样本会退化为两人完全一致的题；大纲要求 2~3 人，"
            "如条件允许建议补到 3 人以保留分歧信息")
    if n_effective < 2:
        body = f"""# 人类标注一致性（human κ）实验报告

> 生成时间：{now_iso()}  
> 本文件由 `python scripts/human_kappa.py agree` 与 `compare` 生成，
> 手工修改会在下次运行时被覆盖。

## 有效样本量不足

| 项目 | 数值 |
|---|---|
| 评题表份数 | {mat.shape[1]} |
| 抽样题数 | {len(mat)} |
| **有效样本量（所有评题人均已评）** | **{n_effective}** |
| 众数并列题数 | {n_tie_total} |

> ⚠ 有效样本量不足 2，无法计算 κ。请检查评题表是否已回填
> （`difficulty_1to5` 列需填 1~{n_bins()} 的整数）。

## 结论分支判定规则（先定规则再看结果）

| 判定条件（按顺序匹配） | 允许写入正文的表述 |
|---|---|
""" + "\n".join(f"| {c} | {t} |" for c, t in conclusion_rule_rows()) + """

**本次结果：无法判定**（有效样本量不足）
"""
        md_path.write_text(body + _keep_marked_block(md_path), encoding="utf-8")
        _update_json(json_path, {
            "meta": {"updated_at": now_iso(), "stage": "agree", "n_bins": n_bins(),
                     "bin_edges": list(C.DIFFICULTY_BINS),
                     "bin_interval_text": [bin_interval_text(i) for i in range(n_bins())],
                     "sheets": [str(p) for p in sheets], "key": str(args.key)},
            "agreement": {"n_items_total": int(len(mat)), "n_effective": n_effective,
                          "raters": infos, "conclusion_branch": "无法判定"},
        })
        log(f"[warn] 有效样本量仅 {n_effective} 题，未计算 κ；已写出占位报告 {md_path}")
        return 0

    sub = mat.loc[complete]
    vals = sub.to_numpy(dtype=np.float64)

    # ---- Fleiss κ：自实现 + statsmodels 交叉验证（不一致直接报错，不出报告）
    cats = list(range(1, n_bins() + 1))
    counts = np.zeros((len(sub), len(cats)), dtype=np.float64)
    for j in range(len(cats)):
        counts[:, j] = (vals == cats[j]).sum(axis=1)
    f_manual = fleiss_kappa_manual(counts)
    f_stats = fleiss_kappa_statsmodels(counts)
    if not (math.isnan(f_manual) or math.isnan(f_stats)) and abs(f_manual - f_stats) > 1e-9:
        raise SystemExit(
            f"Fleiss κ 自实现 {f_manual:.12f} 与 statsmodels {f_stats:.12f} 不一致，"
            f"说明实现或输入口径有问题，拒绝出报告"
        )

    pairs = pairwise_kappa(sub)
    valid_pairs = [p for p in pairs if p["cohen_kappa"] is not None]
    mean_cohen = float(np.mean([p["cohen_kappa"] for p in valid_pairs])) if valid_pairs else None
    mean_qwk = float(np.mean([p["qwk"] for p in valid_pairs])) if valid_pairs else None
    raw_agreement = float(np.mean(vals.min(axis=1) == vals.max(axis=1)))
    within1 = float(np.mean((vals.max(axis=1) - vals.min(axis=1)) <= 1))
    spread = vals.max(axis=1) - vals.min(axis=1)
    alpha_ord = krippendorff_alpha(vals, level="ordinal", n_cats=n_bins())
    alpha_nom = krippendorff_alpha(vals, level="nominal", n_cats=n_bins())
    # 分档分歧必须在**有效子集**上统计：spread 的长度是 n_effective，
    # 若拿全样本的共识数组来索引会错位（自检时踩过：boolean index 长度不匹配）。
    cons, tie_sub, n_tie_sub = consensus_of(sub)
    branch, why = branch_for(f_manual, mean_qwk)

    # ---- 分档分歧：以众数档归组（并列题不在其中），看"哪一档人最不一致"
    per_bin: list[dict[str, Any]] = []
    for i, c in enumerate(cats):
        m = ((cons == c) & ~tie_sub)
        n_c = int(m.sum())
        if n_c == 0:
            per_bin.append({"bin": c, "interval": bin_interval_text(i), "n_items": 0,
                            "n_unanimous": 0, "unanimous_share": None, "mean_spread": None})
            continue
        sp = spread[m]
        per_bin.append({
            "bin": c, "interval": bin_interval_text(i), "n_items": n_c,
            "n_unanimous": int((sp == 0).sum()), "unanimous_share": float((sp == 0).mean()),
            "mean_spread": float(sp.mean()),
        })

    result: dict[str, Any] = {
        "n_items_total": int(len(mat)), "n_effective": n_effective,
        "n_dropped_missing": int(len(mat) - n_effective),
        "dropped_items": mat.index[~complete].tolist(),
        "n_tie_consensus": n_tie_total,
        "n_tie_in_effective": n_tie_sub,
        "fleiss_kappa": f_manual, "fleiss_kappa_statsmodels": f_stats,
        "fleiss_kappa_abs_diff": abs(f_manual - f_stats),
        "mean_pairwise_cohen_kappa": mean_cohen, "mean_pairwise_qwk": mean_qwk,
        "raw_agreement": raw_agreement, "within1_agreement": within1,
        "mean_spread": float(spread.mean()),
        "krippendorff_alpha_ordinal": alpha_ord,
        "krippendorff_alpha_nominal": alpha_nom,
        "pairs": pairs, "per_bin": per_bin, "raters": infos,
        "conclusion_branch": branch, "conclusion_branch_reason": why,
    }

    body = f"""# 人类标注一致性（human κ）实验报告

> 生成时间：{now_iso()}  
> 口径来源：`src/task2difficulty/config.py`（档位边界/抽样量/评题人数均从 config 读取，
> 脚本内无硬编码）  
> 本文件由 `python scripts/human_kappa.py agree` 与 `compare` 生成，
> 手工修改会在下次运行时被覆盖。

## 一、实验口径（大纲四·（五））

| 项目 | 本次设置 |
|---|---|
| 评题人看到的信息 | 题面全文 + 算法标签；**无任何提交统计** |
| 为什么不能给提交统计 | 与 D 格（时间划分 + 统计特征置空）的特征条件对齐，否则人类一致率与模型一致率不可比 |
| 评委身份 | 有竞赛经验的教师/研究生（估计任务的人类可达上界）；**与五·（一）教学阈值重校准的任课教师是两拨人，必须分开报告** |
| 抽样 | {len(mat)} 题，按 rating 段分层（低/中/高）+ 新题旧题混合 |
| 分档定义 | 等宽 {n_bins()} 档，边界取自 `config.DIFFICULTY_BINS`：{'；'.join(f'{i + 1}档 {bin_interval_text(i)}' for i in range(n_bins()))} |
| 缺失值处理 | 只对**所有评题人都评过**的题计算；有效样本量见下，未做任何众数/均值填补 |

## 二、评题人回填情况

| 评题人 | 评了 | 留空 | {' | '.join(f'{k}档' for k in range(1, n_bins() + 1))} |
|---|---|---|---|
""" + "\n".join(
        f"| {i['rater']} | {i['n_rated']} | {i['n_blank']} | "
        + " | ".join(str(i["counts_by_level"][str(k)]) for k in range(1, n_bins() + 1))
        + " |"
        for i in infos
    ) + f"""

## 三、有效样本量

| 项目 | 数值 |
|---|---|
| 抽样总题数 | {len(mat)} |
| **有效样本量（所有评题人均已评）** | **{n_effective}** |
| 因有人留空而整题剔除 | {len(mat) - n_effective} |
| 众数并列（无唯一共识档）的题数 | {n_tie_total}（其中落在有效子集内 {n_tie_sub} 题） |

> 有效样本量低于 {len(mat)} 时，正文必须写明"κ 基于 N={n_effective} 的有效子集"，
> 不得按 {len(mat)} 题表述。缺失题不得用众数填补后混入计算。

## 四、整体一致性指标

| 指标 | 值 | 说明 |
|---|---|---|
| **Fleiss' κ**（自实现） | **{fmt_k(f_manual)}** | {mat.shape[1]} 名评题人的整体一致性 |
| Fleiss' κ（statsmodels 交叉验证） | {fmt_k(f_stats)} | 与自实现差值 {abs(f_manual - f_stats):.2e}（要求 &lt; 1e-9） |
| Cohen's κ（两两均值） | {fmt_k(mean_cohen)} | 未加权；忽略档位顺序 |
| **二次加权 κ（QWK，两两均值）** | **{fmt_k(mean_qwk)}** | ★ 难度是有序等级，本文正文口径 |
| 原始一致率（全部评题人同档） | {fmt_pct(raw_agreement)} | 最严格口径 |
| ±1 档一致率 | {fmt_pct(within1)} | 相邻档互混是否为主 |
| 平均极差（max−min） | {fmt_k(float(spread.mean()))} | 单位：档 |
| Krippendorff's α（有序） | {fmt_k(alpha_ord)} | 允许缺失的稳健佐证；与 QWK 不是同一指标 |
| Krippendorff's α（名义） | {fmt_k(alpha_nom)} | 忽略顺序，作下界参照 |

> 解读参考（Landis &amp; Koch, 1977）：&lt;0.20 极弱 / 0.21–0.40 一般 / 0.41–0.60 中等 /
> 0.61–0.80 较强 / &gt;0.80 很强。QWK 通常高于未加权 κ（它惩罚"距离"而非"是否同档"），
> 两者并列报告才能说明等级体系的可分辨性。

## 五、两两配对明细

| 配对 | 有效 N | Cohen's κ | QWK | 原始一致率 | ±1 档一致率 |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {p['pair']} | {p['n']} | {fmt_k(p['cohen_kappa'])} | {fmt_k(p['qwk'])} | "
        f"{fmt_pct(p['raw_agreement'])} | {fmt_pct(p['within1_agreement'])} |"
        for p in pairs
    ) + f"""

## 六、分档分歧程度

| 档位 | rating 区间 | 众数档为该档的题数 | 完全一致 | 完全一致占比 | 平均极差 |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {b['bin']} | {b['interval']} | {b['n_items']} | {b['n_unanimous']} | "
        f"{fmt_pct(b['unanimous_share'])} | {fmt_k(b['mean_spread'])} |"
        for b in per_bin
    ) + f"""

## 七、结论分支（规则先行，不预先下结论）

判断规则如下，**先固定规则再看数值**，不允许事后挑选口径：

| 判定条件（按顺序匹配） | 允许写入正文的表述 |
|---|---|
""" + "\n".join(f"| {c} | {t} |" for c, t in conclusion_rule_rows()) + f"""

**本次结果的机械判定：{branch}** —— {why}
（Fleiss κ = {fmt_k(f_manual)}，平均 QWK = {fmt_k(mean_qwk)}，
原始一致率 = {fmt_pct(raw_agreement)}，有效 N = {n_effective}）

> 若 Fleiss κ 与平均 QWK 落在不同分支（例如 κ=0.58 而 QWK=0.65），正文必须同时
> 报告两者并说明分歧原因（QWK 计入了序信息，故通常更高），不得只挑有利的那一个。
"""

    md_path.write_text(body + _keep_marked_block(md_path), encoding="utf-8")
    _update_json(json_path, {
        "meta": {"updated_at": now_iso(), "stage": "agree", "n_bins": n_bins(),
                 "bin_edges": list(C.DIFFICULTY_BINS),
                 "bin_interval_text": [bin_interval_text(i) for i in range(n_bins())],
                 "sheets": [str(p) for p in sheets], "key": str(args.key)},
        "agreement": result,
    })

    log(f"有效样本量 {n_effective}/{len(mat)}；Fleiss κ = {fmt_k(f_manual)}"
        f"（statsmodels 差值 {abs(f_manual - f_stats):.1e}）；平均 QWK = {fmt_k(mean_qwk)}")
    log(f"原始一致率 {fmt_pct(raw_agreement)}；±1 档一致率 {fmt_pct(within1)}；"
        f"Krippendorff α(ordinal) = {fmt_k(alpha_ord)}")
    log(f"结论分支机械判定：{branch}（{why}）")
    log(f"-> {md_path}\n-> {json_path}")
    return 0


# ---------------------------------------------------------------- compare


def _discover_preds(cond: str, models: Sequence[str] | None,
                    seeds: Sequence[int] | None) -> list[tuple[str, int]]:
    """扫描 `outputs/results/preds/{condition}_{model}_seed{N}.npz`。

    文件名规则与 `experiments.save_predictions` 一致；这里只负责枚举，
    真正读取仍走 `experiments.load_predictions`（口径唯一来源）。
    """
    pred_dir = C.RESULTS_DIR / "preds"
    if not pred_dir.exists():
        return []
    pat = re.compile(rf"^{re.escape(cond)}_(?P<model>.+)_seed(?P<seed>\d+)$")
    found: set[tuple[str, int]] = set()
    for p in sorted(pred_dir.glob(f"{cond}_*_seed*.npz")):
        m = pat.match(p.stem)
        if not m:
            continue
        model, seed = m.group("model"), int(m.group("seed"))
        if models and model not in set(models):
            continue
        if seeds and seed not in set(seeds):
            continue
        found.add((model, seed))
    return sorted(found)


def _cm_table(cm: np.ndarray, row_label: str, col_label: str) -> str:
    k = cm.shape[0]
    head = (f"| {row_label} \\ {col_label} | "
            + " | ".join(str(i + 1) for i in range(k)) + " | 合计 |")
    sep = "|" + "---|" * (k + 2)
    rows = [f"| **{i + 1}** | " + " | ".join(str(int(v)) for v in cm[i])
            + f" | {int(cm[i].sum())} |" for i in range(k)]
    rows.append("| **合计** | " + " | ".join(str(int(v)) for v in cm.sum(axis=0))
                + f" | {int(cm.sum())} |")
    return "\n".join([head, sep] + rows)


def cmd_compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / KAPPA_MD
    json_path = out_dir / KAPPA_JSON
    cond_key = str(args.condition).upper()
    if cond_key not in C.CONDITIONS:
        raise SystemExit(f"未知因子格 {cond_key}；可选 {sorted(C.CONDITIONS)}")
    if cond_key != "D" and not args.i_know_its_not_D:
        raise SystemExit(
            "人类只看题面+标签（无提交统计），只能与 D 格（时间划分 + 统计特征置空）比较；"
            f"当前指定的是 {cond_key} 格。A/C 格含统计特征（C 格还是泄漏上界），"
            "B 格是随机划分、测试集里混有旧题，都与人类可见信息不可比。"
            "如确要比较（仅作敏感性分析，不能进正文主结论）请显式加 --i-know-its-not-D。"
        )
    if cond_key != "D":
        log(f"[warn] 正在用 {cond_key} 格做比较——这不是论文口径，结果只能作敏感性分析")
    cond = C.CONDITIONS[cond_key]

    # 惰性导入：experiments 会连带导入 models_deep（torch），sample/agree 不需要它
    try:
        from task2difficulty import experiments as E
    except Exception as ex:      # pragma: no cover - 环境缺 torch 时给出清晰提示
        raise SystemExit(f"无法导入 task2difficulty.experiments（读预测缓存依赖它）：{ex}")

    pairs = _discover_preds(cond_key, args.model, args.seed)
    if not pairs:
        msg = (f"未找到 {cond_key} 格的预测缓存 "
               f"`outputs/results/preds/{cond_key}_{{model}}_seed{{N}}.npz`。"
               f"请先运行：python scripts/run_experiments.py --stage factors。"
               f"⚠ 按大纲口径**不得**改用 A 格（含统计特征 + 随机划分）冒充 D 格。")
        log(f"[skip] {msg}")
        block = f"""{BLOCK_BEGIN}
## 人类 vs 模型一致率（{cond.label}）

> ⚠ **跳过**：{msg}

在补齐 D 格预测缓存之前，本报告只包含人类内部一致性（上界），不含模型对比。
{BLOCK_END}"""
        md = _replace_marked_block(
            md_path.read_text(encoding="utf-8") if md_path.exists()
            else "# 人类标注一致性（human κ）实验报告\n", block)
        md_path.write_text(md, encoding="utf-8")
        _update_json(json_path, {"human_vs_model": {
            "skipped": True, "reason": msg, "condition": cond_key, "updated_at": now_iso(),
        }})
        return 0

    # ---- 人类侧：共识档 + 内部一致率（上界参照）
    key = load_key(args.key)
    sheets = resolve_sheets(args.sheets, out_dir)
    mat, infos = load_rating_matrix(sheets, key["item_id"].astype(str).tolist())
    cons_all, tie_all, n_tie = consensus_of(mat)
    # 先判"表填了没有"：若所有题都没有唯一共识档，后面的"无可比样本"报错会指向
    # 抽样与测试集不重叠这一错误原因，把真正的问题（评题表还是空的）掩盖掉。
    if int((~tie_all & ~np.isnan(cons_all)).sum()) == 0:
        raise SystemExit(
            "评题表尚未填写（或所有题都不存在唯一共识档），无法计算人类共识。"
            "请先把回收的 rating_sheet_rater*.csv 放回 --out-dir，再运行 agree / compare。"
        )
    human = pd.DataFrame({
        "item_id": mat.index.astype(str), "human_consensus": cons_all, "is_tie": tie_all,
    }).merge(key[["item_id", "problem_key", "rating", "year"]], on="item_id", how="left")
    human["human_true_bin"] = rating_to_bin(human["rating"].to_numpy())
    # 抽样中的新题占比（= 能与 D 格比较的题占比，正文口径说明会引用）
    new_share = float((key["year"].to_numpy() >= C.TEMPORAL_TEST_START_YEAR).mean())

    # ---- 模型侧：D 格测试集的题序必须与 npz 里的阵列一一对应。
    # 这里用「dataset.csv 行序 == split_index.csv 行序」这一不变式重建测试集位置，
    # 再用 y_test 与数据集 rating 逐题核对——对不上就说明映射假设错了，必须立刻报错，
    # 而不是算出一个看似合理的错误一致率。
    raw = pd.read_csv(C.DATASET_PATH, encoding="utf-8-sig")
    si = D.load_split_index(C.SPLIT_INDEX_PATH)
    if len(raw) != len(si.problem_key) or not (
        raw["problem_key"].astype(str).to_numpy() == si.problem_key.astype(str).to_numpy()
    ).all():
        raise SystemExit(
            "dataset.csv 与 split_index.csv 的行序/题目集合不一致，无法把 npz 预测对回题目身份。"
            "请重新生成划分索引（python scripts/run_experiments.py --stage prepare）。"
        )
    # ⚠ 测试集必须按**当前条件自己的划分**取：D 格是时间划分的测试段（= 新题），
    # 而 A/B 格是随机划分的测试段（39 题，含旧题）。写死 SPLIT_TEMPORAL 会让
    # 非 D 格的位置索引越界（自检 A 格敏感性路径时踩过：index 88 out of bounds for size 39）。
    test_pos = np.flatnonzero(si.mask(cond.split, "test"))
    split_name = "时间划分" if cond.split == C.SPLIT_TEMPORAL else "随机划分"
    # ⚠ pos_of_key 必须是"在测试子集内的位置"（0..n_test-1），而不是数据集全局行号：
    # npz 里的 test_pred/y_test 的长度就是 n_test，用全局行号索引会越界或错位。
    test_keys = si.problem_key.astype(str).to_numpy()[test_pos]
    pos_of_key = {str(k): i for i, k in enumerate(test_keys)}
    raw_rating = raw["rating"].to_numpy(dtype=np.float64)
    # 人类抽样里落在该格测试集中的题数：与模型无关的常数，先算一次
    n_human_in_test = int(human["problem_key"].astype(str).isin(pos_of_key).sum())
    log(f"  {cond.label} 测试集 {len(test_pos)} 题（{split_name}），"
        f"与人类抽样交集 {n_human_in_test} 题")

    rows: list[dict[str, Any]] = []
    cms: dict[str, dict[str, Any]] = {}
    align_reports: list[dict[str, Any]] = []
    for model, seed in pairs:
        z = E.load_predictions(cond_key, model, seed)
        if z is None:
            log(f"  [warn] {cond_key}/{model}/seed{seed} 读取失败，跳过")
            continue
        test_pred = np.asarray(z["test_pred"], dtype=np.float64)
        y_test = np.asarray(z["y_test"], dtype=np.float64)
        edges = np.asarray(z["width_edges"], dtype=np.float64)

        # 分箱口径校验：模型用的等宽边界必须与 config 一致，否则人类档与模型档
        # 不是同一套档位定义，一致率没有意义（宁可报错也不能出一张假表）。
        if not np.allclose(edges[1:-1], np.asarray(C.DIFFICULTY_BINS, dtype=np.float64)):
            raise SystemExit(
                f"{cond_key}/{model}/seed{seed} 的 width_edges={edges.tolist()} 与 "
                f"config.DIFFICULTY_BINS={list(C.DIFFICULTY_BINS)} 不一致，"
                f"人类档位与模型档位不可比；请统一分箱口径后重跑。"
            )
        thr = M.BinThresholds(edges=tuple(float(e) for e in edges),
                              scheme="equal_width", fitted_on="valid")
        model_bin = thr.apply(test_pred) + 1

        # 可比子集：人类抽中 ∩ D 格测试集 ∩ 有唯一共识档
        sel = human[human["problem_key"].astype(str).isin(pos_of_key)].copy()
        sel = sel[~sel["is_tie"].astype(bool) & sel["human_consensus"].notna()]
        if sel.empty:
            log(f"  [warn] {model}: 人类抽样与 D 格测试集无交集（新题抽得太少？），跳过")
            continue
        sel["pos"] = sel["problem_key"].astype(str).map(pos_of_key)
        sel["model_bin"] = model_bin[sel["pos"].to_numpy()]
        sel["rating_in_npz"] = y_test[sel["pos"].to_numpy()]
        # 双通道核对：npz 的 y_test 必须同时等于 key.csv 的 rating 与 dataset.csv 的 rating
        n_mismatch_key = int(np.sum(np.abs(sel["rating_in_npz"] - sel["rating"]) > 1e-6))
        n_mismatch_raw = int(np.sum(
            np.abs(sel["rating_in_npz"]
                   - raw_rating[test_pos[sel["pos"].to_numpy()]]) > 1e-6))
        align_reports.append({
            "model": model, "seed": seed, "n_overlap": int(len(sel)),
            "n_rating_mismatch": n_mismatch_key + n_mismatch_raw,
            "n_mismatch_vs_key": n_mismatch_key, "n_mismatch_vs_dataset": n_mismatch_raw,
        })
        if n_mismatch_key or n_mismatch_raw:
            raise SystemExit(
                f"{cond_key}/{model}/seed{seed}: npz 的 y_test 与 key.csv/dataset.csv 的 rating "
                f"不符（{n_mismatch_key}/{n_mismatch_raw} 题），说明测试集位置映射错误，拒绝出报告。"
            )

        y_h = sel["human_consensus"].to_numpy(dtype=np.float64)
        y_m = sel["model_bin"].to_numpy(dtype=np.float64)
        n = int(len(sel))
        agree = float(np.mean(y_h == y_m))
        within1 = float(np.mean(np.abs(y_h - y_m) <= 1))
        # 退化情形（只有一个取值 / 标签集合不相交）由 safe_kappa 统一处理
        kappa = safe_kappa(y_h, y_m)
        qwk = safe_kappa(y_h, y_m, weights="quadratic")

        # 人类内部一致率（同一可比子集上）——"人类可达水平"的上界参照
        sub_mat = mat.loc[sel["item_id"].astype(str).tolist()]
        sub_complete = sub_mat.notna().all(axis=1).to_numpy()
        hv = sub_mat.to_numpy(dtype=np.float64)
        human_raw = (float(np.mean(hv[sub_complete].min(axis=1) == hv[sub_complete].max(axis=1)))
                     if sub_complete.sum() else None)
        human_w1 = (float(np.mean((hv[sub_complete].max(axis=1) - hv[sub_complete].min(axis=1)) <= 1))
                    if sub_complete.sum() else None)
        human_true_agree = float(np.mean(y_h == sel["human_true_bin"].to_numpy(dtype=np.float64)))

        k = n_bins()
        cm = np.zeros((k, k), dtype=int)
        np.add.at(cm, (y_h.astype(int) - 1, y_m.astype(int) - 1), 1)
        cms[f"{model}_seed{seed}"] = {"model": model, "seed": seed, "n": n, "cm": cm.tolist()}
        rows.append({
            "model": model, "seed": seed, "n_compared": n, "agreement": agree,
            "within1_agreement": within1, "cohen_kappa": kappa, "qwk": qwk,
            "human_internal_raw_agreement": human_raw, "human_internal_within1": human_w1,
            "human_vs_true_bin_agreement": human_true_agree,
            "gap_model_minus_human": (None if human_raw is None else agree - human_raw),
        })
        log(f"  {cond_key}/{model}/seed{seed}: 可比 N={n}，一致率 {fmt_pct(agree)}，"
            f"±1 {fmt_pct(within1)}，Cohen κ {fmt_k(kappa)}，QWK {fmt_k(qwk)}，"
            f"人类内部原始一致率 {fmt_pct(human_raw)}")

    if not rows:
        raise SystemExit(
            "没有任何 (模型, 种子) 能与人类抽样对齐。常见原因："
            "① 抽样里的新题（year ≥ TEMPORAL_TEST_START_YEAR）为 0，与 D 格测试集无交集；"
            "② D 格预测缓存来自另一版数据集/划分索引。"
            "请核对 sample 日志里的新旧题占比，并确认 split_index.csv 与 dataset.csv 同源。"
        )

    df = pd.DataFrame(rows)
    agg_rows: list[dict[str, Any]] = []
    for model, g in df.groupby("model", sort=True):
        rec: dict[str, Any] = {"model": model, "n_seeds": int(len(g))}
        for col in ("agreement", "within1_agreement", "cohen_kappa", "qwk",
                    "human_internal_raw_agreement", "gap_model_minus_human"):
            v = g[col].dropna().to_numpy(dtype=np.float64)
            rec[col] = float(v.mean()) if v.size else None
            rec[col + "_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        rec["n_compared"] = int(g["n_compared"].iloc[0])
        agg_rows.append(rec)
    agg = pd.DataFrame(agg_rows)

    first = df.iloc[0]
    gap = first["gap_model_minus_human"]
    if gap is None:
        compare_branch = "无法判定（缺少人类内部一致率）"
    elif gap >= 0:
        compare_branch = ("分支 A：模型一致率已达/超过人类内部一致率，"
                          "可写“模型已达人类标注水平”")
    elif gap > -0.10:
        compare_branch = "分支 B：模型一致率略低于人类内部一致率（差距 < 10 pp），写“接近人类水平”"
    else:
        compare_branch = ("分支 C：模型一致率明显低于人类内部一致率（差距 ≥ 10 pp），"
                          "不得声称达到人类标注水平")

    compared_models = sorted(set(df["model"]))
    missing_models = [m for m in list(C.MAIN_TABLE_MODELS) if m not in compared_models]
    block = f"""{BLOCK_BEGIN}
## 人类 vs 模型一致率（{cond.label}）

### 8.1 为什么论文口径只能对 D 格

""" + ("" if cond_key == "D" else
       f"""> ⚠⚠ **本节用的是 {cond_key} 格（{cond.label}），不是论文口径**：
> 该格 {"含统计特征（人类看不到提交统计，条件不对齐）" if cond.stat != C.STAT_MASKED else "使用随机划分（测试集含旧题，与“预测新题”场景不符）"}
> {"且是泄漏乐观上界" if cond.leaky else ""}。这些数字**只能作敏感性分析**，
> 不得写入正文主结论。

""") + f"""- 评题人只看到**题面 + 算法标签**，没有任何提交统计；
- D 格 = 时间划分 + **统计特征置空**，且测试集（year ≥ {C.TEMPORAL_TEST_START_YEAR}）
  全部是新题——这正是"新题上线时无提交统计"的真实场景，与人类可见信息一一对应；
- B 格虽然也是冷启动，但用随机划分、测试集里混有旧题，与"预测新题"的场景不符；
  A/C 格含统计特征（C 格还是泄漏上界），**绝不能**用来与人类比较（大纲四·（二））；
- D 格中的**纯文本模型**（textcnn/bert）更贴近人类可见信息（人只看题面+标签，
  看不到统计与约束规模特征的数值化结果），因此它们是最干净的一列；
  本次可用的预测缓存见下表。

### 8.2 可比样本

| 项目 | 数值 |
|---|---|
| 人类抽样题数 | {int(len(mat))} |
| 其中落在 {cond_key} 格测试集（{split_name}，共 {len(test_pos)} 题） | {n_human_in_test} |
| 其中因"无唯一共识档 / 有人留空"而排除 | {n_human_in_test - int(first['n_compared'])} |
| **实际比较的 N（人类有唯一共识档）** | **{int(first['n_compared'])}** |
| npz y_test 与 key.csv/dataset.csv 的 rating 核对不一致数 | 0（校验通过，位置映射正确） |

> 只有落在该格测试集里的题才能与模型比较（模型没见过的题才有意义；
> 训练/验证集上的预测不算泛化）。这也是抽样时把新题比例提高到
> 约 {new_share:.0%} 的原因：新题（year ≥ {C.TEMPORAL_TEST_START_YEAR}）与 D 格测试集
> 是同一批题。旧题虽不参与 D 格比较，但要用于分档分歧与"新旧题一致性对比"分析。

### 8.3 汇总（每个模型 × 种子）

| 模型 | 种子 | 可比 N | 与人类共识一致率 | ±1 档一致率 | Cohen's κ | QWK | 人类内部原始一致率（上界） | 差值(模型−人类) |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r['model']} | {r['seed']} | {r['n_compared']} | {fmt_pct(r['agreement'])} | "
        f"{fmt_pct(r['within1_agreement'])} | {fmt_k(r['cohen_kappa'])} | {fmt_k(r['qwk'])} | "
        f"{fmt_pct(r['human_internal_raw_agreement'])} | "
        f"{('+' if (r['gap_model_minus_human'] or 0) >= 0 else '')}"
        f"{fmt_pct(r['gap_model_minus_human'])} |"
        for _, r in df.iterrows()
    ) + """

按模型聚合（多种子均值，括号内为标准差）：

| 模型 | 种子数 | 一致率 | ±1 档 | Cohen's κ | QWK | 人类内部一致率 | 差值 |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r['model']} | {r['n_seeds']} | {fmt_pct(r['agreement'])} "
        f"({fmt_pct(r['agreement_std'])}) | {fmt_pct(r['within1_agreement'])} "
        f"({fmt_pct(r['within1_agreement_std'])}) | {fmt_k(r['cohen_kappa'])} | "
        f"{fmt_k(r['qwk'])} | {fmt_pct(r['human_internal_raw_agreement'])} | "
        f"{fmt_pct(r['gap_model_minus_human'])} |"
        for _, r in agg.iterrows()
    ) + f"""

### 8.4 人类内部一致率（上界参照）

在**同一个可比子集**上，人类评题人之间的原始一致率为
{fmt_pct(first['human_internal_raw_agreement'])}、±1 档一致率为
{fmt_pct(first['human_internal_within1'])}。这就是"人类可达水平"的直接度量：
模型一致率高于它并不说明模型"比人强"，只说明该子集上题目的档位恰好落进了模型
更容易命中的区间；明显低于它则说明模型尚未达到人类标注水平。

**机械判定（以表中第一行 {first['model']} seed={first['seed']} 为准）：{compare_branch}**
（差值 = {fmt_pct(gap)}）

### 8.5 混淆矩阵（人类共识档 × 模型预测档；每个模型取第一个可用种子）

""" + "\n\n".join(
        f"**{r['model']}（seed {r['seed']}）**\n\n"
        + _cm_table(np.asarray(cms[f"{r['model']}_seed{r['seed']}"]["cm"]), "人类共识", "模型预测")
        for _, r in df.drop_duplicates("model").iterrows()
    ) + f"""

> 相邻档互混为主 → 难度是连续量、分类边界有固有模糊性；这与人类 κ 的分档分歧
> （见上文章节"分档分歧程度"）互相印证：模型错在哪，往往人也一样错在哪。
> 注意最低档含 Codeforces 800 下钳位（rating 无法低于 800），该档内部本身噪声最大。

### 8.6 口径提醒

- 人类一致率与模型一致率的**分母必须相同**（本节均为
  {int(first['n_compared'])} 题），不得用人评的 {int(len(mat))} 题去比模型的另一批题；
- 本节所有模型结果均取自 D 格（时间划分 + 统计特征置空）的测试集预测缓存，
  未使用 A/B/C 格的任何数字；
- 若 D 格的 textcnn/bert 缓存缺失，只能说明"传统模型可达/未达人类水平"，
  不能推广到"本文所有模型"——缺失的模型必须如实列出。

### 8.7 参与与未参与比较的模型

| 情形 | 模型 |
|---|---|
| 已有 D 格缓存并完成比较 | {', '.join(compared_models)} |
| D 格无预测缓存（本次未比较） | {', '.join(missing_models) if missing_models else '—'} |

对齐校验明细：

| 模型 | 种子 | 可比 N | rating 核对不一致数 |
|---|---|---|---|
""" + "\n".join(
        f"| {a['model']} | {a['seed']} | {a['n_overlap']} | {a['n_rating_mismatch']} |"
        for a in align_reports
    ) + f"\n{BLOCK_END}"

    md = _replace_marked_block(
        md_path.read_text(encoding="utf-8") if md_path.exists()
        else ("# 人类标注一致性（human κ）实验报告\n\n"
              "> ⚠ 本文件由 compare 生成时 agree 尚未运行：上方缺少人类 κ 正文章节，"
              "说明评题表还没回收完整或 agree 未跑。\n"),
        block)
    md_path.write_text(md, encoding="utf-8")
    _update_json(json_path, {"human_vs_model": {
        "updated_at": now_iso(), "condition": cond_key, "condition_label": cond.label,
        "leaky": cond.leaky, "n_bins": n_bins(), "skipped": False,
        "rows": df.to_dict("records"), "by_model": agg.to_dict("records"),
        "confusion_matrices": cms, "alignment": align_reports,
        "human_internal_on_overlap": {
            "raw_agreement": first["human_internal_raw_agreement"],
            "within1_agreement": first["human_internal_within1"],
            "n_tie_excluded": n_tie,
        },
        "compare_branch": compare_branch, "raters": infos,
    }})
    log(f"人类 vs 模型比较完成：{len(df)} 组（模型×种子），可比 N={int(first['n_compared'])}")
    log(f"机械判定：{compare_branch}")
    log(f"-> {md_path}\n-> {json_path}")
    return 0


# ---------------------------------------------------------------- selftest


def cmd_selftest(args: argparse.Namespace) -> int:
    """开发自检：临时目录里造模拟填答 → 跑通 agree/compare，并核对 κ 的简单算例。

    ⚠ 全部产物写在系统临时目录，验证完即删；**不会**污染 outputs/human_kappa/。
    这样"没有真实填写数据"也能验证整条计算链路。
    """
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'✓' if cond else '✗'}] {name}" + (f" —— {detail}" if detail else ""))

    def counts_of(vals: np.ndarray) -> np.ndarray:
        c = np.zeros((len(vals), n_bins()), dtype=np.float64)
        for j in range(n_bins()):
            c[:, j] = (vals == j + 1).sum(axis=1)
        return c

    log("自检 1/4：κ 实现的简单算例")
    # 算例 1：3 人完全一致 → 所有指标 = 1.0
    perfect = np.array([[3, 3, 3]] * 40, dtype=np.float64)
    k_perfect = fleiss_kappa_manual(counts_of(perfect))
    check("3 人完全一致：Fleiss κ = 1.0", abs(k_perfect - 1.0) < 1e-12, f"实际 {k_perfect:.12f}")
    check("3 人完全一致：Krippendorff α(ordinal) = 1.0",
          abs(krippendorff_alpha(perfect, "ordinal", n_bins()) - 1.0) < 1e-12)
    # ⚠ sklearn 1.9 对"只有一个取值"的输入返回 NaN，故这里走 safe_kappa 的退化分支
    check("3 人完全一致：两两 Cohen κ = 1.0（sklearn 会返回 NaN，由 safe_kappa 兜住）",
          safe_kappa(perfect[:, 0], perfect[:, 1]) == 1.0,
          f"safe_kappa={safe_kappa(perfect[:, 0], perfect[:, 1])}")
    check("标签不相交时 κ 如实返回 NaN（不伪造数字）",
          math.isnan(safe_kappa(np.array([1.0, 2.0]), np.array([4.0, 5.0]))))

    # 算例 2：手算例——2 人 50 题，混淆矩阵 [[12,13],[13,12]]
    # 原始一致率 24/50 = 0.48；边际 25/25 → P_e = 0.5 → κ = (0.48−0.5)/(1−0.5) = −0.04
    # 2 人时 Fleiss κ 与未加权 Cohen κ 恒等；2 档时 QWK 也恒等（权重矩阵只有跨档项）
    y1 = np.array([1] * 12 + [1] * 13 + [2] * 13 + [2] * 12, dtype=np.float64)
    y2 = np.array([1] * 12 + [2] * 13 + [1] * 13 + [2] * 12, dtype=np.float64)
    hand = (0.48 - 0.5) / (1 - 0.5)
    c2 = counts_of(np.stack([y1, y2], axis=1))
    check("手算例：Cohen κ = −0.04", abs(safe_kappa(y1, y2) - hand) < 1e-12,
          f"实际 {safe_kappa(y1, y2):.12f}，手算 {hand:.12f}")
    check("手算例：Fleiss κ = −0.04（2 人时与 Cohen κ 恒等）",
          abs(fleiss_kappa_manual(c2) - hand) < 1e-12,
          f"实际 {fleiss_kappa_manual(c2):.12f}")
    check("手算例：QWK = −0.04（2 档时与未加权 κ 相同）",
          abs(safe_kappa(y1, y2, weights="quadratic") - hand) < 1e-12)
    check("手算例：自实现与 statsmodels 一致",
          abs(fleiss_kappa_manual(c2) - fleiss_kappa_statsmodels(c2)) < 1e-9,
          f"差值 {abs(fleiss_kappa_manual(c2) - fleiss_kappa_statsmodels(c2)):.2e}")

    # 算例 3：完全随机分歧 → κ 应接近 0
    rng = np.random.default_rng(20260101)
    rand = rng.integers(1, n_bins() + 1, size=(4000, 3)).astype(np.float64)
    k3 = fleiss_kappa_manual(counts_of(rand))
    a3 = krippendorff_alpha(rand, "ordinal", n_bins())
    check("完全随机分歧：|Fleiss κ| < 0.03（4000 题 × 3 人）", abs(k3) < 0.03, f"实际 {k3:+.4f}")
    check("完全随机分歧：|Krippendorff α(ordinal)| < 0.05", abs(a3) < 0.05, f"实际 {a3:+.4f}")

    # 算例 4：有序性——同一位评题人、同样 20 题分歧，只把分歧距离从"相邻档"改成"跨 3 档"：
    # 未加权 κ 完全不变（它只看"是否同档"），QWK 显著下降（它看"差多远"）。
    # 这正是正文用 QWK 而不是只用 κ 的理由，也是自检必须覆盖的性质。
    # ⚠ 设计要点：数据里必须五档都出现，否则 sklearn 会按"实际出现的标签"重排秩，
    # 数值距离被压缩，"跨档"与"相邻"会算出同一个数（自检第二版踩过这个坑）。
    y1o = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20, dtype=np.float64)
    y_near = y1o.copy()
    y_near[y1o == 2] = 3          # 距离 1
    y_far = y1o.copy()
    y_far[y1o == 2] = 5           # 距离 3
    k_near, k_far = safe_kappa(y1o, y_near), safe_kappa(y1o, y_far)
    q_near = safe_kappa(y1o, y_near, weights="quadratic")
    q_far = safe_kappa(y1o, y_far, weights="quadratic")
    check("未加权 κ 对'相邻档'与'跨 3 档'给出同一个值（忽略序信息）",
          abs(k_near - k_far) < 1e-12, f"相邻 {k_near:.4f}，跨 3 档 {k_far:.4f}")
    check("QWK 对'跨 3 档'的惩罚显著大于'相邻档'（有序信息被计入）",
          q_far < k_far - 0.05 and q_near > k_near + 0.05,
          f"QWK 相邻 {q_near:.4f} > κ {k_near:.4f} = κ {k_far:.4f} > QWK 跨档 {q_far:.4f}")

    log("自检 2/4：临时目录跑 sample（不写正式 outputs/）")
    with tempfile.TemporaryDirectory(prefix="hk_selftest_") as tmp:
        tmpdir = Path(tmp)
        rc = cmd_sample(argparse.Namespace(
            out_dir=str(tmpdir), seed=int(C.SPLIT_SEED), n_items=int(C.HUMAN_KAPPA_N_ITEMS),
            n_raters=int(C.HUMAN_KAPPA_N_RATERS), new_frac=0.5,
            dataset=str(C.DATASET_PATH), force=True,
        ))
        check("sample 返回 0", rc == 0)
        sheets = sorted(tmpdir.glob(SHEET_PATTERN))
        check(f"生成 {C.HUMAN_KAPPA_N_RATERS} 份评题表",
              len(sheets) == int(C.HUMAN_KAPPA_N_RATERS), f"实际 {len(sheets)} 份")
        for p in sheets:
            cols = [str(c) for c in pd.read_csv(p, encoding="utf-8-sig", nrows=0).columns]
            leaked = [c for c in cols if _is_forbidden_sheet_column(c)]
            check(f"{p.name} 列名无泄漏且恰为 SHEET_COLUMNS",
                  not leaked and cols == list(SHEET_COLUMNS), f"列={cols}")
            body = pd.read_csv(p, encoding="utf-8-sig")
            check(f"{p.name} 填答列为空、题面非空",
                  bool(body["difficulty_1to5"].isna().all())
                  and bool(body["statement"].notna().all()))
        check("key.csv 首行为警告注释且可被 load_key 读回",
              (tmpdir / KEY_FILE).read_text(encoding="utf-8-sig").startswith("#")
              and len(load_key(tmpdir / KEY_FILE)) == int(C.HUMAN_KAPPA_N_ITEMS))
        check("instructions.md 已生成", (tmpdir / INSTRUCTIONS_FILE).exists())

        log("自检 3/4：造模拟填答（含缺失与分歧）并跑 agree")
        sim = tmpdir / "sim"
        sim.mkdir()
        base = pd.read_csv(sheets[0], encoding="utf-8-sig", dtype=str)
        rng = np.random.default_rng(7)
        truth = np.clip(np.round(np.linspace(1, n_bins(), len(base))), 1, n_bins()).astype(int)
        rng.shuffle(truth)
        for i, p in enumerate(sheets, start=1):
            df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
            vals = truth.copy()
            n_noise = {1: 6, 2: 18, 3: 30}[i]
            idx = rng.choice(len(vals), size=n_noise, replace=False)
            vals[idx] = rng.integers(1, n_bins() + 1, size=n_noise)
            s = pd.Series([str(int(v)) for v in vals])
            n_missing = {1: 0, 2: 8, 3: 15}[i]
            if n_missing:
                s.iloc[rng.choice(len(s), size=n_missing, replace=False)] = ""
            df["difficulty_1to5"] = s.to_numpy()
            df["comment"] = ""
            df.to_csv(sim / p.name, index=False, encoding="utf-8-sig")

        rc = cmd_agree(argparse.Namespace(out_dir=str(sim), key=str(tmpdir / KEY_FILE),
                                          sheets=None))
        check("agree 返回 0", rc == 0)
        js = json.loads((sim / KAPPA_JSON).read_text(encoding="utf-8"))["agreement"]
        check("agree 报告了有效样本量（少于 100，因有人留空）",
              0 < js["n_effective"] < int(C.HUMAN_KAPPA_N_ITEMS),
              f"有效 N={js['n_effective']}，剔除 {js['n_dropped_missing']}")
        check("自实现与 statsmodels 的 Fleiss κ 一致", js["fleiss_kappa_abs_diff"] < 1e-9,
              f"差值 {js['fleiss_kappa_abs_diff']:.2e}")
        check("模拟数据下 κ 落在合理区间 (0.3, 1.0)",
              0.3 < js["fleiss_kappa"] < 1.0,
              f"Fleiss κ={js['fleiss_kappa']:.3f}，平均 QWK={js['mean_pairwise_qwk']:.3f}，"
              f"原始一致率={js['raw_agreement']:.3f}")
        check("两两配对给出 3 组", len(js["pairs"]) == 3)
        md_text = (sim / KAPPA_MD).read_text(encoding="utf-8")
        check("报告含条件化结论分支规则", "Fleiss κ ≥ 0.600" in md_text and "Fleiss κ < 0.400" in md_text)
        expect_n = int(pd.concat(
            [pd.read_csv(sim / p.name, encoding="utf-8-sig")["difficulty_1to5"]
             for p in sheets], axis=1).notna().all(axis=1).sum())
        check("有效样本量 = 所有评题人均评过的题数（未做任何填补）",
              js["n_effective"] == expect_n, f"{js['n_effective']} vs {expect_n}")

        log("自检 4/4：跑 compare（对 D 格预测缓存）")
        rc = cmd_compare(argparse.Namespace(
            out_dir=str(sim), key=str(tmpdir / KEY_FILE), sheets=None,
            condition="D", model=None, seed=None, i_know_its_not_D=False,
        ))
        check("compare 返回 0", rc == 0)
        payload = json.loads((sim / KAPPA_JSON).read_text(encoding="utf-8"))["human_vs_model"]
        if payload.get("skipped"):
            check("compare 在无 D 格缓存时给出明确跳过提示（未用 A 格冒充）",
                  "不得" in payload["reason"], payload["reason"][:70] + "…")
        else:
            row = payload["rows"][0]
            check("compare 完成模型对比且可比 N > 0", row["n_compared"] > 0,
                  f"{row['model']} seed={row['seed']}，N={row['n_compared']}，"
                  f"一致率={row['agreement']:.3f}，κ={row['cohen_kappa']:.3f}，"
                  f"QWK={row['qwk']:.3f}")
            print("      ⚠ 模拟填答与真实 rating 无关，故上面的一致率/κ 数值本身没有意义，"
                  "这里只验证代码路径与口径（真实数值要等评题表回收后才有）")
            check("npz 与 key.csv/dataset.csv 的 rating 逐题核对无不一致",
                  all(a["n_rating_mismatch"] == 0 for a in payload["alignment"]))
            check("报告含人类内部一致率作为上界参照",
                  payload["human_internal_on_overlap"]["raw_agreement"] is not None,
                  f"人类内部一致率="
                  f"{payload['human_internal_on_overlap']['raw_agreement']:.3f}")
            md_text = (sim / KAPPA_MD).read_text(encoding="utf-8")
            check("kappa_results.md 同时含人类 κ 与模型对比两节",
                  "Fleiss" in md_text and "人类 vs 模型一致率" in md_text
                  and "混淆矩阵" in md_text)

        if args.keep:
            keep = Path(args.keep)
            keep.mkdir(parents=True, exist_ok=True)
            for f in sorted(tmpdir.rglob("*")):
                if f.is_file():
                    (keep / f.name).write_bytes(f.read_bytes())
            log(f"自检产物已保留到 {keep}")

    print()
    if ok:
        log("自检全部通过。⚠ 产物写在系统临时目录并已删除，未触碰 outputs/human_kappa/。")
        return 0
    log("自检存在失败项，见上方 [✗]。")
    return 1


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="human_kappa.py",
        description="人类标注一致性（human κ）实验工具链：分层抽样 / 一致性计算 / 与 D 格模型对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/human_kappa.py sample\n"
            "  # 回收评题表后放到 outputs/human_kappa/ 再运行：\n"
            "  python scripts/human_kappa.py agree\n"
            "  python scripts/human_kappa.py compare\n"
            "  python scripts/human_kappa.py selftest   # 无真实填写数据时的自检（临时目录）\n"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="按 rating 段分层抽题并生成评题表")
    p.add_argument("--n-items", type=int, default=C.HUMAN_KAPPA_N_ITEMS,
                   help=f"抽样题数（默认 config.HUMAN_KAPPA_N_ITEMS={C.HUMAN_KAPPA_N_ITEMS}）")
    p.add_argument("--n-raters", type=int, default=C.HUMAN_KAPPA_N_RATERS,
                   help=f"评题人数（默认 config.HUMAN_KAPPA_N_RATERS={C.HUMAN_KAPPA_N_RATERS}）")
    p.add_argument("--seed", type=int, default=C.SPLIT_SEED,
                   help=f"抽样随机种子（默认 config.SPLIT_SEED={C.SPLIT_SEED}）")
    p.add_argument("--new-frac", type=float, default=0.5,
                   help="每层中“新题”（year ≥ config.TEMPORAL_TEST_START_YEAR）的目标占比，"
                        "默认 0.5；新题同时就是 D 格测试集里的题，占比太低会削弱 compare")
    p.add_argument("--dataset", default=str(C.DATASET_PATH), help="数据集 CSV 路径")
    p.add_argument("--out-dir", default=str(KAPPA_DIR), help="输出目录")
    p.add_argument("--force", action="store_true",
                   help="即使已有填写内容也重新抽样（会覆盖答卷！）")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("agree", help="计算人类一致性（Fleiss/Cohen/QWK/Krippendorff α）")
    p.add_argument("--sheet", dest="sheets", nargs="*", default=None,
                   help=f"评题表路径或通配符；默认 <out-dir>/{SHEET_PATTERN}")
    p.add_argument("--key", default=None, help=f"key.csv 路径（默认 <out-dir>/{KEY_FILE}）")
    p.add_argument("--out-dir", default=str(KAPPA_DIR), help="输出目录")
    p.set_defaults(func=cmd_agree)

    p = sub.add_parser("compare", help="人类共识 vs 模型预测（只能对 D 格）")
    p.add_argument("--condition", default="D", help="因子格，默认且通常只能是 D")
    p.add_argument("--model", nargs="*", default=None, help="只比较指定模型（默认自动扫描）")
    p.add_argument("--seed", nargs="*", type=int, default=None, help="只比较指定种子")
    p.add_argument("--sheet", dest="sheets", nargs="*", default=None,
                   help="评题表路径/通配符")
    p.add_argument("--key", default=None, help=f"key.csv 路径（默认 <out-dir>/{KEY_FILE}）")
    p.add_argument("--out-dir", default=str(KAPPA_DIR), help="输出目录")
    p.add_argument("--i-know-its-not-D", dest="i_know_its_not_D", action="store_true",
                   help="显式确认要用非 D 格比较（仅敏感性分析，不能进正文主结论）")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("selftest", help="开发自检：临时目录模拟填答，跑通 agree/compare")
    p.add_argument("--keep", default=None, help="把自检产物另存到该目录（默认用完即删）")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "out_dir"):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if getattr(args, "key", None) is None:
            args.key = str(out_dir / KEY_FILE)
    issues = C.validate()
    if issues:
        for i in issues:
            log(f"[口径自检失败] {i}")
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
