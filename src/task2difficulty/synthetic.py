"""合成数据集：在真实数据抓取完成前，让整条流水线可端到端自测。

为什么需要它
------------
真实数据管线（API + 爬题面 + 清洗）跑完之前，模型层、因子设计、指标层都无法验证。
更关键的是：**必须有一个已知真值（ground truth）的合成数据**来验证
"统计特征置空后性能确实下降""时间划分确实产生泛化损失"这类结论逻辑是否成立——
如果合成数据上因子设计都跑不出预期方向，说明实验脚本本身有 bug。

设计要点：合成数据按**可控机制**生成，使得各因子的预期效应方向已知：
  - `stat` 特征携带部分标签信息 → 置空后性能必然下降（对应 A>B、C>D）；
  - 题面文本携带部分标签信息 → 纯文本模型有非零能力（对应 B、D 格）；
  - 标签与约束特征携带少量信息 → 纯冷启动下结构化通道仍有**残余增益**（融合模型 > 纯文本）；
  - 时间漂移：rating 生成机制随年份缓慢变化 → 时间划分产生泛化损失（对应 B>D）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

#: 合成数据的标签词表（与真实 CF 标签同名的子集，便于可视化与分组分析）
SYNTHETIC_TAGS: tuple[str, ...] = (
    "implementation", "math", "greedy", "dp", "data structures", "brute force",
    "constructive algorithms", "graphs", "sortings", "binary search",
    "strings", "trees", "number theory", "combinatorics", "geometry",
)


@dataclass
class SyntheticSpec:
    n: int = 1200
    year_start: int = 2013
    year_end: int = 2026
    seed: int = 0
    #: 统计特征对标签的解释比例（0~1）；越大则置空后下降越明显
    stat_signal: float = 0.55
    #: 文本信号强度
    text_signal: float = 1.0
    #: 结构化（标签+约束）信号强度；决定纯冷启动下的"残余增益"
    struct_signal: float = 0.35
    #: 时间漂移强度：rating 生成机制随年份的漂移幅度
    drift: float = 0.6
    #: 下钳位样本占比（模拟 CF 的 800 下限堆积）
    floor_share: float = 0.06


def make_synthetic(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """生成合成题目表，列结构与 `dataset.load_dataset` 期望的一致。"""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n

    years = rng.integers(spec.year_start, spec.year_end + 1, size=n)

    # ---- 潜在难度（未观测的真值），先归一化到 [0,1] ----
    latent = rng.beta(2.2, 2.2, size=n)

    # ---- 时间漂移：rating 的映射随年份缓慢移动（越晚略偏难，模拟出题方变化）----
    drift_term = spec.drift * (years - years.mean()) / max(1.0, years.std())

    # ---- 结构性信号：标签数越多越难，约束规模越大越难 ----
    tag_count = rng.integers(1, 6, size=n)
    constraint_n = np.round(10 ** rng.uniform(1, 6, size=n)).astype(np.int64)
    multi_test = rng.integers(0, 2, size=n)
    is_interactive = (rng.random(n) < 0.06).astype(np.int64)

    struct_signal = (
        0.18 * (tag_count - tag_count.mean()) / max(1e-9, tag_count.std())
        + 0.22 * (np.log10(constraint_n) - np.log10(constraint_n).mean())
        / max(1e-9, np.log10(constraint_n).std())
        + 0.10 * is_interactive
    ) * spec.struct_signal

    # ---- 标签：由潜在难度 + 噪声决定 ----
    z = latent + drift_term * 0.35 + struct_signal
    z = (z - z.mean()) / max(1e-9, z.std())
    rating = C.RATING_MIN + (z - z.min()) / max(1e-9, z.max() - z.min()) * (
        C.RATING_MAX - C.RATING_MIN
    )
    # 落到 100 步长并施加 800 下钳位（真实 CF 的标签事实）
    rating = np.clip(np.round(rating / C.RATING_STEP) * C.RATING_STEP,
                     C.RATING_FLOOR, C.RATING_MAX)
    n_floor = int(round(spec.floor_share * n))
    if n_floor:
        idx = rng.choice(n, size=n_floor, replace=False)
        rating[idx] = C.RATING_FLOOR

    # ---- 统计特征：与标签同源（这正是大纲要正面处理的泄漏来源）----
    easy = (rating - rating.mean()) / max(1e-9, rating.std())
    solved_count = np.exp(
        np.log(3000) - spec.stat_signal * 1.6 * easy + rng.normal(0, 0.9, size=n)
    ).round().astype(np.int64)
    attempts = solved_count + np.exp(rng.normal(7.0, 0.8, size=n)).astype(np.int64)
    acceptance_rate = solved_count / np.maximum(1, attempts)
    days_since_publish = (2026 - years) * 365 + rng.integers(0, 365, size=n)

    # ---- 题面文本：由潜在难度决定用哪些模板句，注入"捷径"信息 ----
    statements = [
        _synth_statement(rng, float(latent[i]), int(constraint_n[i]),
                         bool(is_interactive[i]), spec.text_signal)
        for i in range(n)
    ]

    # ---- 算法标签 multi-hot ----
    tags = [_pick_tags(rng, float(latent[i])) for i in range(n)]

    # ---- 来源特征（高危：题号与 Div 是标签的强代理）----
    # 刻意让题号 index 与 Div 携带真实预测力，以检验 provenance 消融能否探测到
    div_type = np.where(rating > 2000, 1, np.where(rating > 1400, 2, 3))
    hard_rank = np.argsort(np.argsort(rating + rng.normal(0, 30, size=n)))
    idx_letters = np.array(list("ABCDEFGH"))[
        np.clip((hard_rank / max(1, n) * 6).astype(int), 0, 7)
    ]

    df = pd.DataFrame({
        "contest_id": rng.integers(1, 2000, size=n),
        "problem_index": idx_letters,
        "problem_key": [f"{cid}{ix}" for cid, ix in
                        zip(rng.integers(1, 2000, size=n), idx_letters)],
        "name": [f"Synthetic Problem {i}" for i in range(n)],
        "year": years,
        "div_type": div_type,
        "rating": rating.astype(int),
        "statement": statements,
        "solved_count": solved_count,
        "acceptance_rate": acceptance_rate,
        "days_since_publish": days_since_publish,
        "constraint_n_upper": constraint_n,
        "multi_test_count": multi_test,
        "is_interactive": is_interactive,
        "tag_count": tag_count,
        "snapshot_date": "SYNTHETIC",
    })
    for t in SYNTHETIC_TAGS:
        col = f"{C.TAG_MULTIHOT_PREFIX}{t.replace(' ', '_')}"
        df[col] = [int(t in ts) for ts in tags]
    # 让 problem_key 唯一
    df["problem_key"] = [f"{c}-{i}-{k}" for c, i, k in
                         zip(df["contest_id"], df["problem_index"], range(n))]
    return df


def _pick_tags(rng: np.random.Generator, latent: float) -> list[str]:
    """按难度倾向选择标签：dp/graph/math 偏难，implementation 偏易。"""
    easy_tags = ["implementation", "brute force", "sortings", "strings"]
    mid_tags = ["math", "greedy", "binary search", "constructive algorithms",
                "number theory", "combinatorics"]
    hard_tags = ["dp", "graphs", "data structures", "trees", "geometry"]
    pool = easy_tags if latent < 0.35 else (mid_tags if latent < 0.7 else hard_tags)
    others = [t for t in SYNTHETIC_TAGS if t not in pool]
    k = int(rng.integers(1, 4))
    chosen = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    if rng.random() < 0.25:
        chosen.append(str(rng.choice(others)))
    return chosen


def _synth_statement(rng: np.random.Generator, latent: float, n_upper: int,
                     interactive: bool, signal: float) -> str:
    """合成题面。刻意混入模板句，供文本端 provenance 消融使用。"""
    parts: list[str] = []

    # 模板句：与难度相关的"捷径"
    if latent > 0.72 and signal > 0:
        parts.append(rng.choice([
            "This is the hard version of the problem.",
            "The only difference between the versions is the constraints.",
        ]))
    elif latent < 0.30 and signal > 0:
        parts.append(rng.choice([
            "This is an easy version of the problem.",
            "This problem is straightforward.",
        ]))
    if interactive:
        parts.append(rng.choice([
            "This problem is interactive.",
            "You must interact with the interactor.",
        ]))

    # 主体描述：难度越高句数越多、词汇越"抽象"
    n_sent = int(rng.integers(2, 4 + int(latent * 8)))
    subjects = ["array", "graph", "tree", "permutation", "string", "matrix",
                "sequence", "set", "grid", "directed graph"]
    verbs = ["find", "compute", "determine", "count", "minimize", "maximize"]
    for _ in range(n_sent):
        s = str(rng.choice(subjects))
        v = str(rng.choice(verbs))
        if latent > 0.6 and signal > 0:
            extra = str(rng.choice(["under modulo 998244353",
                                    "in O(n log n) time",
                                    "using dynamic programming",
                                    "with exactly k operations",
                                    "such that the answer is minimized over all subsets"]))
        else:
            extra = str(rng.choice(["in ascending order", "for each query",
                                    "where 1 <= a_i <= n", "in a single line"]))
        parts.append(f"You are given a {s}. {v.capitalize()} the required value {extra}.")

    parts.append(f"The constraints are 1 <= n <= {n_upper} and 1 <= t <= 10000.")
    return " ".join(parts)
