"""数据集与公平协议（大纲二·（五））。

三条不可违反的规则，全部在此强制
--------------------------------
1. **共用同一索引文件**：所有模型从 `data/processed/split_index.csv` 读划分，
   任何模型模块都不得自行 `train_test_split`；
2. **时间划分三段式**：训练<=2021 / 验证=2022 / 测试>=2023。
   若允许"从 2022 年后的数据里随机抽验证集"，早停与超参就看到了测试期的未来信息，
   A/B/D 三格相对 C 的无泄漏性会全部作废——因此这里物理上不提供随机抽验证集的入口；
3. **统计特征置空 = NaN，不是 0**：0 伪造了"这题没人做过"这一事实，
   NaN 才表示"信息不存在"。LightGBM/HistGradientBoosting 原生处理 NaN，
   线性模型与神经网络侧需要显式填补（见 `impute_for_model`），
   填补策略本身是特征可得性实验的一部分，必须记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------- 数据集容器


@dataclass
class DifficultyDataset:
    """题目难度数据集。

    `X` 只含数值/结构化特征；题面原文在 `text`，避免它被误当作数值特征。
    `meta` 保留不可入模的标识与来源列，供 provenance 消融与分组分析使用。
    """

    X: pd.DataFrame
    y: np.ndarray
    text: pd.Series
    meta: pd.DataFrame
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    snapshot_date: str = ""

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y) or len(self.X) != len(self.text):
            raise ValueError(
                f"X/y/text 长度不一致: {len(self.X)}/{len(self.y)}/{len(self.text)}"
            )
        if self.meta.index.tolist() != self.X.index.tolist():
            raise ValueError("meta 与 X 的索引不一致")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def groups(self) -> dict[str, list[str]]:
        return self.feature_groups

    def group_columns(self, group: str) -> list[str]:
        cols = self.feature_groups.get(group, [])
        missing = [c for c in cols if c not in self.X.columns]
        if missing:
            raise KeyError(f"特征组 {group} 声明的列不存在于 X：{missing[:5]}…")
        return list(cols)

    def select(self, idx: np.ndarray | Sequence[int]) -> "DifficultyDataset":
        idx = np.asarray(idx)
        return DifficultyDataset(
            X=self.X.iloc[idx].reset_index(drop=True),
            y=self.y[idx],
            text=self.text.iloc[idx].reset_index(drop=True),
            meta=self.meta.iloc[idx].reset_index(drop=True),
            feature_groups={k: list(v) for k, v in self.feature_groups.items()},
            snapshot_date=self.snapshot_date,
        )

    # ------------------------------------------------------------ 特征可得性

    def with_feature_condition(
        self,
        stat_condition: str,
        drop_provenance: bool = True,
        drop_groups: Iterable[str] = (),
        keep_groups: Iterable[str] | None = None,
    ) -> "DifficultyDataset":
        """返回指定"特征可得性"下的数据集副本。

        Parameters
        ----------
        stat_condition:
            - `full`：统计特征用快照值（A、C 格）；
            - `masked`：统计特征整组置空（B、D 格）；**整列删除**而非填 0，
              这样下游必须显式决定如何处理，不会静默把缺失当 0；
            - `in_contest`：S1' 赛时统计（调用方须已在数据里准备好 `_s1` 后缀列）。
        drop_provenance:
            是否剔除来源特征（题号/Div/contestId）。主实验恒为 True，
            仅在 provenance 消融时置 False。
        drop_groups:
            额外整组剔除的特征组（用于实验 3 的特征消融）。
        keep_groups:
            若给出，仅保留这些组（用于"仅文本""仅结构化"等配置）。
        """
        drop: set[str] = set()
        rename: dict[str, str] = {}

        if stat_condition == C.STAT_MASKED:
            drop.update(self.group_columns("stat"))
        elif stat_condition == C.STAT_IN_CONTEST:
            # S1' 列由 fetcher 写入 dataset，命名约定：<stat 列>__s1。
            # 只对**基础统计特征**（config.STAT_FEATURES 里声明的两项）做改名启用；
            # 其余 `__s1` 列（如 acceptance_rate__s1 / attempts_in_contest__s1）
            # 是 S1' 专属的增量特征，没有对应的基础列名可映射，因此直接保留原名。
            renamed_any = False
            for col in C.STAT_FEATURES:
                s1 = f"{col}__s1"
                if s1 in self.X.columns:
                    drop.add(col)
                    rename[s1] = col
                    renamed_any = True
            if not renamed_any:
                raise KeyError(
                    "S1' 赛时统计列一个都没找到（期望 <stat 列>__s1）。"
                    "请先运行 python scripts/build_s1_in_contest.py，"
                    "再用 python scripts/build_dataset.py --skip-crawl 重建数据集。"
                )
        elif stat_condition != C.STAT_FULL:
            raise ValueError(f"未知统计特征条件: {stat_condition!r}")

        # ★ 关键：非 S1' 条件下，必须**显式丢弃所有 `*__s1` 列**。
        # 它们虽在 stat 组里却不属于 C.STAT_FEATURES（那是"置空要移除的集合"），
        # 因此只 drop group_columns("stat") 是**删不掉** S1' 列的——
        # 结果就是 A/B/C/D 四格全都悄悄拿到了赛时统计，而 `__s1` 与真实统计高度共线，
        # 表面只会"性能变好"，极难察觉（本项已由 experiments.assert_no_stat_leak 拦截）。
        if stat_condition != C.STAT_IN_CONTEST:
            drop.update({c for c in self.X.columns if c.endswith("__s1")})

        if drop_provenance:
            drop.update(self.group_columns("provenance"))

        for g in drop_groups:
            drop.update(self.group_columns(g))

        if keep_groups is not None:
            allowed: set[str] = set()
            for g in keep_groups:
                allowed.update(self.group_columns(g))
            drop.update(set(self.X.columns) - allowed)

        keep_cols = [c for c in self.X.columns if c not in drop]
        X = self.X[keep_cols].rename(columns=rename)
        groups = {
            g: [rename.get(c, c) for c in cols if rename.get(c, c) in keep_cols]
            for g, cols in self.feature_groups.items()
        }
        return DifficultyDataset(
            X=X, y=self.y, text=self.text, meta=self.meta,
            feature_groups=groups, snapshot_date=self.snapshot_date,
        )

    def impute_for_model(self, kind: str = "median") -> pd.DataFrame:
        """本方法**故意不实现**：在整表上做全局填补会泄漏验证/测试信息。

        请改用 `impute_with_train_stats(X_train, X_other, kind)`：
        填补统计量只在训练集上算，再原样套用到验证/测试集。
        """
        raise NotImplementedError(
            "请使用 impute_with_train_stats(X_train, X_other, kind)；"
            "在这里做全局填补会泄漏验证/测试信息。"
        )


def impute_with_train_stats(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    kind: str = "median",
    *,
    stat_condition: str = C.STAT_FULL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """用**训练集**统计量填补，返回 (X_train_filled, X_other_filled, fill_values)。

    ⚠⚠ **仅可用于统计特征可得的条件（A / C 格）。冷启动条件（B / D 格）绝对禁用。**

    原因（这是本文最隐蔽的一处自泄漏）：统计特征被置空后必是整列 NaN，
    而"训练集中位数"本身由训练集的统计特征算出——用它填补等于把
    "训练集里这道题的累计通过量长什么样"偷偷塞回冷启动场景，
    会系统性高估 B/D 格，**直接污染 A−B 与 B−D 两个核心差分**。

    冷启动格的正确做法由 `experiments.prepare_struct_matrix` 实现：
    **常量哨兵 0 + 显式缺失指示列**（纯常量变换，不依赖训练集任何统计量）。

    为防误用，本函数在 `stat_condition` 为置空时会直接抛错，而不是静默照做。
    """
    if stat_condition in (C.STAT_MASKED, C.STAT_IN_CONTEST):
        raise ValueError(
            f"stat_condition={stat_condition!r} 禁止使用训练集统计量填补："
            f"中位数/均值来自训练集的统计特征，会污染 A−B 与 B−D 差分。"
            f"冷启动格请改用 experiments.prepare_struct_matrix 的"
            f"「常量哨兵 + 缺失指示列」方案。"
        )
    if kind == "median":
        fill = X_train.median(numeric_only=True)
    elif kind == "mean":
        fill = X_train.mean(numeric_only=True)
    elif kind == "zero":
        fill = pd.Series(0.0, index=X_train.columns)
    else:
        raise ValueError(f"未知填补方式: {kind!r}")

    # 训练集整列全 NaN 时 median/mean 仍是 NaN。此时退回 0，
    # 但**必须显式告警**——静默填 0 等于伪造"该特征的典型值为 0"。
    all_nan = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_nan:
        import warnings

        warnings.warn(
            f"以下列在训练集上全为 NaN，已退回填 0：{all_nan[:8]}"
            f"{'…' if len(all_nan) > 8 else ''}。"
            f"请确认这不是「统计特征被置空却仍走了本函数」的情形。",
            RuntimeWarning, stacklevel=2,
        )
    fill = fill.fillna(0.0)
    return (
        X_train.fillna(fill).astype(np.float64),
        X_other.fillna(fill).astype(np.float64),
        fill,
    )


# ---------------------------------------------------------------- 划分协议


@dataclass
class SplitIndex:
    """两套划分的索引容器。所有模型必须从这里取索引。"""

    problem_key: pd.Series          # (contestId, index) 复合键，用于与原始数据校验
    random_split: pd.Series         # 'train' | 'valid' | 'test'
    temporal_split: pd.Series
    year: pd.Series                 # 题目所属比赛年份

    def mask(self, split_kind: str, part: str) -> np.ndarray:
        """返回布尔掩码。"""
        if split_kind == C.SPLIT_RANDOM:
            s = self.random_split
        elif split_kind == C.SPLIT_TEMPORAL:
            s = self.temporal_split
        else:
            raise ValueError(f"未知划分方式: {split_kind!r}")
        if part not in ("train", "valid", "test"):
            raise ValueError(f"未知子集: {part!r}")
        return (s == part).to_numpy()

    def indices(self, split_kind: str, part: str) -> np.ndarray:
        return np.flatnonzero(self.mask(split_kind, part))

    def ratios(self, split_kind: str) -> dict[str, float]:
        return {
            part: float(self.mask(split_kind, part).mean())
            for part in ("train", "valid", "test")
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "problem_key": self.problem_key,
            "year": self.year,
            "random_split": self.random_split,
            "temporal_split": self.temporal_split,
        })


#: 划分索引文件的固定列名
SPLIT_INDEX_COLUMNS = ("problem_key", "year", "random_split", "temporal_split")


def make_split_index(
    year: pd.Series,
    problem_key: pd.Series,
    seed: int = C.SPLIT_SEED,
) -> SplitIndex:
    """生成两套划分。**时间划分严格三段式，验证集按时间切**。

    随机划分用固定种子的置换，保证可复现且与模型种子无关。
    """
    year = pd.Series(year).reset_index(drop=True)
    problem_key = pd.Series(problem_key).reset_index(drop=True)
    if year.isna().any():
        raise ValueError(f"存在 {int(year.isna().sum())} 条缺失年份的记录，无法做时间划分")

    n = len(year)

    # ---- 随机划分：固定种子置换后按 8:1:1 切
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * C.RANDOM_SPLIT[0]))
    n_valid = int(round(n * C.RANDOM_SPLIT[1]))
    random_split = np.empty(n, dtype=object)
    random_split[perm[:n_train]] = "train"
    random_split[perm[n_train:n_train + n_valid]] = "valid"
    random_split[perm[n_train + n_valid:]] = "test"

    # ---- 时间划分：三段式，验证集 = 指定年当年（不得随机抽样）
    temporal = np.where(
        year.to_numpy() <= C.TEMPORAL_TRAIN_END_YEAR, "train",
        np.where(year.to_numpy() == C.TEMPORAL_VALID_YEAR, "valid", "test"),
    )
    for part in ("train", "valid", "test"):
        if (temporal == part).sum() == 0:
            yc = year.value_counts().sort_index()
            spread = ", ".join(f"{int(y)}:{int(n)}" for y, n in yc.items())
            raise ValueError(
                f"时间划分的 {part} 段为空，检查年份分布与边界设置。\n"
                f"  边界：训练 <= {C.TEMPORAL_TRAIN_END_YEAR} / "
                f"验证 = {C.TEMPORAL_VALID_YEAR} / 测试 >= {C.TEMPORAL_TEST_START_YEAR}\n"
                f"  实际年份分布：{spread}\n"
                f"  提示：若用 --limit 按时间取前 N 题，验证段会为空；请改用跨年份抽样。"
            )

    return SplitIndex(
        problem_key=problem_key,
        random_split=pd.Series(random_split, dtype="category"),
        temporal_split=pd.Series(temporal, dtype="category"),
        year=year.astype(int),
    )


def check_split_ratios(si: SplitIndex) -> list[str]:
    """对照大纲【十】的实测占比做自检，返回告警列表。"""
    warns: list[str] = []
    for kind, expected in ((C.SPLIT_RANDOM, C.EXPECTED_RANDOM_RATIOS),
                           (C.SPLIT_TEMPORAL, C.EXPECTED_TEMPORAL_RATIOS)):
        got = si.ratios(kind)
        for part, exp in expected.items():
            diff_pp = abs(got[part] - exp) * 100
            if diff_pp > C.RATIO_TOLERANCE_PP:
                warns.append(
                    f"{kind}/{part} 占比 {got[part]:.1%} 与大纲实测 {exp:.1%} "
                    f"相差 {diff_pp:.1f} pp（容差 {C.RATIO_TOLERANCE_PP} pp）"
                )
    return warns


def save_split_index(si: SplitIndex, path: Path | str = C.SPLIT_INDEX_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    si.to_frame().to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_split_index(path: Path | str = C.SPLIT_INDEX_PATH) -> SplitIndex:
    df = pd.read_csv(path)
    missing = [c for c in SPLIT_INDEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"划分索引文件缺列: {missing}")
    return SplitIndex(
        problem_key=df["problem_key"],
        random_split=df["random_split"].astype("category"),
        temporal_split=df["temporal_split"].astype("category"),
        year=df["year"].astype(int),
    )


# ---------------------------------------------------------------- 数据集读写


#: 题号字母 -> 序号（A=0, B=1, …）。来源特征必须可入模，否则 provenance 消融
#: 与弱基线自曝（大纲四·（四））在岭回归/MLP 上会直接报 "could not convert string to float"。
_INDEX_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode_problem_index(s: pd.Series) -> pd.Series:
    """把题号 index（'A'…'Z' / 可能带数字）编码为有序整数。

    语义：题号字母顺序本身就是难度序（A 最简单），因此用**有序编码**而非 one-hot，
    这与大纲"题号 index 是标签的强代理"的判断一致——有序编码保留了这个单调性，
    正是弱基线要探测的东西。
    """
    def one(v: object) -> float:
        t = str(v).strip().upper()
        if not t:
            return np.nan
        if t[0] in _INDEX_LETTERS:
            return float(_INDEX_LETTERS.index(t[0]))
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(one).astype(np.float64)


def load_dataset(path: Path | str = C.DATASET_PATH) -> DifficultyDataset:
    """从清洗后的 CSV 读入数据集。

    期望列：
      - `contest_id`, `problem_index`, `problem_key`, `year` 等标识列
      - `rating`（标签）
      - `statement`（清洗后的题面文本）
      - 其余为特征列；`tag__*` 为算法标签 multi-hot
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("problem_key", "rating", "statement"):
        if col not in df.columns:
            raise ValueError(f"数据集缺少必需列: {col}")

    # 来源特征里的题号是字母，必须先转成有序数值才能进入任何模型
    if "problem_index" in df.columns:
        df["problem_index"] = encode_problem_index(df["problem_index"])

    y = df["rating"].to_numpy(dtype=np.float64)
    text = df["statement"].fillna("")

    id_cols = [c for c in ("contest_id", "problem_index", "problem_key",
                           "year", "name", "div_type", "snapshot_date") if c in df.columns]
    meta = df[id_cols].copy()

    tag_cols = [c for c in df.columns if c.startswith(C.TAG_MULTIHOT_PREFIX)]
    # S1' 列（`*__s1`）单独识别：它们**绝不能**混进普通特征矩阵，
    # 只在 stat_condition='in_contest' 时由 with_feature_condition 改名启用。
    # 若误把它们当普通特征，A/B/C/D 四格就都悄悄拿到了赛时统计，
    # 泄漏讨论与全部差分一次性作废（这正是本文最忌讳的那类静默错误）。
    s1_cols = [c for c in df.columns if c.endswith("__s1")]
    feature_groups = {
        "stat": [c for c in C.STAT_FEATURES if c in df.columns] + s1_cols,
        "constraint": [c for c in C.CONSTRAINT_FEATURES if c in df.columns],
        "tag": tag_cols + ([C.TAG_COUNT_FEATURE] if C.TAG_COUNT_FEATURE in df.columns else []),
        "provenance": [c for c in C.PROVENANCE_FEATURES if c in df.columns],
        "text": [],  # 文本特征由 vectorizers/transformers 在训练时构造，不落在 CSV 里
    }
    feature_cols = [c for g in ("stat", "constraint", "tag", "provenance")
                    for c in feature_groups[g]]
    if not feature_cols:
        raise ValueError("未识别到任何结构化特征列，检查列名前缀与 config 中的特征定义")

    snapshot = ""
    if "snapshot_date" in df.columns and len(df):
        snapshot = str(df["snapshot_date"].iloc[0])

    return DifficultyDataset(
        X=df[feature_cols].copy(),
        y=y,
        text=text,
        meta=meta,
        feature_groups=feature_groups,
        snapshot_date=snapshot,
    )


def dataset_summary(ds: DifficultyDataset) -> dict[str, Any]:
    """表 1「数据集统计信息」的素材。"""
    rating = pd.Series(ds.y)
    tag_cols = ds.feature_groups.get("tag", [])
    tag_cols = [c for c in tag_cols if c.startswith(C.TAG_MULTIHOT_PREFIX)]
    tag_counts: Mapping[str, int] = {}
    if tag_cols:
        tag_counts = (
            ds.X[tag_cols].sum().sort_values(ascending=False).head(10)
            .rename(lambda s: s[len(C.TAG_MULTIHOT_PREFIX):]).to_dict()
        )
    text_len = ds.text.str.len()
    return {
        "n_problems": len(ds),
        "snapshot_date": ds.snapshot_date,
        "rating_min": float(rating.min()),
        "rating_max": float(rating.max()),
        "rating_mean": float(rating.mean()),
        "rating_std": float(rating.std()),
        "rating_floor_share": float((rating <= C.RATING_FLOOR).mean()),
        "n_tags": len(tag_cols),
        "tag_top10": {k: int(v) for k, v in tag_counts.items()},
        "text_len_mean": float(text_len.mean()),
        "text_len_median": float(text_len.median()),
        "text_len_p95": float(text_len.quantile(0.95)),
    }
