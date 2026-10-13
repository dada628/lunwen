"""评价指标（大纲三·（五））。

核心口径，逐条对应大纲红线
--------------------------
1. **指标恰好 5 项**：MAE、±200 命中率、macro-F1、QWK、Spearman ρ。
   RMSE/R²（与 MAE 信息重叠）、Cliff's delta、NDCG 一律不出现。
2. **分类指标不单独训练**：macro-F1 与 QWK 由**同一回归输出**按阈值分箱派生。
3. **分箱阈值只在验证集上确定**，再原样套用到测试集。随机划分取随机划分的
   10% 验证集，时间划分取 2022 年验证集——两套划分各自独立。
4. **显著性以配对 bootstrap 95% CI 为主**，附 p 值。万级样本下 p 值几乎必然显著，
   只报 p 值会被质疑"只赢一个百分点"，故必须同时给效应幅度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from . import config as C

# ---------------------------------------------------------------- 分箱


@dataclass(frozen=True)
class BinThresholds:
    """难度分档阈值。由验证集确定，冻结后套用测试集。"""

    edges: tuple[float, ...]      # 长度 = 档数 + 1，含 ±inf 端点
    scheme: str                   # 'equal_width' | 'equal_freq'
    fitted_on: str = "valid"      # 定阈值的来源，写进正文以备复核

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def apply(self, rating: np.ndarray) -> np.ndarray:
        """把连续 rating 映射为 0-based 档位索引。

        ★ 分档约定：**左开右闭**（大纲二·（五）的"≤1200 / ≤1600 / ≤2000 / ≤2400 / >2400"）。
        即 rating 恰好等于切点时归入**较低**那一档（1200 → 第 1 档，不是第 2 档）。

        为什么这里必须显式写 `right=True`：`np.digitize` 的默认 `right=False` 是
        **左闭右开**，会让 rating 恰为 1200 的题落进第 2 档，与大纲措辞相反。
        本任务 rating 以 100 为步长、切点又都是 100 的整数倍，**边界命中率极高**
        （实测 391 题中 63 题、16.1% 正压在 1200/1600/2000/2400 上），
        约定取反会让最低档人数在 88 与 105 之间摆动——足以影响 macro-F1 与 QWK。
        因此三处调用点（metrics / figures / models_tabular）必须保持一致。
        """
        return np.clip(
            np.digitize(rating, self.edges[1:-1], right=True),
            0, self.n_bins - 1,
        )

    def to_dict(self) -> dict[str, object]:
        return {"edges": list(self.edges), "scheme": self.scheme, "fitted_on": self.fitted_on}


def fit_bin_thresholds(
    y_valid: np.ndarray,
    scheme: str = "equal_width",
    n_bins: int | None = None,
) -> BinThresholds:
    """在**验证集**上确定分档阈值。

    equal_width: 按 config.DIFFICULTY_BINS 的等宽切分（≤1200 / ~1600 / ~2000 / ~2400 / >2400）。
                 固定口径，跨实验可比，是主口径。
    equal_freq : 等频切分，按验证集分位数确定，作为类别不平衡的敏感性分析。
    """
    y_valid = np.asarray(y_valid, dtype=np.float64)
    if n_bins is None:
        n_bins = len(C.DIFFICULTY_BINS) + 1

    if scheme == "equal_width":
        interior = list(C.DIFFICULTY_BINS)
        if len(interior) != n_bins - 1:
            raise ValueError(
                f"等宽口径要求 {n_bins - 1} 个内部切点，config 给了 {len(interior)} 个"
            )
    elif scheme == "equal_freq":
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        interior = np.unique(np.quantile(y_valid, qs)).tolist()
    else:
        raise ValueError(f"未知分箱口径: {scheme!r}")

    edges = tuple([-np.inf] + [float(x) for x in interior] + [np.inf])
    return BinThresholds(edges=edges, scheme=scheme, fitted_on="valid")


# ---------------------------------------------------------------- 单指标实现


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """回归主指标，单位 = rating 点。"""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, tol: int = C.HIT_TOLERANCE) -> float:
    """±tol 命中率：落在真实 rating ±tol 内的比例。"""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) <= tol))


def macro_f1_from_bins(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """类别不平衡下的分档表现（macro 平均，非 weighted）。"""
    return float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))


def qwk_from_bins(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """二次加权 Kappa：教育测量对**有序**等级的标准指标。"""
    return float(cohen_kappa_score(y_true_bin, y_pred_bin, weights="quadratic"))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """相对排序质量——组卷/推题真正需要的指标。"""
    from scipy.stats import spearmanr

    rho, _p = spearmanr(np.asarray(y_true, float), np.asarray(y_pred, float))
    return float(rho)


# ---------------------------------------------------------------- 结果容器


@dataclass
class MetricResult:
    """一个模型在一格（condition × seed）上的全部指标。"""

    values: dict[str, float]
    thresholds: BinThresholds
    n_test: int
    meta: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, k: str) -> float:
        return self.values[k]

    def as_row(self) -> dict[str, float]:
        return dict(self.values)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: BinThresholds,
    meta: Mapping[str, object] | None = None,
) -> MetricResult:
    """由**一次回归输出**派生全部 5 项指标。

    `y_pred` 必须是 rating 空间的回归预测（不是档位）。分箱在函数内部完成，
    阈值由调用方从验证集传入——本函数绝不在测试集上拟合任何东西。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    true_bin = thresholds.apply(y_true)
    pred_bin = thresholds.apply(y_pred)

    values = {
        "mae": mae(y_true, y_pred),
        "hit200": hit_rate(y_true, y_pred),
        "macro_f1": macro_f1_from_bins(true_bin, pred_bin),
        "qwk": qwk_from_bins(true_bin, pred_bin),
        "spearman": spearman(y_true, y_pred),
    }
    # 自检：不得出现大纲已删的指标
    for bad in C.FORBIDDEN_METRICS:
        if bad in values:
            raise AssertionError(f"出现了大纲已删除的指标: {bad}")
    assert set(values) == set(C.METRICS), f"指标集合与 config 不符: {set(values)}"

    return MetricResult(
        values=values, thresholds=thresholds, n_test=len(y_true),
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------- 显著性


@dataclass
class BootstrapResult:
    """配对 bootstrap 结果。"""

    metric: str
    delta: float                 # 模型1 − 模型2 的观测差值
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    alpha: float

    @property
    def significant(self) -> bool:
        """CI 是否不跨 0。"""
        return self.ci_low > 0 or self.ci_high < 0

    def as_row(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "significant": self.significant,
            "n_boot": self.n_boot,
        }

    def to_text(self) -> str:
        star = "*" if self.significant else ""
        return (f"{self.metric}: Δ={self.delta:+.4f} "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] p={self.p_value:.4f}{star}")


def _metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "mae":
        return mae
    if name == "hit200":
        return hit_rate
    if name == "spearman":
        return spearman
    raise ValueError(
        f"{name} 依赖分箱阈值，不适用于逐样本 bootstrap；"
        f"请在调用前把分档结果作为整数 y_true/y_pred 传入，"
        f"或改用 bootstrap_binned_metrics。"
    )


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "mae",
    n_boot: int = C.BOOTSTRAP_N,
    alpha: float = C.BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> BootstrapResult:
    """配对 bootstrap：同一批测试样本上两个模型的指标差。

    配对（不是独立两样本）是这里的关键——两模型跑的是同一测试集，
    差值方差因此远小于各自方差，这也是"差距显著"应当被如此检验的原因。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    n = len(y_true)
    if len(pred_a) != n or len(pred_b) != n:
        raise ValueError("三个数组长度必须一致")

    fn = _metric_fn(metric)
    observed = fn(y_true, pred_a) - fn(y_true, pred_b)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    # 双侧 p 值：bootstrap 分布中越过 0 的比例（加倍取双侧）
    p = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    p = float(min(max(p, 1.0 / (n_boot + 1)), 1.0))

    return BootstrapResult(
        metric=metric, delta=float(observed), ci_low=float(lo), ci_high=float(hi),
        p_value=p, n_boot=n_boot, alpha=alpha,
    )


def bootstrap_binned_metrics(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    thresholds: BinThresholds,
    metric: str = "macro_f1",
    n_boot: int = C.BOOTSTRAP_N,
    alpha: float = C.BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> BootstrapResult:
    """对依赖分箱的指标（macro-F1、QWK）做配对 bootstrap。

    阈值固定为验证集上确定的那一套，bootstrap 重采样只作用于测试样本，
    不重新拟合阈值——否则每次重采样都在测试集上"重新定档"，等于泄漏。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)

    tb = thresholds.apply(y_true)
    ab = thresholds.apply(pred_a)
    bb = thresholds.apply(pred_b)

    if metric == "macro_f1":
        fn = macro_f1_from_bins
    elif metric == "qwk":
        fn = qwk_from_bins
    else:
        raise ValueError(f"未知分箱指标: {metric!r}")

    n = len(y_true)
    observed = fn(tb, ab) - fn(tb, bb)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = fn(tb[idx], ab[idx]) - fn(tb[idx], bb[idx])

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    p = float(min(max(p, 1.0 / (n_boot + 1)), 1.0))
    return BootstrapResult(
        metric=metric, delta=float(observed), ci_low=float(lo), ci_high=float(hi),
        p_value=p, n_boot=n_boot, alpha=alpha,
    )


# ---------------------------------------------------------------- 分段与混淆

def segment_hit_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: Sequence[float] = C.DIFFICULTY_BINS,
    tol: int = C.HIT_TOLERANCE,
) -> list[dict[str, float]]:
    """图 4：分 rating 段的 ±200 命中率（两端误差大的证据）。

    最低段（≤800 下钳位）单独标注——Div4 的 A 题与 Div2 的 B 题可能同为 800
    而实际难易不同，最低档内部噪声最大，构成回归 MAE 的系统性下限。
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    bounds = [-np.inf] + [float(e) for e in edges] + [np.inf]
    out: list[dict[str, float]] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        m = (y_true > lo) & (y_true <= hi)
        if not m.any():
            continue
        out.append({
            "segment": f"({lo:g}, {hi:g}]",
            "n": float(m.sum()),
            "hit_rate": hit_rate(y_true[m], y_pred[m], tol),
            "mae": mae(y_true[m], y_pred[m]),
            "at_rating_floor": float(lo <= C.RATING_FLOOR < hi),
        })
    return out


def confusion_matrix_binned(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: BinThresholds,
) -> np.ndarray:
    """图 4 另一面板：档位混淆矩阵（相邻档互混为主，说明难度是连续量）。"""
    tb = thresholds.apply(np.asarray(y_true, float))
    pb = thresholds.apply(np.asarray(y_pred, float))
    k = thresholds.n_bins
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (tb, pb), 1)
    return cm


def per_tag_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag_matrix: np.ndarray,
    tag_names: Sequence[str],
    tol: int = C.HIT_TOLERANCE,
    min_n: int = 30,
) -> list[dict[str, object]]:
    """按标签（dp/graph/math 等）分组的误差，用于误差分析。"""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    tag_matrix = np.asarray(tag_matrix)
    rows: list[dict[str, object]] = []
    for j, name in enumerate(tag_names):
        m = tag_matrix[:, j].astype(bool)
        if m.sum() < min_n:
            continue
        rows.append({
            "tag": name, "n": int(m.sum()),
            "hit_rate": hit_rate(y_true[m], y_pred[m], tol),
            "mae": mae(y_true[m], y_pred[m]),
        })
    rows.sort(key=lambda r: float(r["mae"]), reverse=True)
    return rows


# ---------------------------------------------------------------- 汇总

def summarize_seeds(results: Sequence[MetricResult]) -> dict[str, dict[str, float]]:
    """把多个随机种子聚合成"均值±标准差"（主表格式）。"""
    if not results:
        raise ValueError("results 为空")
    out: dict[str, dict[str, float]] = {}
    for m in C.METRICS:
        vals = np.array([r[m] for r in results], dtype=np.float64)
        out[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": float(len(vals)),
        }
    return out


def format_mean_std(stats: Mapping[str, Mapping[str, float]], metric: str,
                    digits: int = 3) -> str:
    """主表单元格：如 118.42±3.10。"""
    s = stats[metric]
    return f"{s['mean']:.{digits}f}±{s['std']:.{digits}f}"
