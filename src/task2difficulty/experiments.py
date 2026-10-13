"""实验编排：公平对比协议 + 因子设计（大纲四·（一）（二）（三））。

本模块把大纲的核心实验逻辑变成不可绕过的代码路径
------------------------------------------------
1. **同一划分文件**：所有模型从 `dataset.SplitIndex` 取索引，没有第二个入口；
2. **调参只在 A 格**：`tune_hyperparams` 只在条件 A 上运行一次，结果存入
   `outputs/results/hyperparams_{model}.json`，B/C/D 格直接读该文件复用；
3. **测试集只碰一次**：`_guard_test_access` 会拦住同一 (模型, 条件) 的第二次测试预测；
4. **冷启动格的填补不引入训练集统计量**：见 `prepare_struct_matrix`——
   这是最隐蔽的一处泄漏，用训练集中位数填补会把统计信息偷偷塞回冷启动场景。
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C
from . import dataset as D
from . import metrics as M
from . import models_deep as MD
from . import models_tabular as MT
from . import vectorizers as V

# ================================================================ 特征矩阵构造

#: 冷启动哨兵值。取**常量**（不依赖训练集任何统计量），因此不构成泄漏。
SENTINEL: float = 0.0

#: 需要做量级压缩（log10）的列。
#:
#: ⚠ 为什么必须压缩（实测踩坑）：题面里真实存在超大约束——
#:   - 630-A  "2 ≤ n ≤ 2·10^18"（求 5 的 n 次方）
#:   - 592-C  "1 ≤ t, w, b ≤ 5·10^18"
#:   - 1487-F "1 ≤ n < 10^50"
#: 这些值是**解析正确**的，不是解析 bug。但直接把 1e50 喂进模型会导致：
#:   1. 随机森林在 float32 下溢出并直接报错
#:      （`Input X contains infinity or a value too large for dtype('float32')`；
#:        且只在拼上 TF-IDF 形成稠密矩阵后才触发，仅看结构化特征很难发现）；
#:   2. 岭回归/MLP 被单个超大数值主导，其余特征的信息被压制
#:      （实测 20045 维稠密特征下岭回归 test MAE 624.9，几乎等于常数基线）。
#: 因此对"规模"类特征统一取 log10：10^50 → 50、2·10^18 → 18.3。
#: 语义上也更恰当——建模关心的是**数量级**而非精确值。
#: 只压规模类；比例类（acceptance_rate）与标志类（is_interactive）不动。
LOG_SCALE_COLUMNS: Final[tuple[str, ...]] = (
    "constraint_n_upper",
    "multi_test_count",
    "solved_count",
    "days_since_publish",
    "solved_count__s1",
    "attempts_in_contest__s1",
    "first_solve_seconds__s1",
)


def log_compress(X: pd.DataFrame) -> pd.DataFrame:
    """对规模类特征取 log10，抑制极端量级。

    只处理 `LOG_SCALE_COLUMNS` 中出现的列，其余列原样保留。
    ⚠ NaN 必须保持为 NaN——它表示"该特征不可得"（冷启动格），
    若被 log 变换成某个数值就伪造出了信息，会污染 A−B 与 B−D 的差分。
    log10 是**常量变换**，不依赖训练集统计量，因此不构成泄漏。
    """
    X = X.copy()
    for c in LOG_SCALE_COLUMNS:
        if c not in X.columns:
            continue
        col = X[c].astype(np.float64)
        # log10(x+1)：+1 容纳 0 与 1（否则 log10(1)=0 会与"零值"混淆）
        # where(col >= 0) 把负值（理论上不应出现）转为 NaN，避免 log 报错
        X[c] = np.log10(col.where(col >= 0) + 1.0)
    return X


def prepare_struct_matrix(ds: D.DifficultyDataset, model_name: str) -> pd.DataFrame:
    """按模型构造结构化特征矩阵。

    这是全文最容易自己踩的隐蔽泄漏点，处理方式必须写进正文二·（二）：

    - 树模型（LightGBM/RF）原生吃 NaN，统计特征置空后就是整列 NaN，保持原样；
    - 线性/核方法/神经网络不吃 NaN。**不能**用训练集中位数填补——
      中位数本身由训练集统计特征算出，等于把"训练集里这题的累计通过量"
      偷偷塞回冷启动场景，系统性高估 B/D 格，直接污染 A−B 与 B−D 两个核心差分。
      改用**常量哨兵 0 + 显式缺失指示列**，让模型知道"这一维不存在"。

    另外对规模类特征做 log10 压缩（见 `LOG_SCALE_COLUMNS`），
    否则题面中真实存在的 1e50 级约束会让随机森林溢出、岭回归失效。

    注意：传入的 `ds` 应当已经过 `with_feature_condition()` 处理，
    统计特征在冷启动下已整列移除（而非填 0）。
    """
    X = ds.X
    if model_name == "tag_mean":
        # 朴素基线只看算法标签
        return log_compress(X[ds.group_columns("tag")].astype(np.float64))

    Xf = log_compress(X.astype(np.float64))

    if model_name in C.IMPUTATION_REQUIRED_MODELS:
        nan_cols = [c for c in Xf.columns if Xf[c].isna().any()]
        for c in nan_cols:
            Xf[f"{c}__missing"] = Xf[c].isna().astype(np.float64)
        Xf = Xf.fillna(SENTINEL)
    # 树模型：保留 NaN，不做任何填补

    return Xf


def struct_feature_stats(struct_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """算出结构化特征的标准化参数（均值/标准差），**只用训练集**。

    深度模型必须标准化输入：本任务特征跨度极大（solved_count 可达 1e6，
    多测组数上界 1e4，交互题标志 0/1），不标准化时第一个 batch 就会
    梯度爆炸、loss 变 NaN（实测）。参数只由训练集给出，故不构成泄漏。
    """
    arr = struct_train.to_numpy(np.float64)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-8), 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def assert_no_stat_leak(ds: D.DifficultyDataset, stat_condition: str) -> None:
    """自检：统计特征可得性必须与条件严格一致。

    两条红线，任一被违反都会让泄漏讨论与全部差分作废：
    1. **冷启动条件**（masked）下不得残留任何统计特征列；
    2. **普通条件**（full/masked）下不得残留 S1' 的 `*__s1` 列——
       S1' 只在 `in_contest` 条件才允许启用（由 with_feature_condition 改名）。
       若误把 `__s1` 列当普通特征，A/B/C/D 四格会同时拿到赛时统计，
       而 `__s1` 与真实统计高度共线，表面上只会"性能变好"，极难察觉。
    """
    cols = list(ds.X.columns)
    if stat_condition == C.STAT_MASKED:
        present = [c for c in C.STAT_FEATURES if c in cols]
        if present:
            raise AssertionError(
                f"冷启动条件下统计特征列不应存在，却发现 {present}。"
                f"请检查 with_feature_condition(stat_condition='masked') 是否已调用。"
            )
    if stat_condition != C.STAT_IN_CONTEST:
        s1_left = [c for c in cols if c.endswith("__s1")]
        if s1_left:
            raise AssertionError(
                f"条件 {stat_condition!r} 下不应出现 S1' 列，却发现 {s1_left}。"
                f"S1' 赛时统计只允许在 stat_condition='in_contest' 时启用；"
                f"否则 A/B/C/D 四格都会悄悄拿到赛时统计，泄漏讨论全部作废。"
            )


# ================================================================ 数据结构


@dataclass
class DataPackage:
    """某个条件下的全部输入材料。"""

    cond_key: str
    split: str
    stat_condition: str
    ds_train: D.DifficultyDataset
    ds_valid: D.DifficultyDataset
    ds_test: D.DifficultyDataset
    tfidf_train: Any = None
    tfidf_valid: Any = None
    tfidf_test: Any = None
    vocab_size: int = 0
    text_mode: str = "clean"
    #: 结构化矩阵缓存（按模型名），避免重复构造
    _struct_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = \
        field(default_factory=dict, repr=False)
    #: 文本降维缓存（随机化 SVD 只在训练集上拟合一次）
    _text_svd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = \
        field(default_factory=dict, repr=False)
    #: 随机化 SVD 的元信息（奇异值谱、解释方差比），写进实验元数据
    svd_info: dict[str, Any] = field(default_factory=dict, repr=False)

    def struct(self, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if model_name not in self._struct_cache:
            self._struct_cache[model_name] = (
                prepare_struct_matrix(self.ds_train, model_name),
                prepare_struct_matrix(self.ds_valid, model_name),
                prepare_struct_matrix(self.ds_test, model_name),
            )
        return self._struct_cache[model_name]

    def text_svd(self, k: int | None = None, n_iter: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """文本通道的随机化截断 SVD 表示（GPU 上拟合与投影）。

        ⚠ 投影方向只在**训练集**上拟合，验证/测试集只用 `transform`。
        这决定了它是"训练时确定的常量变换"，可以写进特征口径而不算泄漏。
        """
        from . import accel as A

        k = int(C.TEXT_SVD_DIMS if k is None else k)
        n_iter = int(C.TEXT_SVD_N_ITER if n_iter is None else n_iter)
        key = f"k{k}_i{n_iter}"
        if key not in self._text_svd_cache:
            if self.tfidf_train is None:
                raise RuntimeError("未构造 TF-IDF，无法做文本降维（need_tfidf=True）")
            svd = A.RandomizedSvd(k=k, n_iter=n_iter, seed=C.SEEDS[0])
            tr = svd.fit_transform(self.tfidf_train)
            va = svd.transform(self.tfidf_valid)
            te = svd.transform(self.tfidf_test)
            self._text_svd_cache[key] = (tr, va, te)
            s = svd.singular_values_
            ev = s ** 2
            self.svd_info = {
                "svd_k": int(len(s)), "svd_n_iter": n_iter,
                "svd_explained_ratio": float(ev.sum() / max(self.tfidf_train.power(2).sum(), 1e-12)),
                "svd_device": A.device_label(),
            }
        return self._text_svd_cache[key]

    def text_blocks(self, model_name: str):
        """按模型取文本通道：None / 稀疏 TF-IDF / 降维后的稠密数组。

        只有 `config.TEXT_SVD_MODELS` 里的模型走降维——其余模型用全维稀疏表示，
        既保留词袋的可解释性，也不需要额外的近似。
        """
        if self.tfidf_train is None or model_name == "tag_mean":
            return None, None, None
        if model_name in C.TEXT_SVD_MODELS:
            tr, va, te = self.text_svd()
            return tr, va, te
        return self.tfidf_train, self.tfidf_valid, self.tfidf_test

    @property
    def n_struct_features(self) -> int:
        x, _, _ = self.struct("lightgbm")
        return int(x.shape[1])

    def sizes(self) -> dict[str, int]:
        return {
            "n_train": len(self.ds_train),
            "n_valid": len(self.ds_valid),
            "n_test": len(self.ds_test),
        }


#: 条件数据包缓存：TF-IDF 单次拟合约 6 秒，而阶段循环是"按种子重建数据包"，
#: 5 种子 × 4 条件 × 6 模型会白白重复拟合上百次（≈12 分钟纯浪费）。
#: 数据包构造只依赖 (条件, 文本模式, 词表上限, 特征组取舍)，与种子无关，
#: 因此用这个键缓存；LRU 上限避免把 4 个条件的稀疏矩阵全留在内存里。
_PKG_CACHE: "OrderedDict[tuple, DataPackage]" = OrderedDict()
_PKG_CACHE_MAX = 6


def clear_package_cache() -> None:
    _PKG_CACHE.clear()


def build_data_package(
    ds: D.DifficultyDataset,
    si: D.SplitIndex,
    cond: C.Condition,
    *,
    drop_provenance: bool = True,
    drop_groups: Iterable[str] = (),
    keep_groups: Iterable[str] | None = None,
    need_tfidf: bool = False,
    tfidf_max_features: int = 20000,
    text_mode: str = "clean",
) -> DataPackage:
    """按条件切分数据、应用特征可得性、**只在训练集上**拟合文本向量化。

    text_mode:
        - 'clean'：常规清洗后题面；
        - 'stripped'：额外删除模板句（大纲四·（四）文本端 provenance 剥离）；
        - 'with_samples'：保留样例段（敏感性对照，可选）。
    """
    ck = (cond.key, cond.split, cond.stat, text_mode, int(tfidf_max_features),
          bool(need_tfidf), bool(drop_provenance),
          tuple(sorted(drop_groups)), tuple(keep_groups) if keep_groups else None)
    hit = _PKG_CACHE.get(ck)
    if hit is not None:
        _PKG_CACHE.move_to_end(ck)
        return hit

    sub = ds.with_feature_condition(
        cond.stat, drop_provenance=drop_provenance,
        drop_groups=drop_groups, keep_groups=keep_groups,
    )
    tr = sub.select(si.indices(cond.split, "train"))
    va = sub.select(si.indices(cond.split, "valid"))
    te = sub.select(si.indices(cond.split, "test"))
    for part in (tr, va, te):
        assert_no_stat_leak(part, cond.stat)

    if text_mode == "stripped":
        for part in (tr, va, te):
            part.text = part.text.map(V.strip_template_phrases)

    pkg = DataPackage(
        cond_key=cond.key, split=cond.split, stat_condition=cond.stat,
        ds_train=tr, ds_valid=va, ds_test=te, text_mode=text_mode,
    )
    if need_tfidf:
        vec = V.TfidfTextVectorizer(max_features=tfidf_max_features)
        pkg.tfidf_train = vec.fit_transform(tr.text)   # fit 只用训练集
        pkg.tfidf_valid = vec.transform(va.text)
        pkg.tfidf_test = vec.transform(te.text)
        pkg.vocab_size = vec.vocabulary_size
    _PKG_CACHE[ck] = pkg
    while len(_PKG_CACHE) > _PKG_CACHE_MAX:
        _PKG_CACHE.popitem(last=False)
    return pkg


# ================================================================ 结果与守卫


@dataclass
class RunResult:
    """一个（模型 × 条件 × 种子）的完整产出。"""

    model: str
    condition: str
    seed: int
    valid_pred: np.ndarray
    test_pred: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    valid_metrics: M.MetricResult
    test_metrics: M.MetricResult
    thresholds_width: M.BinThresholds
    thresholds_freq: M.BinThresholds
    train_seconds: float = 0.0
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def meta_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model": self.model, "condition": self.condition, "seed": self.seed,
            "n_train": self.n_train, "n_valid": self.n_valid, "n_test": self.n_test,
            "train_seconds": round(self.train_seconds, 3),
        }
        row.update({f"valid_{k}": v for k, v in self.valid_metrics.values.items()})
        row.update({f"test_{k}": v for k, v in self.test_metrics.values.items()})
        row.update(self.extra)
        return row


#: 测试集访问记录：{(model, condition): 次数}
_TEST_ACCESS: dict[tuple[str, str], int] = {}


def _guard_test_access(model: str, condition: str, allow_repeat: bool) -> None:
    """大纲红线：测试集只碰一次。"""
    key = (model, condition)
    n = _TEST_ACCESS.get(key, 0)
    if n > 0 and not allow_repeat:
        raise RuntimeError(
            f"测试集守卫：(模型={model}, 条件={condition}) 已被预测 {n} 次。"
            f"大纲要求测试集只碰一次；多种子重复实验请显式传 allow_repeat=True。"
        )
    _TEST_ACCESS[key] = n + 1


def reset_test_guard() -> None:
    _TEST_ACCESS.clear()


def test_access_log() -> dict[str, int]:
    return {f"{m}|{c}": n for (m, c), n in _TEST_ACCESS.items()}


# ---------------------------------------------------------------- 预测缓存

#: 预测缓存目录：配对 bootstrap 需要**逐样本预测值**，CSV 里只有聚合指标。
#: 没有它就只能用聚合指标近似，无法做真正的"配对"检验（同一批测试样本上的差值）。
PRED_DIR = C.RESULTS_DIR / "preds"


def save_predictions(res: "RunResult", suffix: str = "",
                     problem_key: Sequence[str] | None = None,
                     split: str | None = None) -> Path:
    """把逐样本预测与真值存成 npz，供配对 bootstrap 与误差分析使用。

    ⚠ **必须一并写入题目标识**：只存预测数组时，消费方只能靠
    `dataset.csv` + `split_index.csv` 的**行序**自行重建测试集索引——
    这两个文件任一变化（重建数据集、改抽样、换快照）都会让索引**静默错位**，
    而错位后的配对 bootstrap 与误差分析看不出任何异常（实测踩过）。
    因此这里把 `test_problem_key` / `valid_problem_key` 与 split/condition 一起落盘，
    消费方应当核对（或直接使用）这些键，而不是行序。
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{res.condition}_{res.model}_seed{res.seed}{suffix}"
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    path = PRED_DIR / f"{tag}.npz"

    payload: dict[str, np.ndarray] = {
        "valid_pred": res.valid_pred, "test_pred": res.test_pred,
        "y_valid": res.y_valid, "y_test": res.y_test,
        "width_edges": np.asarray(res.thresholds_width.edges, dtype=np.float64),
        "freq_edges": np.asarray(res.thresholds_freq.edges, dtype=np.float64),
        # 标量元信息（npz 里存成 0 维数组）
        "model": np.asarray(res.model),
        "condition": np.asarray(res.condition),
        "seed": np.asarray(res.seed),
        "split": np.asarray(split if split is not None else ""),
    }
    if problem_key is not None:
        pk = np.asarray(list(problem_key), dtype=np.str_)
        if len(pk) != len(res.test_pred):
            raise ValueError(
                f"problem_key 长度 {len(pk)} 与测试集预测数 {len(res.test_pred)} 不一致；"
                f"请确认传入的是**测试集**的题目键（按测试集顺序）。"
            )
        payload["test_problem_key"] = pk
    np.savez_compressed(path, **payload)
    return path


def load_predictions(condition: str, model: str, seed: int = 0,
                     suffix: str = "") -> dict[str, np.ndarray] | None:
    tag = f"{condition}_{model}_seed{seed}{suffix}"
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    path = PRED_DIR / f"{tag}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_keys_of(condition: str, model: str, seed: int = 0,
                 suffix: str = "") -> np.ndarray | None:
    """取出某次运行缓存的**测试集题目键**；旧缓存没有该字段时返回 None。

    消费方应优先用它与自己的数据集核对题目顺序，而不是假设行序一致。
    """
    d = load_predictions(condition, model, seed, suffix)
    if d is None or "test_problem_key" not in d:
        return None
    return d["test_problem_key"]


def _evaluate_pair(y_valid, valid_pred, y_test, test_pred, extra_meta=None):
    """统一的分箱+评估：**阈值只在验证集上确定**，等宽与等频两套都算。"""
    thr_w = M.fit_bin_thresholds(y_valid, "equal_width")
    thr_f = M.fit_bin_thresholds(y_valid, "equal_freq")
    vm = M.evaluate(y_valid, valid_pred, thr_w, extra_meta)
    tm = M.evaluate(y_test, test_pred, thr_w, extra_meta)
    # 等频口径的敏感性（大纲二·（三）：两种口径都报）
    tm_freq = M.evaluate(y_test, test_pred, thr_f, {"scheme": "equal_freq"})
    tm.meta["macro_f1_equal_freq"] = tm_freq["macro_f1"]
    tm.meta["qwk_equal_freq"] = tm_freq["qwk"]
    return vm, tm, thr_w, thr_f


# ================================================================ 传统模型


def run_tabular(
    model_name: str,
    pkg: DataPackage,
    params: Mapping[str, Any] | None = None,
    tag_columns: Sequence[str] | None = None,
    seed: int = 0,
    guard_test: bool = True,
    allow_test_repeat: bool = False,
) -> RunResult:
    """跑一个传统侧模型：文本走 TF-IDF，结构化走 prepare_struct_matrix。

    ⚠ 特征矩阵保持**稀疏**（不转稠密）：TF-IDF 非零占比仅约 1.2%，
    稠密化会让 20045 维矩阵占 1.3 GB，岭回归单次训练从秒级恶化到 166 秒。
    因此这里把矩阵原样传给模型层，由各模型自行决定如何处理。
    """
    t0 = time.time()
    Xtr_s, Xva_s, Xte_s = pkg.struct(model_name)
    Ttr, Tva, Tte = pkg.text_blocks(model_name)

    def stack(struct_df, text):
        if text is None:
            return struct_df.to_numpy(np.float64)
        if isinstance(text, np.ndarray):
            # 降维后的稠密文本（SVR 路径）：直接横向拼，维度可控（~200 维）
            return np.hstack([struct_df.to_numpy(np.float64), text])
        return MT.hstack_features(struct_df, text)

    Xtr = stack(Xtr_s, Ttr)
    Xva = stack(Xva_s, Tva)
    Xte = stack(Xte_s, Tte)

    m = MT.make_tabular_model(model_name, tag_columns=tag_columns, params=params)
    # ⚠ 种子只注入到**确有随机性**的模型（随机森林 / LightGBM）。
    # 这里用类属性 `uses_seed` 判定，而不是判断 `"random_state" in m.params`：
    # 后者一旦因为参数合并方式变化而失效，就会静默变成"多种子跑出完全相同的结果"，
    # 标准差恒为 0 却看不出任何异常（实测踩过）。
    if seed and getattr(m, "uses_seed", False):
        m.params["random_state"] = int(seed)
    m.fit(Xtr, pkg.ds_train.y, Xva, pkg.ds_valid.y)

    valid_pred = m.predict(Xva)
    if guard_test:
        _guard_test_access(model_name, pkg.cond_key, allow_test_repeat)
    test_pred = m.predict(Xte)

    vm, tm, thr_w, thr_f = _evaluate_pair(
        pkg.ds_valid.y, valid_pred, pkg.ds_test.y, test_pred,
        {"model": model_name, "condition": pkg.cond_key},
    )
    return RunResult(
        model=model_name, condition=pkg.cond_key, seed=seed,
        valid_pred=valid_pred, test_pred=test_pred,
        y_valid=pkg.ds_valid.y, y_test=pkg.ds_test.y,
        valid_metrics=vm, test_metrics=tm,
        thresholds_width=thr_w, thresholds_freq=thr_f,
        train_seconds=time.time() - t0,
        n_train=len(pkg.ds_train), n_valid=len(pkg.ds_valid), n_test=len(pkg.ds_test),
        extra={
            "n_features": int(Xtr.shape[1]),
            "n_struct_features": int(Xtr_s.shape[1]),
            "text_repr": ("svd" + str(Ttr.shape[1])) if isinstance(Ttr, np.ndarray)
                         else ("tfidf" if Ttr is not None else "none"),
            "text_mode": pkg.text_mode,
            "tfidf_vocab": int(pkg.vocab_size),
            "test_macro_f1_equal_freq": tm.meta.get("macro_f1_equal_freq"),
            "test_qwk_equal_freq": tm.meta.get("qwk_equal_freq"),
            "device": getattr(m, "device_", "cpu"),
        },
    )


# ================================================================ 深度模型


def _make_batches(pkg: DataPackage, model_name: str, part: str,
                  vec: Any, max_length: int, struct_key: str) -> MD.Batch:
    """按模型构造某一子集（train/valid/test）的输入批次容器。

    返回**单个 Batch**（全量数组），由 `make_loader` 负责按 batch_size 切分——
    不要在这里按样本拆成 N 个 Batch，那只是把 DataLoader 的活重做一遍。
    """
    ds = {"train": pkg.ds_train, "valid": pkg.ds_valid, "test": pkg.ds_test}[part]
    y = np.asarray(ds.y, dtype=np.float64)

    token_ids = attention_mask = token_type_ids = None
    import torch  # noqa: F401  (类型提示用)

    if model_name in ("textcnn", "bilstm_attn"):
        token_ids = vec.encode(ds.text)
        attention_mask = (token_ids != 0).astype(np.int64)
    elif model_name in ("bert", "fusion"):
        enc = vec.encode(ds.text)
        token_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        token_type_ids = enc.get("token_type_ids")

    struct = None
    if model_name in ("mlp", "fusion"):
        s_tr, s_va, s_te = pkg.struct(struct_key)
        struct = {"train": s_tr, "valid": s_va, "test": s_te}[part].to_numpy(np.float64)

    return MD.Batch(
        struct=struct, token_ids=token_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids, y=y,
    )


def _build_deep_model(model_name: str, d_struct: int, vocab_size: int,
                      cfg: MD.TrainConfig, seed: int, bert_name: str = "bert-base-uncased",
                      in_mean: np.ndarray | None = None, in_std: np.ndarray | None = None):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_name == "mlp":
        return MD.build_mlp(d_struct, hidden=(256, 128), dropout=cfg.dropout,
                            in_mean=in_mean, in_std=in_std)
    if model_name == "textcnn":
        return MD.build_textcnn(vocab_size, embed_dim=cfg.embed_dim,
                                dropout=cfg.dropout)
    if model_name == "bilstm_attn":
        return MD.build_bilstm_attn(vocab_size, embed_dim=cfg.embed_dim,
                                    dropout=cfg.dropout)
    if model_name == "bert":
        return MD.build_bert_regressor(bert_name, dropout=min(0.3, cfg.dropout))
    if model_name == "fusion":
        return MD.build_fusion(bert_name, d_struct=d_struct, dropout=cfg.dropout,
                               in_mean=in_mean, in_std=in_std)
    raise ValueError(f"未知深度模型: {model_name!r}")


def run_deep(
    model_name: str,
    pkg: DataPackage,
    cfg: MD.TrainConfig,
    seed: int = 0,
    device: Any = None,
    bert_name: str = "bert-base-uncased",
    text_vocab: V.SimpleVocab | None = None,
    bert_tokenizer: V.BertTokenizerWrapper | None = None,
    log: Any = None,
    guard_test: bool = True,
    allow_test_repeat: bool = False,
) -> RunResult:
    """跑一个深度模型。

    早停/选模只看验证集 MAE；测试集在 `train_torch_model` 返回后**预测一次**。
    """
    t0 = time.time()
    if device is None:
        device = MD.pick_device()

    use_bert = model_name in ("bert", "fusion")
    vec = bert_tokenizer if use_bert else text_vocab
    if model_name != "mlp" and vec is None:
        raise ValueError(f"{model_name} 需要文本向量器")

    max_len = cfg.max_length if use_bert else cfg.seq_length
    # 融合模型的结构化通道保留 NaN（网络内 nan_to_num 处理），
    # 因此取 "lightgbm" 键（不插缺失指示列）；MLP 必须填补，取 "mlp" 键。
    struct_key = "lightgbm" if model_name == "fusion" else "mlp"
    d_struct = 0
    if model_name in ("mlp", "fusion"):
        d_struct = int(pkg.struct(struct_key)[0].shape[1])

    vocab_size = 0
    if model_name in ("textcnn", "bilstm_attn"):
        vocab_size = int(text_vocab.vocab_size)

    tr_batches = _make_batches(pkg, model_name, "train", vec, max_len, struct_key)
    va_batches = _make_batches(pkg, model_name, "valid", vec, max_len, struct_key)
    te_batches = _make_batches(pkg, model_name, "test", vec, max_len, struct_key)

    tr_loader = MD.make_loader(tr_batches, cfg.batch_size, True, seed=seed)
    va_loader = MD.make_loader(va_batches, cfg.batch_size, False, seed=seed)
    te_loader = MD.make_loader(te_batches, cfg.batch_size, False, seed=seed)

    # 标准化参数只用训练集算（深度模型不做标准化会梯度爆炸、loss 变 NaN）
    in_mean = in_std = None
    if model_name in ("mlp", "fusion"):
        in_mean, in_std = struct_feature_stats(pkg.struct(struct_key)[0])

    # 目标标准化：只用训练集标签算，预测时逆变换回 rating 空间
    target_scaler = MD.TargetScaler.fit(pkg.ds_train.y)

    model = _build_deep_model(model_name, d_struct, vocab_size, cfg, seed, bert_name,
                              in_mean=in_mean, in_std=in_std)
    model, history, valid_pred = MD.train_torch_model(
        model, tr_loader, va_loader, cfg, device, log=log,
        scaler_target=target_scaler,
    )

    if guard_test:
        _guard_test_access(model_name, pkg.cond_key, allow_test_repeat)
    test_pred = MD.predict_torch_model(model, te_loader, device,
                                       scaler_target=target_scaler)

    vm, tm, thr_w, thr_f = _evaluate_pair(
        pkg.ds_valid.y, valid_pred, pkg.ds_test.y, test_pred,
        {"model": model_name, "condition": pkg.cond_key},
    )
    return RunResult(
        model=model_name, condition=pkg.cond_key, seed=seed,
        valid_pred=valid_pred, test_pred=test_pred,
        y_valid=pkg.ds_valid.y, y_test=pkg.ds_test.y,
        valid_metrics=vm, test_metrics=tm,
        thresholds_width=thr_w, thresholds_freq=thr_f,
        train_seconds=time.time() - t0,
        n_train=len(pkg.ds_train), n_valid=len(pkg.ds_valid), n_test=len(pkg.ds_test),
        extra={
            "n_struct_features": int(d_struct),
            "text_mode": pkg.text_mode,
            "epochs_run": len(history.epoch),
            "best_epoch": history.best_epoch,
            "best_valid_mae": history.best_valid_mae,
            "stopped_early": history.stopped_early,
            "peak_vram_gb": history.peak_vram_gb,
            "device": str(device),
            "lr": cfg.lr, "batch_size": cfg.batch_size, "max_length": max_len,
            "test_macro_f1_equal_freq": tm.meta.get("macro_f1_equal_freq"),
            "test_qwk_equal_freq": tm.meta.get("qwk_equal_freq"),
        },
    )


# ================================================================ 调参与复用


def hyperparams_path(model_name: str, out_dir: Path = C.RESULTS_DIR) -> Path:
    return Path(out_dir) / f"hyperparams_{model_name}.json"


def save_hyperparams(model_name: str, params: Mapping[str, Any],
                     tuned_on: str = "A", extra: Mapping[str, Any] | None = None,
                     out_dir: Path = C.RESULTS_DIR) -> Path:
    """把 A 格调出的最优超参写盘，供 B/C/D 格复用。"""
    p = hyperparams_path(model_name, out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "tuned_on_condition": tuned_on,
        "params": dict(params),
        "extra": dict(extra or {}),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_hyperparams(model_name: str, out_dir: Path = C.RESULTS_DIR) -> dict[str, Any]:
    """读取 A 格调好的超参。找不到就报错——不允许各格自行调参。"""
    p = hyperparams_path(model_name, out_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"未找到 {model_name} 的超参文件 {p}。"
            f"大纲要求超参只在 A 格调一次再复用；请先运行 A 格的调参阶段。"
        )
    return json.loads(p.read_text(encoding="utf-8"))["params"]


def sample_params(space: Mapping[str, Sequence[Any]], n: int,
                  seed: int = 0) -> list[dict[str, Any]]:
    """从搜索空间随机采样 n 组（不重复）。"""
    keys = list(space.keys())
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    max_tries = n * 50
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        cfg = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        sig = tuple(sorted((k, str(v)) for k, v in cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def budget_for(model_name: str) -> int:
    fam = C.MODEL_BUDGET_FAMILY[model_name]
    return C.HYPERPARAM_BUDGET[fam]


def search_space_for(model_name: str) -> dict[str, Sequence[Any]]:
    """返回模型的超参搜索空间。

    零超参模型（朴素基线 tag_mean）返回空字典——调参阶段会直接记录默认参数，
    这不是错误：该基线本来就没有可调的东西，其价值正在于"零成本参照"。
    """
    if model_name == "tag_mean":
        return {}
    if model_name in MT.TABULAR_SEARCH_SPACE:
        return dict(MT.TABULAR_SEARCH_SPACE[model_name])
    if model_name in MD.DEEP_SEARCH_SPACE:
        return dict(MD.DEEP_SEARCH_SPACE[model_name])
    raise ValueError(f"{model_name} 没有定义搜索空间")
