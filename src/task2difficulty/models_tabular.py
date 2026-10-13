"""传统机器学习侧模型（大纲三·（一）+ 三·（四））。

主表传统侧 6 项：岭回归 / SVR / 随机森林 / LightGBM / 有序回归 / 标签均值朴素基线。

两个容易被忽略但必须做对的地方
------------------------------
1. **统计特征置空后必须显式填补**：树模型（LightGBM/RF）原生处理 NaN，
   但岭回归/SVR/有序回归不行。填补统计量**只在训练集上算**，再套用到验证/测试。
2. **有序回归输出连续值**：大纲红线要求"分类指标由回归输出分箱派生，
   不得单独训练分类器"。因此 CLM 输出的是 5 档概率的**期望 rating**，
   而不是档位标签——这样它才能和其他模型一样进入统一的分箱流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C


# ================================================================ 工具


def as_dense(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """把可能稀疏的特征矩阵转成 float64 稠密数组。

    注意：必须先 astype(float) 再取 nan——整型列不允许 NaN，
    直接 to_numpy(dtype=float, na_value=nan) 在 pandas 2.x 下会报错。
    """
    if isinstance(X, pd.DataFrame):
        return X.astype(np.float64).to_numpy(dtype=np.float64, na_value=np.nan)
    from scipy import sparse

    if sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def hstack_features(*blocks: Any) -> Any:
    """水平拼接多组特征（结构化 + TF-IDF）。

    ★ 返回**稀疏矩阵**，不转稠密。
    实测教训：TF-IDF 的非零占比仅 1.22%，但早期实现把所有块拼成稠密数组，
    结果 (8864, 20045) 的矩阵占 1.32 GB —— 岭回归单次训练要 **166 秒**，
    且 test MAE 高达 624.9（几乎等于常数基线），说明数据中的极值把解带偏了。
    保持稀疏后，内存与耗时都降到可接受范围，且线性模型的解也恢复正常。
    """
    from scipy import sparse

    mats: list[Any] = []
    for b in blocks:
        if b is None:
            continue
        if isinstance(b, pd.DataFrame):
            mats.append(sparse.csr_matrix(as_dense(b)))
        elif sparse.issparse(b):
            mats.append(b.tocsr())
        else:
            mats.append(sparse.csr_matrix(np.asarray(b, dtype=np.float64)))
    if not mats:
        raise ValueError("没有可拼接的特征块")
    if len(mats) == 1:
        return mats[0]
    return sparse.hstack(mats, format="csr")


# ================================================================ 基类


class BaseModel:
    """统一接口：fit / predict / get_params / set_params。

    `predict` 必须返回 **rating 空间的连续值**——5 项指标全部由它派生。
    """

    name: str = "base"
    needs_imputation: bool = True   # 是否必须填补 NaN
    handles_nan: bool = False       # 原生处理 NaN（树模型）
    #: 该模型实际执行计算的设备标签，写进实验元数据（复现性）
    device_: str = "cpu"
    #: 是否有随机性（决定多种子实验要不要给它换 `random_state`）
    uses_seed: bool = False

    def fit(self, X, y, X_valid=None, y_valid=None) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return dict(getattr(self, "params", {}))

    def set_params(self, params: Mapping[str, Any]) -> "BaseModel":
        """**合并式**更新超参，而不是整体替换。

        ⚠ 这里踩过一个会同时毁掉速度与结论的坑（实测）：
        原先写成 `self.params = dict(params)`，而超参搜索给出的候选只含
        搜索空间里的键（如 `n_estimators` / `max_features`），
        于是默认参数里的 `n_jobs=-1` 与 `random_state` 被整个丢掉——
          - 随机森林退回单线程：单次训练由 78 秒涨到 900 秒以上，
            40 组随机搜索要跑四个多小时；
          - `random_state` 丢失后，`run_tabular` 里"按种子改写 random_state"
            的代码静默失效，多种子实验得到**完全相同**的结果（标准差恒为 0），
            而表面上一切正常——这正是最危险的一类错误。
        改为合并后，未在搜索空间里出现的键一律保留默认值。
        """
        merged = dict(getattr(self, "params", {}) or {})
        merged.update(dict(params or {}))
        self.params = merged
        return self

    # 便捷：训练后立刻在验证集上预测
    def fit_predict(self, X, y, X_valid, y_valid=None) -> np.ndarray:
        self.fit(X, y, X_valid, y_valid)
        return self.predict(X_valid)


# ================================================================ 岭回归


def keep_sparse(X: Any) -> Any:
    """保持稀疏格式传入模型（仅在模型原生支持稀疏时使用）。

    哪些模型支持稀疏输入：
      ✅ Ridge / SVR / RandomForest / LightGBM —— sklearn/lightgbm 均支持 CSR，
         且稀疏更快更省内存；SVR 另需文本先降维（见 config.TEXT_SVD_MODELS）
      ✅ OrdinalLogitModel                 —— 走 torch 稀疏-稠密乘，同样不稠密化
    因此各模型自行声明，而不是在编排层统一转换（统一转稠密会毁掉线性模型的性能）。
    """
    return X


def to_dense_if_needed(X: Any) -> np.ndarray:
    """转成 numpy 稠密数组（供不支持稀疏的自实现模型使用）。"""
    from scipy import sparse

    if sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    if isinstance(X, pd.DataFrame):
        return as_dense(X)
    return np.asarray(X, dtype=np.float64)


@dataclass
class RidgeModel(BaseModel):
    """岭回归：线性基线。用 TF-IDF 高维稀疏文本时也靠它给出"线性下界"。

    ⚠ 必须**保持稀疏**输入：TF-IDF 非零占比仅约 1.2%，稠密化会让
    (8864, 20045) 的矩阵占 1.3 GB，单次训练从秒级恶化到 166 秒，
    且 test MAE 从可用水平退化到 624.9（≈常数基线）。
    """

    name: str = "ridge"
    needs_imputation: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})

    def fit(self, X, y, X_valid=None, y_valid=None):
        from sklearn.linear_model import Ridge

        self._m = Ridge(**self.params)
        # 不转稠密：sklearn 的线性模型原生支持 CSR
        self._m.fit(keep_sparse(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._m.predict(keep_sparse(X)), dtype=np.float64)


# ================================================================ SVR


@dataclass
class SVRModel(BaseModel):
    """核 SVR：核方法（非线性但非集成）的对照角色，保留在主表。

    大纲红线：SVR **保留在主表**，不得降为附表。其不可替代性正在于
    "非线性但不属于集成/深度"这一格，缺了它，LightGBM 的优势就无从定位。

    ⚠ 实现口径（实测被迫调整，正文须如实说明）
    ------------------------------------------
    原计划直接 `sklearn.svm.SVR(kernel='rbf')`。但在 (8864, 20045) 的
    高维稀疏表示上，libsvm 要先构造全核矩阵再解 n×n 的 QP，
    **实测 2.3 小时未收敛**——这不是调参能解决的问题，而是样本量与
    特征维度的组合本身就超出核 SVR 的可行域。

    处理办法是核方法的标准可扩展化路径（Nyström 低秩近似）：
      1. 文本通道先在**训练集**上做随机化截断 SVD 降到 200 维
         （高维稀疏下 RBF 核的距离集中效应会让全维核矩阵本身失去分辨力）；
      2. 结构化特征与降维文本拼接后按训练集均值方差标准化；
      3. RBF 核用 Nyström 展开成 800 维显式特征（地标点取自训练集），
         于是核 SVR 退化为这 800 维上的**线性** SVR。

    第 3 步之后模型仍是"RBF 核 + ε 不敏感损失"这一族，只是核矩阵被
    低秩近似；求解成本从 O(n²)～O(n³) 降到 O(nm²)，实测由"跑不完"
    变成单次数秒。SVD 投影方向与地标点都只来自训练集，不构成泄漏。
    """

    name: str = "svr"
    needs_imputation: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {
        "C": 10.0, "epsilon": 10.0, "gamma": "scale",
        "n_components": C.NYSTROEM_COMPONENTS,
    })

    #: 超过这个特征维度就不再接受输入——防止有人把未降维的稀疏矩阵直接喂进来，
    #: 那会在 Nyström 内部被稠密化成 1.4 GB 的数组（静默吃掉内存）。
    MAX_INPUT_DIM: int = 4000

    def fit(self, X, y, X_valid=None, y_valid=None):
        from scipy import sparse
        from sklearn.svm import LinearSVR

        from . import accel as A

        n, d = X.shape
        if sparse.issparse(X) and d > self.MAX_INPUT_DIM:
            raise ValueError(
                f"SVR 收到 {n}×{d} 的稀疏输入。核 SVR 在高维稀疏文本上不可解，"
                f"必须先走 TEXT_SVD_MODELS 的文本降维路径（见 config.TEXT_SVD_DIMS）。"
            )
        y = np.asarray(y, dtype=np.float64)

        # ---- 标准化（只统计训练集）----
        # RBF 核只看欧氏距离，特征尺度不齐会让"数值大的维"独占距离，
        # 结构化特征与 SVD 分量量级差一到两个数量级，不标准化核就废了。
        Xd = X if isinstance(X, np.ndarray) else to_dense_if_needed(X)
        self._mu = np.nanmean(Xd, axis=0)
        self._sigma = np.nanstd(Xd, axis=0)
        self._sigma[~np.isfinite(self._sigma) | (self._sigma < 1e-12)] = 1.0
        self._mu[~np.isfinite(self._mu)] = 0.0
        Ztr = (Xd - self._mu) / self._sigma

        # ---- Nyström RBF 展开（GPU）----
        m = int(self.params.get("n_components", C.NYSTROEM_COMPONENTS))
        self._map = A.NystroemRbf(n_components=m, gamma=self.params.get("gamma", "scale"),
                                  seed=int(self.params.get("seed", 42)))
        Ftr = self._map.fit_transform(Ztr)

        # ---- 线性 SVR（凸二次规划，m 维）----
        # max_iter 放大到 20000：liblinear 在部分 (C, epsilon) 组合上会报
        # "failed to converge"，但它返回的是当前最优解，放大迭代上限能让这些
        # 组合真正收敛，避免把"没解完"当成"这个超参不好"。
        self._m = LinearSVR(
            C=float(self.params.get("C", 10.0)),
            epsilon=float(self.params.get("epsilon", 10.0)),
            tol=1e-4, max_iter=20000, dual="auto",
            random_state=int(self.params.get("seed", 42)),
        )
        self._m.fit(Ftr, y)
        self.n_features_in_ = d
        self.nystroem_features_ = int(Ftr.shape[1])
        self.gamma_ = float(self._map.gamma_)
        self.device_ = A.device_label()
        return self

    def _features(self, X) -> np.ndarray:
        Xd = X if isinstance(X, np.ndarray) else to_dense_if_needed(X)
        Z = (Xd - self._mu) / self._sigma
        return self._map.transform(Z)

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._m.predict(self._features(X)), dtype=np.float64)


# ================================================================ 随机森林


@dataclass
class RandomForestModel(BaseModel):
    """随机森林：bagging 集成代表。

    ⚠ 两个必须保持的约束（都是实测踩出来的）：

    1. **输入必须稀疏**：早期实现把特征转成 float64 稠密数组，在混入 TF-IDF 后
       （20045 维）会因极值溢出而报
       `Input X contains infinity or a value too large for dtype('float32')`
       —— 这个错误只在"稠密 + 拼接 TF-IDF"时才出现，很难从结构化特征本身察觉。
    2. **缺失值走哨兵填补**：sklearn 的随机森林只在**稠密**输入下支持 NaN，
       稀疏输入遇到 NaN 直接报错。而特征矩阵必须稀疏，所以随机森林与
       线性/核方法一样使用"哨兵 0 + 缺失指示列"。这一点写进正文二·（二）。
    """

    name: str = "random_forest"
    handles_nan: bool = False
    needs_imputation: bool = True
    uses_seed: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2,
        "max_features": 0.5, "n_jobs": -1, "random_state": 42,
    })

    def fit(self, X, y, X_valid=None, y_valid=None):
        from sklearn.ensemble import RandomForestRegressor

        self._m = RandomForestRegressor(**self.params)
        self._m.fit(keep_sparse(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._m.predict(keep_sparse(X)), dtype=np.float64)


# ================================================================ LightGBM


@dataclass
class LightGBMModel(BaseModel):
    """LightGBM：主结论中"存量题库"场景的占优方法。

    原生处理 NaN 是它能直接吃"统计特征置空"的关键——置空列以 NaN 存在，
    模型会自行学到"该特征不可用"的分裂方向，无需我们伪造填补值。
    """

    name: str = "lightgbm"
    handles_nan: bool = True
    needs_imputation: bool = False
    uses_seed: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 20, "subsample": 0.9, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": -1,
        "random_state": 42, "verbose": -1,
    })

    def fit(self, X, y, X_valid=None, y_valid=None):
        import lightgbm as lgb

        # LightGBM 支持 CSR 稀疏输入，保持稀疏以省内存
        y = np.asarray(y, dtype=np.float64)
        # 静音：40 组随机搜索 × "No further splits" 警告会淹没实验日志
        p = dict(self.params)
        p.setdefault("verbose", -1)
        self._m = lgb.LGBMRegressor(**p)
        fit_kw: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kw["eval_set"] = [(keep_sparse(X_valid), np.asarray(y_valid, dtype=np.float64))]
            fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(period=0)]
        else:
            fit_kw["callbacks"] = [lgb.log_evaluation(period=0)]
        self._m.fit(keep_sparse(X), y, **fit_kw)
        self.best_iteration_ = getattr(self._m, "best_iteration_", None)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._m.predict(keep_sparse(X)), dtype=np.float64)


# ================================================================ 有序回归（累积链接模型）


@dataclass
class OrdinalLogitModel(BaseModel):
    """累积链接模型（ordinal logit），即比例优势模型。

    为什么必须有一条有序回归基线
    ----------------------------
    难度 5 档是 rating **连续量切分**而来的**有序**等级；普通多分类丢失序信息，
    而 QWK 本身就建立在有序结构上。本实现不依赖 `mord`（本机未安装），
    用 **torch 在 GPU 上**直接最大化累积 logit 似然并加 L2 惩罚，
    梯度由 autograd 给出（稀疏输入全程不稠密化，见 `fit` 的说明）。

    ⚠ 输出口径：返回 5 档概率的**期望 rating**（连续值），
    而不是 argmax 档位——否则它就成了一次独立的分类训练，
    违反"分类指标由回归输出分箱派生"的红线。
    """

    name: str = "ordinal_logit"
    needs_imputation: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {
        "l2": 1.0, "max_iter": 500,
    })

    #: 计算期望值用的档位代表点（由**训练集**各档 rating 均值给出）
    _representatives: np.ndarray | None = None

    def _bin_reps(self, y: np.ndarray) -> np.ndarray:
        """档位代表点 = 训练集内该档 rating 的实际均值。

        为什么不用"切点中点"：最高档 (2400, +inf) 的中点若按 data range 取
        会得到 3000，而真实 rating 上限是 3500，等于系统性低估最难档的期望值。
        用训练集实际均值是标准的条件期望估计，且只依赖训练集。
        空档（训练集内无样本）回退为相邻切点的中点。
        """
        bins = self._bin_index(y)
        K = len(C.DIFFICULTY_BINS) + 1
        edges = [-np.inf] + list(C.DIFFICULTY_BINS) + [np.inf]
        reps = np.empty(K, dtype=np.float64)
        for k in range(K):
            m = bins == k
            if m.any():
                reps[k] = float(y[m].mean())
            else:
                lo = edges[k] if np.isfinite(edges[k]) else float(y.min()) - C.RATING_STEP
                hi = edges[k + 1] if np.isfinite(edges[k + 1]) else float(y.max()) + C.RATING_STEP
                reps[k] = (lo + hi) / 2.0
        return reps

    def _bin_index(self, y: np.ndarray) -> np.ndarray:
        """训练标签 → 档位索引。

        ⚠ `right=True`（左开右闭）必须与 `metrics.BinThresholds.apply` 一致，
        否则有序回归的训练分档与评估分档是两套约定、系数含义与报告指标对不上。
        """
        return np.clip(np.digitize(y, list(C.DIFFICULTY_BINS), right=True), 0,
                       len(C.DIFFICULTY_BINS))

    @staticmethod
    def _col_stats(X) -> tuple[np.ndarray, np.ndarray]:
        """列的均值与标准差，**只在训练集上算**。

        稀疏矩阵不能直接 numpy 化（20045 维稠密化要 1.4 GB），
        用 E[X]、E[X²] 的稀疏逐元素乘算出来，代价与非零元个数成正比。
        """
        from scipy import sparse

        if sparse.issparse(X):
            mu = np.asarray(X.mean(axis=0)).ravel()
            sq = X.multiply(X).mean(axis=0)
            ex2 = np.asarray(sq).ravel()
        else:
            Xd = np.asarray(X, dtype=np.float64)
            mu = np.nanmean(Xd, axis=0)
            ex2 = np.nanmean(Xd ** 2, axis=0)
        var = np.maximum(ex2 - mu ** 2, 0.0)
        sigma = np.sqrt(var)
        mu = np.nan_to_num(mu, nan=0.0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1.0)
        return mu, sigma

    def fit(self, X, y, X_valid=None, y_valid=None):
        """在 GPU 上最大化累积 logit 似然（带 L2），梯度由 autograd 给出。

        ⚠ 与原 numpy 实现的两处关键差异，都是为了让它从"跑不完"变成可用：

        1. **解析梯度**：原实现调用 `scipy.minimize(method='L-BFGS-B')` 时
           没有传 `jac`，scipy 改用有限差分，每步要 (d+2) 次函数求值。
           d=20045 时这不是慢，而是根本不可能完成（实测 >300 秒仍在第一步）。
           torch 的 autograd 直接把梯度算出来，成本与一次前向同量级。
        2. **稀疏输入不稠密化**：标准化通过重参数化实现——
           令 γ = β/σ，则 η = Xγ − μ·γ，与 (X−μ)/σ·β 完全等价，
           但全程只做稀疏-稠密乘，20045 维的稠密矩阵一次都不需要出现。

        浮点：稀疏乘用 float32（GPU 上快一个量级），
        概率与似然计算转 float64，避免 sigmoid 相减的灾难性抵消。
        """
        import torch
        from scipy import sparse

        from . import accel as A

        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        K = len(C.DIFFICULTY_BINS) + 1
        bins = self._bin_index(y)
        self._mu, self._sigma = self._col_stats(X)

        device = A.get_device(C.GPU_PREFER)
        A.configure_torch()
        self.device_ = A.device_label(device)
        if sparse.issparse(X):
            A_t = A.to_torch_sparse(X, device=device)
            sparse_input = True
        else:
            A_t = A.to_torch_dense(np.asarray(X, dtype=np.float64), device=device)
            sparse_input = False
        mu_t = torch.as_tensor(self._mu, device=device, dtype=torch.float32)
        sig_t = torch.as_tensor(self._sigma, device=device, dtype=torch.float32)
        yb = torch.as_tensor(bins, device=device, dtype=torch.long)
        arange = torch.arange(n, device=device)

        n_thresh = K - 1
        qs = np.array([(bins <= k).mean() for k in range(n_thresh)], dtype=np.float64)
        qs = np.clip(qs, 0.01, 0.99)
        init_t = np.log(qs / (1 - qs))
        # 阈值参数化：首个阈值自由，后续为"与前一阈值之差取 exp"，保证严格递增
        raw0 = np.concatenate([[init_t[0]], np.zeros(max(n_thresh - 1, 0))]) \
            if n_thresh > 1 else np.array([init_t[0]])

        gamma = torch.zeros(d, device=device, requires_grad=True)
        # 注意：torch.as_tensor 不接受 requires_grad（会 TypeError），要先建再置位
        raw = torch.as_tensor(raw0.astype(np.float32), device=device).requires_grad_(True)
        l2 = float(self.params.get("l2", 1.0))
        neg_inf = torch.tensor(-float("inf"), device=device, dtype=torch.float64)
        pos_inf = torch.tensor(float("inf"), device=device, dtype=torch.float64)

        def thresholds() -> "torch.Tensor":
            if n_thresh == 1:
                return raw[:1].to(torch.float64)
            inc = torch.exp(raw[1:].to(torch.float64))
            return torch.cat([raw[:1].to(torch.float64), raw[:1].to(torch.float64) + torch.cumsum(inc, 0)])

        def eta_of() -> "torch.Tensor":
            if sparse_input:
                eta = torch.sparse.mm(A_t, gamma.unsqueeze(1)).squeeze(1)
            else:
                eta = A_t @ gamma
            return eta - (mu_t * gamma).sum()

        def loss_fn() -> "torch.Tensor":
            eta = eta_of().to(torch.float64)
            th = thresholds()
            ext = torch.cat([neg_inf.unsqueeze(0), th, pos_inf.unsqueeze(0)])
            cum = torch.sigmoid(ext.unsqueeze(0) - eta.unsqueeze(1))     # n×(K+1)
            p = (cum[:, 1:] - cum[:, :-1]).clamp_min(1e-12)
            ll = torch.log(p[arange, yb]).sum()
            beta = gamma * sig_t                                   # β = γσ
            pen = 0.5 * l2 * (beta.to(torch.float64) ** 2).sum()
            return -ll + pen

        opt = torch.optim.LBFGS(
            [gamma, raw], lr=1.0, max_iter=int(self.params.get("max_iter", 500)),
            history_size=10, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        n_closure = {"n": 0}

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            n_closure["n"] += 1
            return loss

        loss0 = float(loss_fn().detach().cpu())
        opt.step(closure)
        final = float(loss_fn().detach().cpu())

        self._gamma_np = gamma.detach().cpu().numpy().astype(np.float64)
        self._beta = self._gamma_np * self._sigma
        self._thresh = thresholds().detach().cpu().numpy().astype(np.float64)
        self._d1 = d
        self._reps = self._bin_reps(y)
        self.opt_result_ = {"success": np.isfinite(final) and final <= loss0 + 1e-6,
                            "nll": final, "nit": n_closure["n"], "device": self.device_}
        # 自检：若所有斜率系数都近似为 0，说明模型退化成常数预测。
        # 这是会直接污染主表排序的严重问题，必须显式暴露而不是静默通过。
        slope = np.abs(self._beta[:d])
        self.degenerate_ = bool(slope.max() < 1e-8)
        if self.degenerate_:
            import warnings

            warnings.warn(
                "OrdinalLogitModel 退化为常数预测（所有斜率系数≈0）。"
                "请检查标准化与 l2 设置。",
                RuntimeWarning, stacklevel=2,
            )
        del A_t, gamma, raw
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _eta(self, X) -> np.ndarray:
        """线性预测子。稀疏输入走 GPU 稀疏乘，不稠密化。

        ⚠ 必须与 `fit` 里的重参数化完全一致：η = Xγ − μ·γ。
        漏掉 `− μ·γ` 会让训练与预测的 η 相差一个常数（等于把整条
        阈值轴平移），指标会整体崩掉——概率全挤到某一档，
        期望 rating 偏离几百，而且从代码上看"只是少了一项"。
        """
        from scipy import sparse

        from . import accel as A

        shift = float(self._mu @ self._gamma_np)
        if sparse.issparse(X):
            import torch

            device = A.get_device(C.GPU_PREFER)
            A_t = A.to_torch_sparse(X, device=device)
            g = torch.as_tensor(self._gamma_np.astype(np.float32), device=device)
            eta = torch.sparse.mm(A_t, g.unsqueeze(1)).squeeze(1)
            return eta.cpu().numpy().astype(np.float64) - shift
        Xd = to_dense_if_needed(X)
        return ((Xd - self._mu) / self._sigma) @ self._beta

    def predict_proba(self, X) -> np.ndarray:
        eta = self._eta(X)
        ext = np.concatenate([[-np.inf], self._thresh, [np.inf]])
        cum = 1.0 / (1.0 + np.exp(-(ext[None, :] - eta[:, None])))
        p = cum[:, 1:] - cum[:, :-1]
        return np.clip(p, 0.0, 1.0)

    def predict(self, X) -> np.ndarray:
        """返回期望 rating（连续），供统一分箱派生 macro-F1 / QWK。"""
        p = self.predict_proba(X)
        return p @ self._reps


# ================================================================ 朴素基线


@dataclass
class TagMeanBaseline(BaseModel):
    """朴素基线：按算法标签均值预测 rating。

    常规参照基线，用来回答"模型是否只是把标签映射成了难度"。
    **零成本，必须保留**（大纲三·（四））。

    ⚠ 与"题号+Div"弱基线区分：本基线回答"标签信息值多少"，
    弱基线回答"模型是不是在走捷径"，二者目的不同，须分别报告。
    """

    name: str = "tag_mean"
    needs_imputation: bool = False
    handles_nan: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __init__(self, tag_columns: Sequence[str] | None = None, **kw):
        self.tag_columns = list(tag_columns) if tag_columns else []
        self.params = dict(kw)
        self.global_mean_: float = float("nan")
        self.tag_means_: dict[int, float] = {}

    def fit(self, X, y, X_valid=None, y_valid=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.global_mean_ = float(y.mean())
        # 逐列（标签）在训练集上的均值
        for j in range(X.shape[1]):
            m = X[:, j] > 0.5
            self.tag_means_[j] = float(y[m].mean()) if m.sum() >= 5 else self.global_mean_
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        for i in range(len(X)):
            active = np.flatnonzero(X[i] > 0.5)
            if len(active) == 0:
                out[i] = self.global_mean_
            else:
                out[i] = float(np.mean([self.tag_means_[int(j)] for j in active]))
        return out


class ShiftedBaseline(BaseModel):
    """常数基线（预测训练集均值）：最保守的下界参照。

    不是主表项，仅用于确认"任何模型都不该输给平均值"。
    """

    name: str = "mean_constant"
    needs_imputation = False
    handles_nan = True

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=np.float64)


# ================================================================ 工厂


#: 传统侧模型的默认构造器
TABULAR_REGISTRY: dict[str, type[BaseModel]] = {
    "ridge": RidgeModel,
    "svr": SVRModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "ordinal_logit": OrdinalLogitModel,
    "mean_constant": ShiftedBaseline,
}


def make_tabular_model(name: str, tag_columns: Sequence[str] | None = None,
                       params: Mapping[str, Any] | None = None) -> BaseModel:
    if name == "tag_mean":
        m = TagMeanBaseline(tag_columns=tag_columns)
    elif name in TABULAR_REGISTRY:
        m = TABULAR_REGISTRY[name]()
    else:
        raise ValueError(f"未知传统模型: {name!r}")
    if params:
        m.set_params(params)
    return m


# ================================================================ 超参搜索空间

#: 大纲四·（一）：线性/树模型随机搜索 30～50 组。此处给出各模型的搜索空间。
TABULAR_SEARCH_SPACE: dict[str, dict[str, Sequence[Any]]] = {
    "ridge": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "svr": {
        # ⚠ 上界必须留够：第一轮搜索在 C=100、epsilon=1 处取到最优，
        # 而这两个值正好压在搜索空间的上/下边界上——说明真正的极值在边界之外，
        # 照原空间定参会让 SVR 被人为削弱。第二轮把上界扩到 1000、下探到 0.5。
        "C": [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "epsilon": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        "gamma": ["scale", 0.001, 0.005, 0.01, 0.05],
        "n_components": [C.NYSTROEM_COMPONENTS],
    },
    "random_forest": {
        # ⚠ max_features 的取值必须结合特征维度来定：本文矩阵是 20045 维，
        # 按"比例 0.3～1.0"搜索意味着每个分裂点要扫 6000～20000 个特征，
        # 单次训练要十几分钟（实测：50 棵、0.3 的比例已需 13 秒），
        # 40 组随机搜索要跑几个小时，而且这么大比例本身也不符合随机森林的
        # 常规做法（回归默认全特征，分类默认 sqrt(d)）。
        # 改为按"每次分裂实际考察的特征数"设档：0.01 ≈ 200 个（≈sqrt(20045)）、
        # 0.05 ≈ 1000 个、0.1 ≈ 2000 个、0.3 ≈ 6000 个，
        # 既覆盖了常规档位，也把单次训练压到 1 分钟量级。
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2, 5],
        "max_features": [0.01, 0.05, 0.1, 0.3],
    },
    "lightgbm": {
        "n_estimators": [300, 600, 1000, 1500],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "num_leaves": [15, 31, 63, 127, 255],
        "min_child_samples": [5, 10, 20, 40, 80],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 5.0, 20.0],
    },
    "ordinal_logit": {"l2": [0.1, 0.5, 1.0, 5.0, 10.0]},
}
