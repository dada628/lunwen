"""论文实验的"可执行规格"：大纲中所有口径在此冻结为常量，正文数字只能从此处派生。

为什么把口径写成代码
--------------------
大纲【八】写作红线里有一半是"口径一致性"检查（A−B 的表述、SVR 保留在主表、
融合模型写"收窄+残余"、指标恰好 5 项、时间划分三段式……）。这些如果只靠人记，
成稿时必然漂移。本模块把它们变成**常量与断言**：任何违反大纲的配置会在导入或
`validate()` 时立刻抛错，而不是等审稿人指出。

⚠ 修改本文件等于修改论文的实验定义，必须同步更新大纲【四】【五】【八】。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------- 路径

#: 仓库根目录（src/task2difficulty/config.py -> 上溯三级）
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
OUTPUTS_DIR: Final[Path] = REPO_ROOT / "outputs"
RESULTS_DIR: Final[Path] = OUTPUTS_DIR / "results"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
REPORTS_DIR: Final[Path] = OUTPUTS_DIR / "reports"

#: 所有模型共用的划分索引文件（大纲二·（五）："共用同一索引文件，所有模型不得自行划分"）
SPLIT_INDEX_PATH: Final[Path] = PROCESSED_DIR / "split_index.csv"
DATASET_PATH: Final[Path] = PROCESSED_DIR / "dataset.csv"

# ---------------------------------------------------------------- 目标标签

#: rating 取值范围（大纲四·（一）主任务为回归，预测 rating）
RATING_MIN: Final[int] = 800
RATING_MAX: Final[int] = 3500

#: CF rating 以 100 为步长，且 800 为下钳位（大纲二·（三）"两个必须写明的标签事实"之一）
RATING_STEP: Final[int] = 100
RATING_FLOOR: Final[int] = 800
#: ±200 命中率的带宽（大纲三·（五））
HIT_TOLERANCE: Final[int] = 200

# ---------------------------------------------------------------- 划分协议

#: 随机划分比例（训练:验证:测试）——大纲四·（二）表：8:1:1
RANDOM_SPLIT: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)

#: 时间划分三段式边界——⚠ 大纲红线：验证集必须按时间切，不得随机抽样
TEMPORAL_TRAIN_END_YEAR: Final[int] = 2021   # 训练集：year <= 2021 (< 2022)
TEMPORAL_VALID_YEAR: Final[int] = 2022       # 验证集：year == 2022
TEMPORAL_TEST_START_YEAR: Final[int] = 2023  # 测试集：year >= 2023

#: 跨随机种子的划分种子固定值；深度模型多种子只在**训练**阶段变，划分不变
SPLIT_SEED: Final[int] = 20260101

#: 大纲【十】实测占比，用于自检：数据管线算出的占比与此偏差过大说明数据源变了
EXPECTED_TEMPORAL_RATIOS: Final[dict[str, float]] = {
    "train": 0.655, "valid": 0.075, "test": 0.270,
}
EXPECTED_RANDOM_RATIOS: Final[dict[str, float]] = {
    "train": 0.800, "valid": 0.100, "test": 0.100,
}
#: 占比自检容差（绝对百分点）。超差只告警不报错——快照日期推移会让占比缓慢漂移
RATIO_TOLERANCE_PP: Final[float] = 3.0

# ---------------------------------------------------------------- 特征组

#: 特征组名 -> 中文显示名（表 2 用）
FEATURE_GROUP_LABELS: Final[dict[str, str]] = {
    "text": "文本特征",
    "constraint": "约束规模特征",
    "tag": "标签特征",
    "stat": "统计特征",
    "provenance": "来源特征",
}

#: 新题可得性（表 2 的"新题可得性"列，必须与大纲一致）
FEATURE_GROUP_AVAILABILITY: Final[dict[str, str]] = {
    "text": "完全可得",
    "constraint": "完全可得",
    "tag": "完全可得",
    "stat": "新题不可得",
    "provenance": "完全可得但高危",
}

#: 纯冷启动下仍完全可得的结构化特征组（大纲三·（三）：残余增益的来源）
COLD_START_STRUCT_GROUPS: Final[tuple[str, ...]] = ("tag", "constraint")

#: ⚠ 来源特征：标签的强代理，不得作为普通特征混入，必须单列 + provenance 消融 + 弱基线自曝
PROVENANCE_FEATURES: Final[tuple[str, ...]] = ("problem_index", "div_type", "contest_id")

#: 约束规模特征（大纲二·（二）：冷启动场景的核心特征）
CONSTRAINT_FEATURES: Final[tuple[str, ...]] = (
    "constraint_n_upper",     # n 的上界
    "multi_test_count",       # 多测组数 t
    "is_interactive",         # 是否交互题
)

#: 统计特征（新题不可得）。这里必须区分"列不存在"与"信息不存在"：
#: 冷启动条件下这些列以**缺失**存在，而不是填 0——填 0 会伪造"这题没人做过"。
#:
#: ⚠ 关于"通过率"的一处实测校正（与大纲二·（四）特征表的差异，须在正文说明）：
#: 大纲表 2 列了"`solvedCount`、通过率（派生）、上线天数"。但实测确认
#: `problemset.problems` 的 `ProblemStatistics` **只有 `contestId`/`index`/`solvedCount`
#: 三个字段**，既无尝试人数也无提交总数，因此**通过率无法派生**。
#: 强行用其他字段凑一个"通过率"会引入与 `solved_count` 高度共线的伪特征，
#: 反而污染 A−B 的差分解释。
#: 处理：主实验的统计特征组收敛为下列**两项**；若将来补做 S1'（`contest.standings`），
#: 可从 `problemResults[].rejectedAttemptCount` 重建真正的通过率（见 `S1_ONLY_STAT_FEATURES`）。
STAT_FEATURES: Final[tuple[str, ...]] = (
    "solved_count",           # 累计通过人数（快照值，只增不减）
    "days_since_publish",     # 上线天数 = 快照日期 − 比赛开始日期
)

#: 仅在 S1'（赛时统计）条件下可得、需爬 `contest.standings` 才能重建的统计特征。
#: 主实验不含这些列，因此 `STAT_FEATURES` 才是"置空"要移除的集合。
S1_ONLY_STAT_FEATURES: Final[tuple[str, ...]] = (
    "acceptance_rate",        # 通过率 = 通过人数 / 尝试人数（需 rejectedAttemptCount）
    "attempts_in_contest",    # 赛时尝试次数
    "first_solve_seconds",    # 首解时间（赛时，历史固定）
)

#: ⚠⚠ 冷启动条件下线性/神经网络模型的缺失值处理——这是本文最容易自己踩的隐蔽泄漏。
#:
#: 问题：统计特征置空后，岭回归/SVR/有序回归/MLP/融合模型的结构化通道都不能吃 NaN。
#: 如果按常规做法"用训练集中位数填补"，在 A 格（统计特征可得）下是合理的；
#: 但在 B/D 格（冷启动）下，中位数本身是**从训练集统计特征算出来的**，
#: 等于把"训练集里这道题的累计通过人数长什么样"这一信息偷偷塞回冷启动场景，
#: 会系统性高估 B/D 格的能力，直接污染 A−B 与 B−D 两个核心差分。
#:
#: 处理：冷启动格统一使用**哨兵值填补 + 显式缺失指示列**。
#:   - 哨兵值取该列的常量 0（等价于"无信息"，不是"典型值"）；
#:   - 同时加入 `<col>__missing` 指示列，让模型知道"这一维不存在"。
#: 这套处理对**所有**非树模型一致适用，且完全是常量变换（不依赖训练集统计量），
#: 因此不构成泄漏。实现见 experiments.py 的 `prepare_struct_matrix`。
COLD_START_IMPUTATION: Final[str] = "sentinel_zero_plus_missing_indicator"
#: 需要"哨兵填补 + 缺失指示"的模型（不吃 NaN 的）
#:
#: 实测校正：随机森林原本按"树模型原生吃 NaN"处理，但 sklearn 的随机森林
#: **只在稠密输入下**支持缺失值；稀疏输入遇到 NaN 会直接报
#: "does not accept missing values encoded as NaN natively"。
#: 而本文的特征矩阵必须保持稀疏（TF-IDF 拼进来后稠密化要 1.3 GB，
#: 且极值会让 float32 溢出）。两条约束合起来只有一条出路：
#: 随机森林也走"哨兵 0 + 缺失指示列"口径，与线性/核方法/神经网络一致。
IMPUTATION_REQUIRED_MODELS: Final[tuple[str, ...]] = (
    "ridge", "svr", "ordinal_logit", "random_forest", "mlp", "fusion",
)
#: 原生处理 NaN 的模型（LightGBM 的稀疏输入可直接携带 NaN 作为缺失）
NAN_NATIVE_MODELS: Final[tuple[str, ...]] = ("lightgbm",)

#: 标签特征：multi-hot 由 {tag}_ 前缀列承载，另加标签数量
TAG_MULTIHOT_PREFIX: Final[str] = "tag__"
TAG_COUNT_FEATURE: Final[str] = "tag_count"

#: 文本端需要剥离的模板句（大纲四·（四）文本端 provenance 剥离）。
#: ⚠ 口径：这是"捷径依赖度诊断"，**不是**泄漏剔除——模板句对新题同样可得。
#:
#: 实测校正（重要）：实际出现形式比大纲的举例更短、更自由。
#: 例如 CF 1286C1/C2 的**标题本身**就写作 "Madhouse (easy version)" /
#: "Madhouse (hard version)"，而大纲举例的整句 "This is an easy version"
#: 在实测 391 题样本中命中数为 **0**。若照抄大纲的整句模式，
#: 文本端消融会因"一条都匹配不到"而完全失效（实测踩过）。
#: 因此改为下面这组**短模式**，并在正文中说明实际匹配口径。
TEMPLATE_PHRASES: Final[tuple[str, ...]] = (
    "easy version",
    "hard version",
    "interactive problem",
    "this problem is interactive",
    "the only difference between the versions",
    "the only difference between easy and hard versions",
    "difference between the versions",
)

# ---------------------------------------------------------------- 因子设计（A/B/C/D）

#: 划分方式
SPLIT_RANDOM: Final[str] = "random"
SPLIT_TEMPORAL: Final[str] = "temporal"

#: 统计特征条件
STAT_FULL: Final[str] = "full"        # 全量快照
STAT_MASKED: Final[str] = "masked"    # 置空
STAT_IN_CONTEST: Final[str] = "in_contest"  # S1' 赛时统计（可选项）

#: 中文显示名（表 4 行标题）
STAT_CONDITION_LABELS: Final[dict[str, str]] = {
    STAT_FULL: "全量快照",
    STAT_MASKED: "置空",
    STAT_IN_CONTEST: "赛时统计",
}


@dataclass(frozen=True)
class Condition:
    """因子设计中的一格。

    Attributes
    ----------
    key: A/B/C/D/S1' 之类的短代号
    split: SPLIT_RANDOM 或 SPLIT_TEMPORAL
    stat: 统计特征条件
    positioning: 该格在正文中的"定位"（大纲四·（二）表格原文口径）
    leaky: 含"标签—特征同源"或"预测时点不可得的信息"。True 的格**不可作归因或部署依据**

        ⚠ 两种成因必须分开讲，否则会把 A 格错按成 C 格的问题：
          - **同源（A、C 都有）**：rating 与累计通过人数由同一作答过程共同决定，
            故精度含标签—特征同源成分，**不能读作"模型理解了题面"**。
          - **时间泄漏（只有 C）**：C 的测试集是 2023 年后的新题，却用它们上线
            数年后的累计统计。这些统计在"题目刚上线"的预测时点上并不存在，
            属用最终统计预测最终标签，**故 C 不作部署依据**。
          A 是随机划分，统计值与标签取自同一快照时点，**不含未来信息**，
          对应"存量题库定级"这一真实任务，故 A 的绝对值是有效结果，
          只是不可作归因解读。

    in_main_conclusion: 是否进入主结论。大纲：主结论只建立在 A/B/D 三格
    """

    key: str
    split: str
    stat: str
    positioning: str
    leaky: bool
    in_main_conclusion: bool
    optional: bool = False

    @property
    def label(self) -> str:
        return f"{self.key}（{'随机' if self.split == SPLIT_RANDOM else '时间'}划分 + " \
               f"{STAT_CONDITION_LABELS[self.stat]}）"


#: 大纲四·（二）的因子设计表，逐字对应
CONDITIONS: Final[dict[str, Condition]] = {
    "A": Condition(
        key="A", split=SPLIT_RANDOM, stat=STAT_FULL,
        positioning="存量题库场景（任务 I）：统计可得时对已入库题目定级；"
                    "精度含标签—特征同源成分，不可作归因解读",
        leaky=True, in_main_conclusion=True,
    ),
    "B": Condition(
        key="B", split=SPLIT_RANDOM, stat=STAT_MASKED,
        positioning="纯冷启动能力（题面+标签+约束）",
        leaky=False, in_main_conclusion=True,
    ),
    "C": Condition(
        key="C", split=SPLIT_TEMPORAL, stat=STAT_FULL,
        positioning="⚠ 含时间泄漏的乐观上界——测试集新题用了上线数年后的累计统计，"
                    "该信息在预测时点并不存在，不作部署依据",
        leaky=True, in_main_conclusion=False,
    ),
    "D": Condition(
        key="D", split=SPLIT_TEMPORAL, stat=STAT_MASKED,
        positioning="★ 最真实的新题场景",
        leaky=False, in_main_conclusion=True,
    ),
    "S1": Condition(
        key="S1", split=SPLIT_TEMPORAL, stat=STAT_IN_CONTEST,
        positioning="中间档：赛时统计（历史固定、赛后可立即获得）",
        leaky=False, in_main_conclusion=False, optional=True,
    ),
}

#: B' 规模对照：把随机划分训练集下采样到与时间划分同规模（仅 LightGBM 与 BERT 各一次）
SIZE_CONTROL_KEY: Final[str] = "B'"
SIZE_CONTROL_NOTE: Final[str] = "随机划分·训练集下采样至时间划分同规模（约 65.5%）"

# ---------------------------------------------------------------- 差分定义（口径冻结）

#: 差分列的表头。**必须写明是哪两个量相减**，否则读者无法判断正负号方向。
DIFF_COLUMN_HEADER: Final[str] = "MAE 差值（左 − 右）"

#: 差分方向措辞。**只给方向与幅度，不带符号**——
#: 表格里已有带符号的列，若措辞里再重复一次数字，正文读起来会像
#: "差为 -297.48 分，A 更优，领先 297.48 分"这样的复读（实测踩过）。
DIFF_DIRECTION_TEMPLATE: Final[str] = "{winner} 更优，领先 {gap:.2f} 分"
DIFF_TIE_TEMPLATE: Final[str] = "两格持平"


def describe_difference(delta: float, left_key: str, right_key: str) -> str:
    """把一个 MAE 差值（左 − 右）翻译成方向化措辞。

    MAE 是"越小越好"，所以差值为负意味着**左侧更优**。本函数把这一层
    反过来讲清楚，避免正文出现"A−B 为负 = 统计特征贡献为负"的误读。

    设计取舍：**只返回方向与幅度，不重复带符号的差值**。
    带符号的原始差值由表格的数值列承担，正文措辞承担"谁更好、好多少"，
    两者分工，避免同一句话里出现两个数字。
    """
    if not (delta == delta):          # NaN
        return "—"
    if abs(delta) < 0.005:
        return DIFF_TIE_TEMPLATE
    winner = left_key if delta < 0 else right_key
    return DIFF_DIRECTION_TEMPLATE.format(winner=winner, gap=abs(delta))


#: 差分表（大纲四·（二））。text 为**必须写进正文的表述**，不得改写。
DIFFERENCES: Final[dict[str, dict[str, object]]] = {
    "A-B": {
        "label": "A − B",
        "meaning": "统计特征的总边际贡献（含同源成分）",
        "clean": True,
        "caveat": "A 格本身是含同源泄漏的上界，故此差值**不等于**"
                  "“统计特征的纯净信息价值”；不得表述为“信息价值”",
    },
    "B-D": {
        "label": "B − D",
        "meaning": "跨时间泛化损失（含训练量差异）",
        "clean": False,
        "caveat": "混杂训练集规模。须报告两套划分的**实测**三段规模，并做 B' 规模对照后方可讨论"
                  "时间漂移；⚠ 下面的 14.5 pp 是**全量 11152 题**下的实测差（随机 80.0% vs "
                  "时间 65.5%），抽样数据的实测差会不同（如 391 题抽样为 +9.5 pp），"
                  "**正文必须引用当次实验的实测值，不可照抄 14.5 pp**。",
        "full_data_size_gap_pp": 14.5,
    },
    "C-D": {
        "label": "C − D",
        "meaning": "统计特征价值（时间划分下）",
        "clean": False,
        "caveat": "被泄漏放大，仅作上界",
    },
    "A-C": {
        "label": "A − C",
        "meaning": "时间漂移（含统计特征时）",
        "clean": False,
        "caveat": "同上，仅作上界",
    },
    "B'-D": {
        "label": "B′ − D",
        "meaning": "剔除训练量差异后的时间效应（仍含训练集构成与测试期分布迁移）",
        "clean": False,
        "caveat": "B′ 只控制了训练规模，未控制训练集构成与测试集不同；"
                  "**不得**写成“纯净”",
    },
}

# ---------------------------------------------------------------- 模型清单（主表口径）

#: 传统侧 6 项（大纲三·（一）+ 三·（四）+ 四·（三）主表构成）
TRADITIONAL_MODELS: Final[tuple[str, ...]] = (
    "ridge",        # 岭回归（线性基线）
    "svr",          # ★ 保留在主表：承担"核方法（非线性但非集成）"对照
    "random_forest",
    "lightgbm",
    "ordinal_logit",  # 有序回归基线（累积链接模型）
    "tag_mean",       # 朴素基线：按算法标签均值预测 rating
)

#: 深度侧主表 4 项（大纲三·（二）（三））
DEEP_MODELS: Final[tuple[str, ...]] = (
    "mlp",      # 结构化特征通道基准
    "textcnn",  # 题面文本
    "bert",     # bert-base-uncased，[CLS] 接回归头
    "fusion",   # 双通道融合（定位为"基线级融合"，不宣称结构创新）
)

#: 附表项：BiLSTM+Attention，不进主表
APPENDIX_MODELS: Final[tuple[str, ...]] = ("bilstm_attn",)

#: 主表行数（大纲【九】：传统 6 + 深度 4 = 10 行）
MAIN_TABLE_MODELS: Final[tuple[str, ...]] = TRADITIONAL_MODELS + DEEP_MODELS

#: 各模型输入通道
MODEL_INPUT_CHANNEL: Final[dict[str, str]] = {
    "ridge": "tfidf+struct",
    "svr": "tfidf+struct",
    "random_forest": "tfidf+struct",
    "lightgbm": "tfidf+struct",
    "ordinal_logit": "tfidf+struct",
    "tag_mean": "tag_only",
    "mlp": "struct",
    "textcnn": "text",
    "bert": "text",
    "fusion": "text+struct",
    "bilstm_attn": "text",
}

# ---------------------------------------------------------------- GPU 加速与文本降维

#: 是否优先使用 GPU。关掉只影响速度，不影响结果口径（CPU 路径数值等价）。
GPU_PREFER: Final[str] = "cuda"

#: 文本通道需要先降维的传统模型。
#:
#: 为什么只有 SVR：它在 8864×20045 的高维稀疏表示上要解 n×n 的核 QP，
#: 实测 2.3 小时未收敛；而高维稀疏下 RBF 核的距离集中效应会让全维核矩阵
#: 本身失去分辨力。降维到 200 维后既跑得动，核也有意义。
#: 岭回归/树模型/有序回归都不需要（前者单次 <1 秒，后两者用稀疏路径）。
TEXT_SVD_MODELS: Final[tuple[str, ...]] = ("svr",)
#: 随机化截断 SVD 的目标维度（只在训练集上拟合）
TEXT_SVD_DIMS: Final[int] = 200
#: 随机化 SVD 的幂迭代次数（2 次足以把尾部奇异值估准）
TEXT_SVD_N_ITER: Final[int] = 2

#: RBF Nyström 展开的地标点数（核 SVR 的显式低秩近似）
NYSTROEM_COMPONENTS: Final[int] = 800

# ---------------------------------------------------------------- 超参预算分档

#: 大纲四·（一）：超参预算按模型族分档。调参只在 A 格进行，固定后复用到 B/C/D。
HYPERPARAM_BUDGET: Final[dict[str, int]] = {
    "linear_tree": 40,   # 线性/树模型：随机搜索 30～50 组
    "small_nn": 12,      # TextCNN/MLP：小网格 10～15 组
    "bert": 9,           # BERT：小网格 6～12 组（学习率 × epoch）
}
BUDGET_RANGES: Final[dict[str, tuple[int, int]]] = {
    "linear_tree": (30, 50),
    "small_nn": (10, 15),
    "bert": (6, 12),
}
MODEL_BUDGET_FAMILY: Final[dict[str, str]] = {
    "ridge": "linear_tree",
    "svr": "linear_tree",
    "random_forest": "linear_tree",
    "lightgbm": "linear_tree",
    "ordinal_logit": "linear_tree",
    "tag_mean": "linear_tree",   # 无超参，占位
    "mlp": "small_nn",
    "textcnn": "small_nn",
    "bert": "bert",
    "fusion": "bert",            # 含 BERT 主干，预算归入 BERT 档
    "bilstm_attn": "small_nn",
}

#: 随机种子（大纲四·（一）：深度模型 3～5 个随机种子，报告均值±标准差）
SEEDS: Final[tuple[int, ...]] = (42, 1337, 2024, 7, 20260101)
MIN_SEEDS: Final[int] = 3

#: 配对 bootstrap 的重复次数与置信水平（大纲三·（五）显著性口径）
BOOTSTRAP_N: Final[int] = 1000
BOOTSTRAP_ALPHA: Final[float] = 0.05

# ---------------------------------------------------------------- 指标（恰好 5 项）

#: ⚠ 大纲红线：指标收敛为 5 项，全部由**一次回归训练**的输出派生；
#: 分类指标不得单独训练，阈值只在验证集上确定。
METRICS: Final[tuple[str, ...]] = ("mae", "hit200", "macro_f1", "qwk", "spearman")

METRIC_LABELS: Final[dict[str, str]] = {
    "mae": "MAE（rating 点）",
    "hit200": "±200 命中率",
    "macro_f1": "macro-F1",
    "qwk": "QWK",
    "spearman": "Spearman ρ",
}

#: 已废弃指标：出现即视为违反大纲（RMSE/R² 与 MAE 信息重叠）
FORBIDDEN_METRICS: Final[tuple[str, ...]] = (
    "rmse", "r2", "r²", "cliffs_delta", "cliff's_delta", "ndcg",
)

#: 难度分档（5 档）：等宽与等频两种口径都要报（大纲二·（三））
DIFFICULTY_BINS: Final[tuple[float, ...]] = (1200, 1600, 2000, 2400)
BINNING_SCHEMES: Final[tuple[str, ...]] = ("equal_width", "equal_freq")

# ---------------------------------------------------------------- 数据效率曲线

DATA_EFFICIENCY_FRACTIONS: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 1.00)

# ---------------------------------------------------------------- 人类一致性实验

HUMAN_KAPPA_N_ITEMS: Final[int] = 100
HUMAN_KAPPA_N_RATERS: Final[int] = 3
#: 分层：低/中/高各约 1/3，且新题与旧题混合
HUMAN_KAPPA_STRATA: Final[int] = 3

# ---------------------------------------------------------------- 自检

def validate() -> list[str]:
    """对冻结口径做一致性自检，返回问题列表（空列表 = 通过）。

    这里逐条对应大纲【八】写作红线中可被代码检查的项。
    """
    problems: list[str] = []

    # 1. 主表行数必须恰好 10（传统 6 + 深度 4）
    if len(MAIN_TABLE_MODELS) != 10:
        problems.append(f"主表应为 10 行，当前 {len(MAIN_TABLE_MODELS)} 行")

    # 2. SVR 必须在主表（红线：不得降为附表）
    if "svr" not in TRADITIONAL_MODELS:
        problems.append("SVR 必须保留在主表（大纲红线）")
    if "svr" in APPENDIX_MODELS:
        problems.append("SVR 不得出现在附表")

    # 3. BiLSTM+Attention 必须在附表、不得进主表
    if "bilstm_attn" in MAIN_TABLE_MODELS:
        problems.append("BiLSTM+Attention 应降为附表，不得进主表")

    # 4. 指标恰好 5 项且不含废弃指标
    if set(METRICS) != {"mae", "hit200", "macro_f1", "qwk", "spearman"}:
        problems.append(f"指标必须恰为 MAE/±200/macro-F1/QWK/Spearman，当前 {METRICS}")
    for bad in FORBIDDEN_METRICS:
        if bad in METRICS:
            problems.append(f"指标中包含已废弃的 {bad}")

    # 5. 主结论格必须包含 A/B/D，且不含 C
    main = {k for k, c in CONDITIONS.items() if c.in_main_conclusion}
    if main != {"A", "B", "D"}:
        problems.append(f"主结论应建立在 A/B/D 三格，当前 {sorted(main)}")
    if not CONDITIONS["C"].leaky:
        problems.append("C 格必须标注为泄漏格")
    if CONDITIONS["C"].in_main_conclusion:
        problems.append("C 格不得进入主结论（大纲红线）")
    if CONDITIONS["A"].leaky is False:
        problems.append("A 格含同源泄漏，leaky 应为 True")

    # 6. 时间划分三段式：验证年必须严格落在训练与测试之间
    if not (TEMPORAL_TRAIN_END_YEAR < TEMPORAL_VALID_YEAR < TEMPORAL_TEST_START_YEAR):
        problems.append("时间划分三段式边界非法（必须 训练年 < 验证年 < 测试起始年）")

    # 7. 差分表述必须含大纲要求的限定语，防止回退到旧口径
    ab = str(DIFFERENCES["A-B"]["meaning"])
    if "含同源成分" not in ab:
        problems.append("A−B 表述必须含“含同源成分”限定")
    if "信息价值" in ab:
        problems.append("A−B 不得表述为“信息价值”（大纲明确排除的旧口径）")
    bd = str(DIFFERENCES["B-D"]["meaning"])
    if "含训练量差异" not in bd:
        problems.append("B−D 表述必须含“含训练量差异”限定")
    bpd = str(DIFFERENCES["B'-D"]["meaning"])
    if "纯净" in bpd and "不得" not in bpd:
        problems.append("B′−D 不得写成“纯净”")

    # 8. 超参预算落在大纲给的分档区间内
    for fam, n in HYPERPARAM_BUDGET.items():
        lo, hi = BUDGET_RANGES[fam]
        if not (lo <= n <= hi):
            problems.append(f"{fam} 预算 {n} 不在大纲区间 [{lo}, {hi}]")

    # 9. 每个主表模型都要有预算分族与输入通道定义
    for m in MAIN_TABLE_MODELS:
        if m not in MODEL_BUDGET_FAMILY:
            problems.append(f"模型 {m} 缺少超参预算分族")
        if m not in MODEL_INPUT_CHANNEL:
            problems.append(f"模型 {m} 缺少输入通道定义")

    # 10. 来源特征不得出现在普通特征组里
    for g in ("text", "constraint", "tag", "stat"):
        for f_ in PROVENANCE_FEATURES:
            if f_ in globals().get("CONSTRAINT_FEATURES", ()) and g == "constraint":
                problems.append(f"来源特征 {f_} 混入了约束规模特征组")

    # 11. 多种子下限
    if len(SEEDS) < MIN_SEEDS:
        problems.append(f"种子数 {len(SEEDS)} 少于下限 {MIN_SEEDS}")

    # 12. 缺失值口径必须覆盖全部主表模型（谁都不许悄悄走"训练集中位数填补"）
    covered = set(IMPUTATION_REQUIRED_MODELS) | set(NAN_NATIVE_MODELS)
    for m in MAIN_TABLE_MODELS:
        # 只用文本通道的模型没有结构化输入，不涉及缺失值口径
        if "struct" not in MODEL_INPUT_CHANNEL.get(m, ""):
            continue
        if m not in covered:
            problems.append(f"模型 {m} 未定义缺失值口径（哨兵填补 or 原生 NaN）")

    # 13. 文本降维只允许给声明的模型，且降维维度必须小于词表规模量级
    for m in TEXT_SVD_MODELS:
        if m not in MAIN_TABLE_MODELS:
            problems.append(f"文本降维模型 {m} 不在主表")
        if MODEL_INPUT_CHANNEL.get(m, "").find("tfidf") < 0:
            problems.append(f"文本降维模型 {m} 的输入通道不含词袋特征，降维无意义")
    if TEXT_SVD_DIMS <= 0 or TEXT_SVD_DIMS > 1000:
        problems.append(f"文本 SVD 维度 {TEXT_SVD_DIMS} 不在合理范围 (0, 1000]")
    if NYSTROEM_COMPONENTS < 100:
        problems.append(f"Nyström 地标点数 {NYSTROEM_COMPONENTS} 过少，核近似会失真")

    return problems


if __name__ == "__main__":
    issues = validate()
    if issues:
        print("口径自检未通过：")
        for i in issues:
            print("  ✗", i)
        raise SystemExit(1)
    print("口径自检通过。当前冻结的论文口径：")
    print(f"  主表模型  : {len(MAIN_TABLE_MODELS)} 行 = 传统 {len(TRADITIONAL_MODELS)} + 深度 {len(DEEP_MODELS)}")
    print(f"  附表模型  : {', '.join(APPENDIX_MODELS)}")
    print(f"  指标      : {', '.join(METRIC_LABELS[m] for m in METRICS)}")
    print(f"  因子格    : {', '.join(c.label for c in CONDITIONS.values())}")
    print(f"  主结论格  : {', '.join(sorted(k for k, c in CONDITIONS.items() if c.in_main_conclusion))}")
    print(f"  时间划分  : 训练<={TEMPORAL_TRAIN_END_YEAR} / 验证={TEMPORAL_VALID_YEAR} / "
          f"测试>={TEMPORAL_TEST_START_YEAR}")
    print(f"  随机划分  : {RANDOM_SPLIT[0]:.0%}:{RANDOM_SPLIT[1]:.0%}:{RANDOM_SPLIT[2]:.0%}")
    print(f"  超参预算  : {HYPERPARAM_BUDGET}")
