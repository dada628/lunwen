"""论文配图生成（大纲【五】：4 表 4 图）。

严格遵守项目内两份配图规范
--------------------------
- 《配图规范 ① · 架构图》：直角几何 + 灰阶 + 唯一强调色；正交折线；分区细点线框；
  **图内无标题**；SimSun 字体；16.0 cm 宽 × 0.72 高宽比；PNG 600 dpi + 矢量 PDF。
- 《配图规范 ② · 数据图》：灰阶 4 级 + 唯一强调色；**无网格线**；刻度朝内；
  只留左/下轴脊；图例无框；轴标目「量 / 单位」格式；中文标签用横向条形图。

⚠ **宽度必须按本刊版心调整**（规范 §13 明确允许）：规范给的 16.0 cm 是按
"版心 16.2～17.0 cm"定的，而《广东农工商职业技术学院学报》模板为
A4 + 左右各 3.17 cm 页边距，版心实为 **21.0 − 3.17×2 = 14.66 cm**。
若仍按 16.0 cm 出图，Word 排版时会被压缩到 14.66 cm，字号与线宽随之失真
（规范要求"最终印刷尺寸下所有文字 ≥5.5/6.5 pt"）。因此本模块取 **14.6 cm**
为通栏宽、9.0 cm 为单栏宽，其余规格不变。

⚠ **图内不写标题**：题注由 Word 文档提供（图题在下）。因此这里不调用
`suptitle()` / `set_title()`；面板编号 (a)/(b) 用 axes 坐标文字标注。
误差棒含义、区间底色含义等**注释一律写进 Word 题注**，不在图内堆文字。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C

# ================================================================ 全局规格

CM = 1 / 2.54

#: 通栏画布宽（cm）。**这是绘图画布尺寸，不是最终显示宽度。**
#: ⚠ 保持 14.6 不变：图内各元素的坐标/边距都按这个画布调好，
#:   改画布宽会改变宽高比（实测把 W_FULL 提到 16.8 后，图 2 的
#:   宽高比由 1.404 变成 1.616，条形图被压扁）。
#:   最终显示宽度由文档侧的 `wp:extent` 决定（本文排为通栏 16.85 cm），
#:   放大倍率 = 显示宽 ÷ 画布宽，故字号需按该倍率预先缩小，
#:   使**放大后**正好达到要求的 pt 值（见 FS_* 的注释）。
W_FULL = 14.6
#: 单栏宽
W_ONE = 9.0

#: 图在 Word 中的实际显示宽度（cm）。通栏 = 版心宽 16.85；见
#: `journal_profile_szpu.json` 的 `page.body_width_cm`。
#: 字号按 `DISPLAY_W / W_FULL` 的倍率反算，保证放大后刚好是目标 pt。
DISPLAY_W = 16.85
#: 显示放大倍率。绘图时把字号除以它，放大后就正好是设计 pt 值。
_ZOOM = DISPLAY_W / W_FULL

# ---- 架构图配色（规范 ①）----
INK, MUTE, EDGE, GRID = "#111111", "#4d4d4d", "#333333", "#909090"
ACCENT, ACCENT_FILL = "#1f4e79", "#e9eff7"      # ★ 唯一强调色
GRAY_FILL, LIGHT_FILL = "#f4f4f4", "#fbfbfb"
#: 主标签 / 副标签 / 分区标签。
#: ⚠ 目标值按**放大后**的口径定：本刊已刊论文图内文字实测 9.00 pt
#:   （`_p12` 图2）与 10.56 pt（`_sample` 图1）；须知第九条要求
#:   "图中汉字用小五号宋体"（小五 = 9 pt）。故目标 10.0 / 8.4 / 9.4 pt，
#:   再除以 `_ZOOM` 反算成画布上的字号。
FS_T, FS_S, FS_LBL = 10.0 / _ZOOM, 8.4 / _ZOOM, 9.4 / _ZOOM
LW_ARCH, MS_ARCH = 0.7, 8

# ---- 数据图配色（规范 ②）----
D_INK, D_MUTE, D_GRID = "#111111", "#4d4d4d", "#b0b0b0"
D_ACCENT = "#1f4e79"                             # ★ 唯一强调色
G1, G2, G3, G4 = "#c8c8c8", "#8c8c8c", "#5a5a5a", "#333333"

#: 轴标签 / 刻度 / 图例 / 标注（同样按放大后的口径定，见 `FS_T` 的注释）。
FS_AXIS = 9.0 / _ZOOM
FS_TICK = 9.0 / _ZOOM
FS_LEG = 9.0 / _ZOOM
FS_ANNO = 8.6 / _ZOOM
LW_AXIS, LW_DATA, LW_ERR = 0.7, 1.1, 0.65

#: 首选中文字体。⚠ 规范 ①/② 均强调：**不要**把 Times New Roman 放最前——
#: matplotlib 不做逐字符回退，中文会变方框。SimSun 同时覆盖中文与拉丁字母。
CJK_FONTS: tuple[str, ...] = ("SimSun", "SimHei", "Microsoft YaHei",
                              "Noto Sans SC", "DengXian")


def setup_style() -> dict[str, Any]:
    """应用两份规范的全局 rcParams（数据图部分）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [f for f in CJK_FONTS if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        # 字体（SimSun 优先，中文与拉丁共用）
        "font.family": "sans-serif",
        "font.sans-serif": chosen + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 7.2,
        # ⚠ 数学文本（`$F_1$` 这类下标）默认走 `dejavusans` 字体集，
        #   会渲染成 DejaVuSans——与周围的 SimSun **不同族**，
        #   肉眼表现为"图里有个数字字形不一样"。
        #   本刊要求图内汉字/数字统一（须知第九条），故改用 `stix`
        #   （衬线体，与 SimSun/Times 观感一致），并把 mathtext 的
        #   常规字体也指向首选字族。
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        # 刻度朝内（规范 ② §1.4）
        "xtick.direction": "in",
        "ytick.direction": "in",
        # 轴线
        "axes.linewidth": LW_AXIS,
        "axes.edgecolor": D_INK,
        "axes.labelcolor": D_INK,
        "xtick.color": D_INK,
        "ytick.color": D_INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        # 图例无框（规范 ② §1.6）
        "legend.frameon": False,
        "legend.fontsize": FS_LEG,
        # 矢量字体嵌入 + 画幅保真（规范 ① §4 / ② §4）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,          # 关键：不用 tight，保证物理宽度 = 设定值
        "savefig.pad_inches": 0.02,
        "figure.dpi": 120,
    })
    return {"cjk_fonts": chosen}


def _save(fig, path: Path, also_pdf: bool = True) -> list[Path]:
    """PNG（600 dpi 预览）+ PDF（矢量投稿）。"""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    outs = [path]
    fig.savefig(path, dpi=600)
    if also_pdf:
        p = path.with_suffix(".pdf")
        fig.savefig(p)
        outs.append(p)
    plt.close(fig)
    return outs


def _clean_axes(ax, *, grid: bool = False) -> None:
    """规范 ② §1.3/§1.5：无网格、只留左/下轴脊。"""
    if not grid:
        ax.grid(False)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def _panel_label(ax, text: str) -> None:
    """面板编号 (a)/(b)：用 axes 坐标文字，不调用 set_title（规范 §1.1）。"""
    ax.text(0.015, 0.975, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_LEG, color=D_INK, zorder=5)


from .results import model_label  # noqa: E402


# ================================================================ 图 1 架构图


def figure_fusion_architecture(path: Path) -> list[Path]:
    """图 1：双通道融合模型结构（规范 ①）。

    布局：三个分区自上而下（输入 → 双通道编码 → 融合与回归），
    通道内用竖直箭头、汇入融合框用正交折线；★ 唯一强调色标记**融合环节**
    （本文所讨论的双通道拼接点）。
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    setup_style()
    fig, ax = plt.subplots(figsize=(W_FULL * CM, W_FULL * CM * 0.72))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])          # 画幅铺满，避免默认边距导致尺寸偏差

    def box(x, y, w, h, title, sub=None, fc=GRAY_FILL, ec=EDGE, lw=LW_ARCH,
            tcolor=INK, scolor=MUTE, fs=FS_T):
        """直角矩形框（规范 ① §1.3：禁圆角胶囊）。"""
        ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, facecolor=fc,
                               edgecolor=ec, zorder=2))
        if sub:
            ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                    fontsize=fs, color=tcolor, zorder=3)
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                    fontsize=FS_S, color=scolor, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=fs, color=tcolor, zorder=3)

    def varrow(x, y1, y2):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                     mutation_scale=MS_ARCH, linewidth=LW_ARCH,
                                     color=EDGE, shrinkA=0, shrinkB=0, zorder=4))

    def elbow(x1, y1, x2, y2, ymid):
        """正交折线：下 → 横 → 下（规范 ① §1.4，禁曲线/斜线）。"""
        ax.plot([x1, x1], [y1, ymid], color=GRID, lw=0.65,
                ls=(0, (3.2, 2.2)), zorder=1)
        ax.plot([x1, x2], [ymid, ymid], color=GRID, lw=0.65,
                ls=(0, (3.2, 2.2)), zorder=1)
        ax.add_patch(FancyArrowPatch((x2, ymid), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=MS_ARCH - 1, linewidth=0.65,
                                     color=GRID, shrinkA=0, shrinkB=0, zorder=4))

    def section(x0, y0, x1, y1, num, label):
        """分区：细点线框 + 编号标签（规范 ① §6，zorder=0 在最底层）。"""
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.6,
                               facecolor="none", edgecolor=GRID,
                               linestyle=(0, (3, 2)), zorder=0))
        ax.text(x0 + 1.0, y1 - 1.4, f"{num}　{label}", ha="left", va="top",
                fontsize=FS_LBL, color=INK, zorder=3)

    # ---------------- 分区 I：输入 ----------------
    section(1.0, 56.5, 99.0, 70.5, "I", "输入")
    box(7.0, 58.0, 40.0, 9.0, "题面文本", "清洗后、去题号标题")
    box(53.0, 58.0, 40.0, 9.0, "结构化特征", "算法标签 / 约束规模 / 提交统计")

    # ---------------- 分区 II：双通道编码 ----------------
    section(1.0, 35.0, 99.0, 53.5, "II", "双通道编码")
    box(7.0, 40.0, 40.0, 10.0, "BERT 编码器", "取 [CLS] 向量")
    box(53.0, 40.0, 40.0, 10.0, "全连接映射", "映射到同一维度")
    # 该刊读者为高职师生，用直觉说明代替公式（大纲三·（三））。
    # ⚠ 语义标签要**横向让开跨区连线**：折线的竖段在 x=27 与 x=73（框中心），
    # 标签若也居中就会被虚线穿过（实测两次）。这里左框标签右移到 x=37、
    # 右框标签左移到 x=63，各距竖段 10 个坐标单位（≈14 mm），足够容纳 7 个汉字。
    ax.text(37.0, 37.6, "反映“考什么”", ha="center", va="center",
            fontsize=FS_S, color=MUTE, zorder=3)
    ax.text(63.0, 37.6, "反映“考倒谁”", ha="center", va="center",
            fontsize=FS_S, color=MUTE, zorder=3)

    # ---------------- 分区 III：融合与回归 ----------------
    section(1.0, 3.0, 99.0, 32.0, "III", "融合与回归")
    # ★ 唯一强调色：融合环节（本文讨论的双通道拼接点）
    box(32.0, 20.5, 36.0, 9.0, "向量拼接", fc=ACCENT_FILL, ec=ACCENT,
        lw=0.9, tcolor=ACCENT)
    box(32.0, 10.5, 36.0, 8.0, "MLP 回归头")
    ax.text(50.0, 5.6, "输出：难度评分（rating）", ha="center", va="center",
            fontsize=FS_T, color=INK, zorder=3)

    # ---------------- 连线 ----------------
    varrow(27.0, 58.0, 50.0)          # 题面文本 → BERT
    varrow(73.0, 58.0, 50.0)          # 结构化特征 → 全连接
    elbow(27.0, 40.0, 41.0, 29.5, 34.0)   # BERT → 拼接（走两区之间空白带）
    elbow(73.0, 40.0, 59.0, 29.5, 34.0)   # 全连接 → 拼接
    varrow(50.0, 20.5, 18.5)          # 拼接 → MLP
    varrow(50.0, 10.5, 7.2)           # MLP → 输出

    return _save(fig, path)


# ================================================================ 图 2 主对比


def figure_main_comparison(path: Path, aggs_json: Path | str,
                           conditions: Sequence[str] = ("A", "B", "D"),
                           metric: str = "mae") -> list[Path]:
    """图 2（★ 核心图）：各设置下各模型 MAE 与 macro-F1 对比。

    规范 ② §7.1：y 轴标签是中文短语时用**横向条形图** `barh`。
    三个设置按"越重要越深"分配：A 最浅、B 中灰、**D 用唯一强调色**
    （D 是最真实的新题场景，也是本文主结论所在）。

    ⚠ 误差棒为**跨随机种子的标准差**，不是 95% CI——置信区间由配对 bootstrap
    单独报告，二者口径不同；该含义写进 Word 题注（规范 ② §8.1）。
    """
    import matplotlib.pyplot as plt

    setup_style()
    data = json.loads(Path(aggs_json).read_text(encoding="utf-8"))
    by: dict[tuple[str, str], dict] = {(d["model"], d["condition"]): d for d in data}

    models = [m for m in C.MAIN_TABLE_MODELS if (m, conditions[0]) in by]
    if not models:
        raise ValueError("聚合结果里找不到主表模型，无法绘图 2")

    # 按 D 格 MAE 排序（最优在上），使读者一眼看出位次
    def key(m: str) -> float:
        d = by.get((m, "D")) or by.get((m, conditions[0]))
        s = d["stats"].get("test_mae") if d else None
        return s["mean"] if s else 1e9
    models = sorted(models, key=key)

    colors = {conditions[0]: G1, conditions[-1]: D_ACCENT}
    for c in conditions[1:-1]:
        colors[c] = G2

    fig = plt.figure(figsize=(W_FULL * CM, 10.4 * CM))
    # ⚠ 图例放在**画幅顶部居中**（横向 3 列），不放坐标区内：
    # 横向条形图的最长条（标签均值基线）在最下方，图例无论放 lower left/right
    # 都会压住它，实测把数值标签糊成了乱码。
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.06,
                          left=0.155, right=0.985, top=0.885, bottom=0.110)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    y = np.arange(len(models))
    h = 0.72 / max(1, len(conditions))
    handles: list[Any] = []

    for ax, mname, xlabel in (
            (ax1, "mae", "平均绝对误差 / rating 点"),
            (ax2, "macro_f1", "宏观平均 $F_1$（无量纲）")):
        for ci, cond in enumerate(conditions):
            means, stds = [], []
            for m in models:
                d = by.get((m, cond))
                s = d["stats"].get(f"test_{mname}") if d else None
                means.append(s["mean"] if s else np.nan)
                stds.append(s["std"] if s else 0.0)
            pos = y + (ci - (len(conditions) - 1) / 2) * h
            # ⚠ `barh` 的误差棒样式仍走 `error_kw`（它转发给 bar），
            # 而 `errorbar` 在 matplotlib 3.11 已把该参数移除、改为直接传
            # `ecolor`/`elinewidth`/`capthick`。两者写法不同，别混用。
            bars = ax.barh(pos, means, height=h * 0.88, color=colors[cond],
                           edgecolor=INK, linewidth=0.45,
                           xerr=stds, capsize=1.8,
                           error_kw=dict(ecolor=INK, lw=LW_ERR, capthick=LW_ERR),
                           zorder=2)
            # ⚠ 图例句柄**只在第一个面板收一次**：两个面板都收的话，
            # handles 会变成 6 个而 labels 只有 3 个，matplotlib 按位置配对，
            # 结果是图例里出现两个"A 格"且颜色错位——D 格（强调色）整个消失（实测）。
            if ax is ax1:
                handles.append(bars)
        ax.set_xlabel(xlabel, fontsize=FS_AXIS, color=INK)
        _clean_axes(ax)

    ax1.set_yticks(y)
    ax1.set_yticklabels([model_label(m) for m in models], fontsize=FS_TICK)
    ax1.set_ylim(-0.6, len(models) - 0.4)
    ax1.invert_yaxis()
    # ⚠ 右侧面板**显式关掉** y 刻度标签：仅靠 `sharey=ax1` 在这里不生效
    # （实测两面板都印出了模型名，右侧那列纯属重复且挤压画幅）。
    ax2.tick_params(axis="y", labelleft=False, left=False)
    # 数值标签会伸到条形之外，故给 x 轴留出余量（否则最长的基线数值被裁）
    for ax, mname in ((ax1, "mae"), (ax2, "macro_f1")):
        vals = [by[(m, c)]["stats"][f"test_{mname}"]["mean"] + 
                by[(m, c)]["stats"][f"test_{mname}"]["std"]
                for m in models for c in conditions if (m, c) in by
                and f"test_{mname}" in by[(m, c)]["stats"]]
        if vals:
            ax.set_xlim(0, max(vals) * 1.14)
    # 数值只标关键点（规范 ② §8.2）：只标 D 格
    for yi, m in zip(y, models):
        d = by.get((m, "D"))
        s = d["stats"].get("test_mae") if d else None
        if not s:
            continue
        ax1.text(s["mean"] + s["std"] + 6, yi + (len(conditions) - 1) / 2 * h,
                 f"{s['mean']:.1f}", va="center", ha="left",
                 fontsize=FS_ANNO, color=INK, zorder=5)

    fig.legend(handles=handles, labels=[f"{c} 格" for c in conditions],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               fontsize=FS_LEG, handlelength=1.4, columnspacing=1.8,
               borderpad=0.15, frameon=False)
    # 面板编号放在**坐标区上沿之外**：放进坐标区内会压住长条（实测）
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.0, 1.012, lab, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=FS_LEG, color=INK)
    return _save(fig, path)


# ================================================================ 图 3 数据效率


def figure_data_efficiency(path: Path, curve_csv: Path | str,
                           models: Sequence[str] = ("lightgbm", "bert")) -> list[Path]:
    """图 3：数据效率曲线（10%/25%/50%/100%）。

    规范 ② §2.3 的"突出 1 条"手法：树模型用**强调色实线**（本文推荐方案），
    深度模型用**深灰虚线**（对照），并加标记区分——不靠颜色数量取胜。
    """
    import matplotlib.pyplot as plt

    setup_style()
    df = pd.read_csv(curve_csv, encoding="utf-8-sig")
    fig = plt.figure(figsize=(W_ONE * CM, 6.4 * CM))
    ax = fig.add_axes([0.175, 0.185, 0.800, 0.790])

    # ⚠ matplotlib 3.11 **移除了 `error_kw`**（实测签名里已无此参数，
    # 它会被转给 Line2D.set() 并抛 "unexpected keyword argument 'error_kw'"）。
    # 现在误差棒的样式参数直接与其它参数并列传入：`ecolor` / `elinewidth` /
    # `capsize` / `capthick`。这一改动是 3.10→3.11 的破坏性变更，
    # 旧写法在 3.9 及以前可用，因此不要照抄老例子。
    style = {
        "lightgbm": dict(color=D_ACCENT, ls="-", marker="o"),
        "bert": dict(color=G4, ls=(0, (5, 1.8)), marker="s"),
    }
    for m in models:
        sub = df[df["model"] == m].sort_values("train_fraction")
        if sub.empty:
            continue
        st = style.get(m, dict(color=G2, ls=(0, (1.6, 1.4)), marker="^"))
        ax.errorbar(sub["train_fraction"] * 100, sub["test_mae"],
                    yerr=sub.get("test_mae_std", 0),
                    linewidth=LW_DATA, markersize=3.4,
                    markerfacecolor="white", markeredgewidth=0.7,
                    label=model_label(m), zorder=3,
                    ecolor=INK, elinewidth=LW_ERR, capsize=1.8, capthick=LW_ERR,
                    **st)

    ax.set_xlabel("训练集使用比例 / %", fontsize=FS_AXIS, color=INK)
    ax.set_ylabel("测试集平均绝对误差 / rating 点", fontsize=FS_AXIS, color=INK)
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xlim(5, 105)
    _clean_axes(ax)
    ax.legend(fontsize=FS_LEG, loc="upper right", handlelength=1.8,
              labelspacing=0.32, borderpad=0.15)
    return _save(fig, path)


# ================================================================ 图 4 误差分析


def figure_error_analysis(path: Path, pred_npz: Path | str,
                          thresholds: tuple[float, ...] = C.DIFFICULTY_BINS,
                          tol: int = C.HIT_TOLERANCE) -> list[Path]:
    """图 4（双面板）：预测—真实散点图 + 分难度段 ±tol 命中率。

    规范 ② §8.3：分档区间底色用**浅灰 + alpha**，不用彩色块。
    """
    import matplotlib.pyplot as plt

    setup_style()
    z = np.load(pred_npz)
    y_true = z["y_test"]
    y_pred = z["test_pred"]
    edges = z["width_edges"] if "width_edges" in z.files else None

    fig = plt.figure(figsize=(W_FULL * CM, 6.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.24,
                          left=0.085, right=0.985, top=0.965, bottom=0.165)

    # ---- 面板 (a)：预测—真实散点 ----
    ax = fig.add_subplot(gs[0])
    lo = min(y_true.min(), y_pred.min()) - 150
    hi = max(y_true.max(), y_pred.max()) + 150
    # 分档区间底色（浅灰 + alpha，规范 §8.3）
    bounds = [-np.inf] + [float(e) for e in thresholds] + [np.inf]
    for i in range(len(bounds) - 1):
        a = max(bounds[i], lo) if np.isfinite(bounds[i]) else lo
        b = min(bounds[i + 1], hi) if np.isfinite(bounds[i + 1]) else hi
        if i % 2 == 0:
            ax.axvspan(a, b, color=G1, alpha=0.30, zorder=0)
    ax.scatter(y_true, y_pred, s=3.2, color=G3, alpha=0.55,
               edgecolors="none", zorder=2)
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.75,
            linestyle=(0, (4, 2)), zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实 rating / 点", fontsize=FS_AXIS, color=INK)
    ax.set_ylabel("预测 rating / 点", fontsize=FS_AXIS, color=INK)
    _clean_axes(ax)
    # 面板编号放**右下角**：左上/右上都会撞上 y 轴刻度或对角参考线（实测）。
    # 散点图右下角是空白区（高真实值被低估的点很少），不会遮数据。
    ax.text(0.985, 0.025, "(a)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FS_LEG, color=INK, zorder=5)

    # ---- 面板 (b)：分难度段命中率 ----
    ax = fig.add_subplot(gs[1])
    labels, rates, counts = [], [], []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        m = (y_true > a) & (y_true <= b)
        if not m.any():
            continue
        # ⚠ 区间标签必须分别处理两端：早期实现只判断了下界是否有限，
        # 于是最低档印成 "-inf~1200"、最高档印成 "≤inf"（把 np.inf 直接印出来）。
        # 最低档只有上界、最高档只有下界，中间档才是 "a~b"。
        if not np.isfinite(a):
            labels.append(f"≤{b:g}")
        elif not np.isfinite(b):
            labels.append(f">{a:g}")
        else:
            # 用连接号「–」而非波浪号「~」：中文排版里后者像"约等于"，
            # 而这里表达的是闭区间，用连接号更准确。
            labels.append(f"{a:g}–{b:g}")
        rates.append(float(np.mean(np.abs(y_true[m] - y_pred[m]) <= tol)))
        counts.append(int(m.sum()))

    pos = np.arange(len(labels))
    # 最低档（含 rating=800 下钳位）用中灰强调其"内部噪声最大"（规范 §2.3 灰阶层级）
    cols = [G2] + [G1] * (len(labels) - 1)
    ax.bar(pos, rates, width=0.62, color=cols, edgecolor=INK,
           linewidth=0.5, zorder=2)
    for p, r, n in zip(pos, rates, counts):
        ax.text(p, r + 0.022, f"{r:.2f}", ha="center", va="bottom",
                fontsize=FS_ANNO, color=INK, zorder=5)
        ax.text(p, 0.025, f"n={n}", ha="center", va="bottom",
                fontsize=FS_ANNO - 0.4, color=MUTE, zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("真实 rating 区间 / 点", fontsize=FS_AXIS, color=INK)
    ax.set_ylabel(f"±{tol} 命中率（无量纲）", fontsize=FS_AXIS, color=INK)
    ax.set_ylim(0, 1.12)
    _clean_axes(ax)
    ax.text(0.985, 0.975, "(b)", transform=ax.transAxes, ha="right",
            va="top", fontsize=FS_LEG, color=INK, zorder=5)
    return _save(fig, path)
