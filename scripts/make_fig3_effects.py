"""生成图3：因子效应分解（表5 的可视化版本）。

设计依据（两份配图规范）：
  · 数据图 → 横条图（y 轴标签是中文短语时用 barh）
  · 灰阶 + 单一强调色；只留左/下轴脊；无网格；刻度朝内；图例无框
  · 误差量已由"跨种子相减后一次舍入"给出，故无误差棒
  · 变量名与单位用斜体/正体规范写法

数据源：outputs/results/table5_effects.csv（与文档表5 同源）
  列为 A−B / B−D / B′−D / C−D / A−C
  Δ = 左格MAE − 右格MAE，**负值表示左侧更优**

语义分组（按正文表5注②）：
  · A−B   统计特征的总边际贡献（含同源成分）
  · B−D   跨时间泛化损失（含训练量差异）
  · B′−D  剔除训练量差异后的时间效应
  · C−D   被泄漏放大，仅作上界
  · A−C   同等统计可得性下的跨时间损失
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import figures as F          # noqa: E402

import numpy as np                                # noqa: E402


#: 五列的展示名与语义（顺序固定）。
#: ⚠ 连接号用 **U+2013 EN DASH（–）**，不用 U+2212 MINUS SIGN（−）：
#:   `figures.py` 已设 `axes.unicode_minus=False`，且 SimSun **缺 U+2212 字形**
#:   （实测 `simsun.ttc` 的 cmap 无 0x2212）——matplotlib 不做逐字符回退，
#:   会渲染成缺字方框。已刊图 `fig4_error` 用的也是 U+2013，此处与之一致。
COLS = [
    ("A–B", "统计特征边际贡献"),
    ("B–D", "跨时间退化"),
    ("B′–D", "剔除训练量后"),
    ("C–D", "含泄漏上界"),
    ("A–C", "同等可得性下"),
]


def load(csv_path: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    head = [h.strip() for h in rows[0]]
    models, data = [], []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        models.append(r[0].strip())
        vals = []
        for x in r[1:]:
            x = x.strip()
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(np.nan)      # "—" 表示该模型未参与
        data.append(vals)
    return models, np.array(data, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成图3：因子效应分解")
    ap.add_argument("--csv", default=str(ROOT / "outputs/results/table5_effects.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/figures"))
    a = ap.parse_args()

    import matplotlib.pyplot as plt

    info = F.setup_style()
    models, D = load(Path(a.csv))

    #: 按 A−B 的绝对值排序（贡献最大的在上），与正文表5 行序一致
    order = np.argsort(-np.abs(D[:, 0]))
    models = [models[i] for i in order]
    D = D[order]

    n_m, n_c = D.shape
    fig = plt.figure(figsize=(F.W_FULL * F.CM, 9.6 * F.CM))
    #: ⚠ 上边距留 0.19 给顶部图例（5 列横排），否则图例被画布裁掉（实测踩过）。
    ax = fig.add_axes([0.150, 0.145, 0.830, 0.665])

    y = np.arange(n_m)
    h = 0.74 / n_c
    #: 灰阶阶梯 + 唯一强调色给时间效应那两列（本图重点）。
    palette = [F.G3, F.D_ACCENT, F.D_ACCENT, F.G2, F.G1]
    handles = []
    for ci, (col, sem) in enumerate(COLS):
        vals = D[:, ci]
        pos = y + (ci - (n_c - 1) / 2) * h
        #: ⚠ **必须区分"未参与"与"真 0"**：CSV 里 "—" 读成 NaN，
        #:   若用 `np.nan_to_num` 当 0 画，会与真实 0.0（如标签均值基线的 A–B=0.0）
        #:   完全同形，读者无法分辨该模型是否参与了这一对比（实测踩过）。
        #:   处置：NaN 的格**不画条**（留空白），并在图注里说明。
        draw = np.isfinite(vals)
        b = ax.barh(pos[draw], vals[draw], height=h * 0.86,
                    color=palette[ci], edgecolor=F.INK, linewidth=0.45,
                    zorder=2, label=col)
        handles.append(b)

    ax.axvline(0, color=F.INK, linewidth=0.8, zorder=3)

    #: ⚠ 必须显式设 x 轴范围并留出标签空间：
    #:   多数 Δ 为负（左侧更优），但「A–C」等列可能为正（右侧更差）。
    #:   若只按数据自动定界，正值一侧会被贴到轴脊上、条与边框重叠。
    vmin = np.nanmin(D)
    vmax = np.nanmax(D)
    span = max(abs(vmin), abs(vmax)) or 1.0
    ax.set_xlim(-span * 1.06, max(span * 0.30, vmax * 1.12))   # 正侧留余量

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=F.FS_TICK, color=F.INK)
    ax.invert_yaxis()                    # 最优/贡献最大者在上
    ax.set_xlabel("Δ ＝ 左格 MAE – 右格 MAE / rating 点",
                  fontsize=F.FS_AXIS, color=F.INK)
    F._clean_axes(ax)

    #: 图例只放对比名（语义写进图注），5 个一行排开；
    #: `ncol=3` 会按列填充，读起来是 A–B/B–D → B′–D/C–D → A–C，顺序错乱（实测）。
    ax.legend(handles, [c for c, _ in COLS],
              loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=len(COLS), fontsize=F.FS_LEG, handlelength=1.3,
              columnspacing=2.2, handletextpad=0.5)

    out = Path(a.out_dir)
    #: 文件名用 `fig3_effect_decomp` 而非 `fig3_effect_decomposition`：
    #: `outputs/figures/` 下已有 `fig3_data_efficiency`（数据效率曲线，未入稿），
    #: 两个 "fig3_" 前缀并存容易混，短名更清晰。
    paths = F._save(fig, out / "fig3_effect_decomp.png")
    print("图3 已生成：")
    for p in paths:
        print(f"  {p}  {p.stat().st_size} bytes")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
