"""论文配图一键生成：结果文件 → 图 1～图 4（+ 表 1 配套的难度分布图）。

    python scripts/make_figures.py                  # 真实数据（默认）
    python scripts/make_figures.py --synthetic      # 合成数据自测结果
    python scripts/make_figures.py --out-dir outputs/figures

产出：每张图同时输出 PNG（600 dpi 预览）与 PDF（投稿矢量图），全部落在 --out-dir。

配图规范（两份规范见项目根目录）
--------------------------------
`src/task2difficulty/figures.py` 已按《配图规范 ① 架构图》《② 数据图》重写：
**图内无标题**（题注由 Word 文档提供，图题在下）、灰阶 + 唯一强调色、
直角几何 / 正交折线、无网格、刻度朝内、只留左/下轴脊、图例无框、
SimSun 字体、PNG 600 dpi + PDF 矢量嵌入。

⚠ 因此本脚本**不再需要**早期那套"运行时把 suptitle 挪到图下方"的补丁——
绘图函数里已经不存在任何标题。图内注释也一律移到 Word 题注中。

⚠ **宽度按本刊版心**：规范给的 16.0 cm 是照"版心 16.2～17.0 cm"定的，
而本刊模板（A4 + 左右 3.17 cm 页边距）版心为 **14.66 cm**，
故 `figures.W_FULL = 14.6`、`W_ONE = 9.0`（规范 §13 明确允许按期刊调整宽度）。

⚠ 缺数据不伪造：图 3 需要数据效率实验记录，没有 `data_efficiency.csv` 时跳过并打印
所需列名，绝不画假曲线。图 2 模型数不足时照实画并在日志里点名缺哪些模型。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import dataset as D         # noqa: E402
from task2difficulty import experiments as E     # noqa: E402
from task2difficulty import figures as F         # noqa: E402
from task2difficulty import results as R         # noqa: E402

#: 表 1 配套的难度分布图：大纲【五】的正式图只有 4 张，这张按"附图"编号，
#: 免得与"图 1 双通道融合模型结构图"抢编号（该刊要求尽量少用图表）。
SUPP_CAPTION = "附图 1  难度标签（rating）分布与分档切点"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def suffix_of(synthetic: bool) -> str:
    return "_synthetic" if synthetic else ""


def dataset_path(synthetic: bool) -> Path:
    return C.PROCESSED_DIR / ("dataset_synthetic.csv" if synthetic else "dataset.csv")


# ---------------------------------------------------------------- 图 2 数据


def load_runs(results_dir: Path, synthetic: bool) -> pd.DataFrame:
    """载入运行记录并剔除消融阶段（消融复用了条件键 D，会污染 D 格聚合）。"""
    runs = R.load_all_runs(results_dir, synthetic=synthetic)
    if "stage" in runs.columns:
        stage = runs["stage"].astype(str)
        keep = stage.str.startswith("factors") | stage.eq("size_control")
        dropped = int((~keep).sum())
        if dropped:
            log(f"  剔除消融阶段记录 {dropped} 行（复用条件键 D，会污染 D 格）")
        runs = runs[keep].copy()
    return runs


def main_conditions(aggs: dict) -> list[str]:
    """图 2 的设置轴：优先用 config 里"进主结论"的格子（A/B/D）。"""
    present = {k for _, k in aggs}
    prefer = [k for k, c in C.CONDITIONS.items() if c.in_main_conclusion and k in present]
    if prefer:
        return prefer
    return [k for k in C.CONDITIONS if k in present]


# ---------------------------------------------------------------- 图 3 数据


def data_efficiency_csv(results_dir: Path, synthetic: bool) -> Path:
    return results_dir / f"data_efficiency{suffix_of(synthetic)}.csv"


def _explain_missing_curve(path: Path) -> list[str]:
    """图 3 缺数据时的提示：把命令与列名格式写清楚，免得下一个人猜。"""
    fr = "/".join(f"{f:.2f}" for f in C.DATA_EFFICIENCY_FRACTIONS)
    return [
        f"未找到数据效率曲线 {path}，跳过图 3（不伪造曲线）。",
        f"需要先跑数据效率实验（大纲四·（六）实验 5）：按训练集比例 "
        f"{fr} 分别重训并记录测试集 MAE。",
        f"由 `python scripts/run_experiments.py --stage data_efficiency` 产出，"
        f"输出到 {path}。",
        f"CSV 列名必须为：model,train_fraction,test_mae,test_mae_std"
        f"（train_fraction 用小数 {fr}，不要写 10/25/50/100）。",
    ]


# ---------------------------------------------------------------- 图 4 数据


def pred_npz_path(condition: str, model: str, seed: int) -> Path:
    """复刻 `experiments.save_predictions` 的文件名规则。"""
    import re

    tag = f"{condition}_{model}_seed{seed}"
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    return E.PRED_DIR / f"{tag}.npz"


def pick_prediction(runs: pd.DataFrame, aggs: dict) -> tuple[Path, str] | None:
    """挑一份逐样本预测缓存：**最真实场景优先**（D > B > A > C），同格内取 MAE 最低的模型。"""
    order = ["D", "B", "A", "C"] + [k for k in C.CONDITIONS
                                    if k not in ("D", "B", "A", "C")]
    triples: dict[tuple[str, str], list[int]] = {}
    for _, r in runs[["model", "condition", "seed"]].drop_duplicates().iterrows():
        triples.setdefault((str(r["model"]), str(r["condition"])), []).append(int(r["seed"]))

    for cond in order:
        cands = [(m, s) for (m, c), seeds in triples.items() if c == cond
                 for s in sorted(seeds)]
        if not cands:
            continue

        def mae_of(ms: tuple[str, int]) -> float:
            m, _ = ms
            a = aggs.get((m, cond))
            if a is not None and "test_mae" in a.stats:
                return float(a.mean("mae"))
            sub = runs[(runs["model"] == m) & (runs["condition"] == cond)]
            return float(sub["test_mae"].mean()) if len(sub) else float("inf")

        def sort_key(ms: tuple[str, int]) -> tuple[float, int, int]:
            # 同格的多个种子 MAE 往往完全相同（确定性模型），必须再用「主表顺序 + 种子」
            # 兜底排序：否则 set 的遍历顺序随进程变化，同一命令会挑中不同种子（实测踩过）。
            m, s = ms
            idx = (list(C.MAIN_TABLE_MODELS).index(m)
                   if m in C.MAIN_TABLE_MODELS else len(C.MAIN_TABLE_MODELS))
            return (mae_of(ms), idx, s)

        for m, s in sorted(set(cands), key=sort_key):
            if E.load_predictions(cond, m, s) is not None:
                return (pred_npz_path(cond, m, s),
                        f"条件 {cond} 格 / {R.model_label(m)} / 种子 {s}")
    return None


# ---------------------------------------------------------------- 附图 1：难度分布


def figure_difficulty_distribution(path: Path, synthetic: bool) -> list[Path]:
    """表 1 配套图：rating 直方图 + 5 档等宽切点（规范 ② 数据图）。

    两个必须画出来的标签事实（大纲二·（三））：
      - rating 以 100 为步长 → 直方图按步长分箱；
      - 800 为下钳位 → 该档用中灰单独标出（灰阶层级，非彩色），
        它内部噪声最大，是回归 MAE 的系统性下限。
    """
    import matplotlib.pyplot as plt

    F.setup_style()
    ds = D.load_dataset(dataset_path(synthetic))
    y = np.asarray(ds.y, dtype=np.float64)
    lo, hi = int(np.floor(y.min())), int(np.ceil(y.max()))
    step = C.RATING_STEP
    edges = np.arange(lo - step / 2, hi + step, step)
    counts, _ = np.histogram(y, bins=edges)
    centers = edges[:-1] + step / 2

    fig = plt.figure(figsize=(F.W_FULL * F.CM, 5.6 * F.CM))
    ax = fig.add_axes([0.085, 0.185, 0.895, 0.735])

    # 下钳位档用中灰（G2），其余浅灰（G1）——灰阶层级，符合规范 §2.3
    cols = [F.G2 if c <= C.RATING_FLOOR else F.G1 for c in centers]
    ax.bar(centers, counts, width=step * 0.86, color=cols, edgecolor=F.D_INK,
           linewidth=0.45, zorder=2)
    for e in C.DIFFICULTY_BINS:
        ax.axvline(e, color=F.D_INK, linestyle=(0, (4, 2)), linewidth=0.75, zorder=3)
    ticks = sorted({lo, *[int(e) for e in C.DIFFICULTY_BINS], hi})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}" for t in ticks], fontsize=F.FS_TICK)

    # 5 档区间标签放在坐标区上方（与表 1 的分档行、图 4(b) 的刻度标签逐字一致）。
    # ⚠ 连接号用「–」而非「~」：与图 4(b) 统一，且「~」在中文里读作"约等于"，
    # 而这里表达的是闭区间。
    bounds = [-np.inf] + [float(e) for e in C.DIFFICULTY_BINS] + [np.inf]
    labels = [f"≤{C.DIFFICULTY_BINS[0]:g}"]
    labels += [f"{C.DIFFICULTY_BINS[i]:g}–{C.DIFFICULTY_BINS[i + 1]:g}"
               for i in range(len(C.DIFFICULTY_BINS) - 1)]
    labels += [f">{C.DIFFICULTY_BINS[-1]:g}"]
    for i, lab in enumerate(labels):
        a = max(bounds[i], lo - step / 2)
        b = min(bounds[i + 1], hi + step / 2)
        ax.text((a + b) / 2, 1.015, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=F.FS_ANNO, color=F.D_INK)

    ax.set_xlabel("题目难度 rating / 点", fontsize=F.FS_AXIS, color=F.D_INK)
    ax.set_ylabel("题目数 / 道", fontsize=F.FS_AXIS, color=F.D_INK)
    ax.set_ylim(0, max(counts) * 1.16)
    F._clean_axes(ax)
    return F._save(fig, path)


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成论文图 1～图 4（PNG 600dpi + 矢量 PDF）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="数据不足时按图跳过并在日志里说明需要先跑什么，不伪造任何曲线。",
    )
    ap.add_argument("--synthetic", action="store_true",
                    help="用合成数据自测结果（runs_synthetic.csv）")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认 outputs/figures")
    ap.add_argument("--results-dir", default=None, help="结果输入目录，默认 outputs/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else C.FIGURES_DIR
    results_dir = Path(args.results_dir) if args.results_dir else C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = suffix_of(args.synthetic)

    issues = C.validate()
    for i in issues:
        log(f"[口径自检失败] {i}")
    if issues:
        return 1

    log(f"模式：{'合成数据自测' if args.synthetic else '真实数据'}；"
        f"结果目录 {results_dir}；产出目录 {out_dir}")
    info = F.setup_style()
    log(f"  中文字体链：{info['cjk_fonts']}；通栏宽 {F.W_FULL} cm（本刊版心 14.66 cm）")
    produced: list[Path] = []
    skipped: list[str] = []

    # ---- 图 1：结构图（不依赖任何数据，永远能出）----
    log("图 1：双通道融合模型结构图")
    produced += F.figure_fusion_architecture(out_dir / f"fig1_fusion_arch{suf}.png")

    # ---- 运行记录（图 2/图 4 都要）----
    runs: pd.DataFrame | None = None
    aggs: dict = {}
    try:
        runs = load_runs(results_dir, args.synthetic)
        aggs = R.aggregate(runs)
        log(f"运行记录 {len(runs)} 行 → 聚合 {len(aggs)} 个（模型 × 条件）格")
    except Exception as ex:
        log(f"  [skip] 无法载入运行记录：{type(ex).__name__}: {ex}")
        skipped.append(f"图 2 / 图 4（无运行记录：{type(ex).__name__}）")

    # ---- 图 2（★ 核心图）----
    if aggs:
        log("图 2：各设置下各模型 MAE / macro-F1 对比")
        json_path = results_dir / f"aggs{suf}.json"
        R.dump_aggs(aggs, json_path)          # 图 2 走 json，保证与表格同一份聚合
        conds = main_conditions(aggs)
        n_models = len({m for m, _ in aggs})
        log(f"  聚合结果 → {json_path.name}；设置轴 {conds}；模型 {n_models} 个")
        if n_models < len(C.MAIN_TABLE_MODELS):
            missing = [R.model_label(m) for m in C.MAIN_TABLE_MODELS
                       if not any(k == m for k, _ in aggs)]
            log(f"  [warn] 数据不足以生成完整图表（{n_models}/"
                f"{len(C.MAIN_TABLE_MODELS)} 个主表模型），缺：" + "、".join(missing))
        try:
            produced += F.figure_main_comparison(
                out_dir / f"fig2_main{suf}.png", json_path, conditions=tuple(conds))
        except Exception as ex:
            skipped.append(f"图 2（{type(ex).__name__}: {ex}）")
            log(f"  [skip] 图 2 生成失败：{type(ex).__name__}: {ex}")

    # ---- 图 3：数据效率曲线 ----
    log("图 3：数据效率曲线（10%/25%/50%/100%）")
    curve = data_efficiency_csv(results_dir, args.synthetic)
    if not curve.exists():
        for line in _explain_missing_curve(curve):
            log("  " + line)
        skipped.append(f"图 3（缺 {curve.name}）")
    else:
        try:
            df_curve = pd.read_csv(curve, encoding="utf-8-sig")
            lack = {"model", "train_fraction", "test_mae"} - set(df_curve.columns)
            if lack:
                raise ValueError(f"{curve.name} 缺列 {sorted(lack)}；"
                                 f"应含 model,train_fraction,test_mae,test_mae_std")
            curve_models = list(dict.fromkeys(df_curve["model"].astype(str)))
            log(f"  用 {curve.name}（{len(df_curve)} 行，模型 {'、'.join(curve_models)}）")
            produced += F.figure_data_efficiency(
                out_dir / f"fig3_data_efficiency{suf}.png", curve, models=curve_models)
        except Exception as ex:
            skipped.append(f"图 3（{type(ex).__name__}: {ex}）")
            log(f"  [skip] 图 3 生成失败：{type(ex).__name__}: {ex}")

    # ---- 图 4：误差分析双面板 ----
    if runs is not None:
        log("图 4：误差分析双面板（预测—真实散点 + 分难度段命中率）")
        picked = pick_prediction(runs, aggs)
        if picked is None:
            log("  [skip] 未找到逐样本预测缓存；需先跑实验的 save_predictions，"
                f"缓存目录 {E.PRED_DIR}")
            skipped.append("图 4（无逐样本预测 npz）")
        else:
            npz, desc = picked
            log(f"  使用预测缓存 {npz.name}（{desc}）")
            try:
                produced += F.figure_error_analysis(
                    out_dir / f"fig4_error{suf}.png", npz)
            except Exception as ex:
                skipped.append(f"图 4（{type(ex).__name__}: {ex}）")
                log(f"  [skip] 图 4 生成失败：{type(ex).__name__}: {ex}")

    # ---- 附图 1：表 1 配套的难度分布图 ----
    log("附图 1：难度标签分布与分档切点（表 1 配套）")
    if not dataset_path(args.synthetic).exists():
        log(f"  [skip] 未找到 {dataset_path(args.synthetic)}")
        skipped.append("附图 1（无数据集）")
    else:
        try:
            produced += figure_difficulty_distribution(
                out_dir / f"figS1_difficulty_dist{suf}.png", args.synthetic)
        except Exception as ex:
            skipped.append(f"附图 1（{type(ex).__name__}: {ex}）")
            log(f"  [skip] 附图 1 生成失败：{type(ex).__name__}: {ex}")

    # ---- 产出清单 ----
    print("\n===== 产出清单（图，PNG 600dpi + 矢量 PDF） =====")
    for p in produced or []:
        print(p.resolve())
    if not produced:
        print("（无）")
    if skipped:
        print("\n===== 跳过项（数据不足，未伪造） =====")
        for s in skipped:
            print(f"- {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

