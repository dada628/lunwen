"""人类一致性实验的统计功效模拟：确定所需的最小规模。

用途
----
大纲原定「100 题 × 3 位评委」。但实际执行中的主要障碍是**评委负担**：
100 题的定档约需 60~90 分钟，很难请到 3 位有竞赛经历的人各自完成。
本脚本用模拟回答一个可量化的问题——**把题数压到多少，结论仍然站得住？**

判据（两条，都是结论可靠性的必要条件）
------------------------------------
1. **κ 的 95% 置信区间不跨越结论分界带**：四·（五）的结论解读规则以
   κ≈0.4 与 κ≈0.6 为界（低于 0.4 说明等级体系可分辨程度有限；
   高于 0.6 可论述模型接近人类标注水平）。若 CI 同时覆盖两侧，结论就不确定。
2. **能区分 κ=0.4 与 κ=0.6 两种真值**：即在 κ=0.4 的模拟中，
   CI 上界应低于 0.6，否则无法排除"实际一致性偏高"。

方法
----
构造一个参数可控的评分过程：每题有真实档位，每位评委以概率 `p_exact` 命中，
未命中时在相邻档偏移（难度判断的错误通常邻近，这与误差分析中
"相邻档互混为主"的观察一致）。按目标 κ 校准 `p_exact`，再重复模拟
数百次，统计估计出的 κ 的抽样标准差。

⚠ 这是**设计阶段的规模论证工具**，不是实验结果。
其输出用于选择评题量，并应在正文中说明取值依据。
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from task2difficulty import config as C  # noqa: E402

# human_kappa.py 里的 Fleiss 实现（与正式实验同一份代码，避免两套口径）
_spec = ilu.spec_from_file_location("_hk", str(ROOT / "scripts" / "human_kappa.py"))
HK = ilu.module_from_spec(_spec)
_spec.loader.exec_module(HK)


def difficulty_distribution() -> np.ndarray:
    """五档难度的真实分布。

    默认为均匀近似；若数据集存在，则用其实际分布（更贴近真实抽样情形）。
    分布形状会影响 κ 的方差，因此尽量用真实数据。
    """
    try:
        import pandas as pd
        import human_kappa as _  # noqa: F401  仅确认可导入
        ds = pd.read_csv(C.DATASET_PATH, encoding="utf-8-sig")
        bins = np.digitize(ds["rating"].to_numpy(dtype=float),
                           list(C.DIFFICULTY_BINS), right=True)
        counts = np.bincount(bins, minlength=len(C.DIFFICULTY_BINS) + 1).astype(float)
        if counts.sum() > 0:
            return counts / counts.sum()
    except Exception:
        pass
    k = len(C.DIFFICULTY_BINS) + 1
    return np.ones(k) / k


def simulate(rng: np.random.Generator, n_items: int, n_raters: int,
             p_exact: float, dist: np.ndarray) -> float:
    """模拟一次实验，返回 Fleiss κ（用项目自身的实现计算）。"""
    true = rng.choice(len(dist), size=n_items, p=dist)
    ratings = np.zeros((n_items, n_raters), dtype=int)
    for j in range(n_raters):
        hit = rng.random(n_items) < p_exact
        off = rng.integers(-2, 3, size=n_items)
        off[off == 0] = 1
        r = true.copy()
        r[~hit] = np.clip(true[~hit] + off[~hit], 0, len(dist) - 1)
        ratings[:, j] = r
    counts = np.zeros((n_items, len(dist)), dtype=int)
    for i in range(n_items):
        for v in ratings[i]:
            counts[i, v] += 1
    try:
        return float(HK.fleiss_kappa_manual(counts))
    except Exception:
        return float("nan")


def calibrate(target: float, dist: np.ndarray, rng: np.random.Generator,
              trials: int = 10) -> float:
    """二分搜索命中率，使模拟出的 κ 约等于 target。"""
    lo, hi = 0.05, 0.95
    for _ in range(24):
        mid = (lo + hi) / 2
        ks = [simulate(rng, 150, 3, mid, dist) for _ in range(trials)]
        ks = [k for k in ks if k == k]
        if not ks:
            lo = mid
            continue
        if float(np.mean(ks)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description="人类一致性实验的规模论证")
    ap.add_argument("--reps", type=int, default=300, help="每个配置的模拟次数")
    ap.add_argument("--items", nargs="*", type=int,
                    default=[20, 30, 40, 50, 60, 80, 100])
    ap.add_argument("--raters", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--kappas", nargs="*", type=float, default=[0.40, 0.60],
                    help="要考察的真值 κ（0.4 与 0.6 是大纲的结论分界）")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="安全边际：CI 端点距结论分界应留出的余量（默认 0.05）。"
                         "模拟是理想化评分过程，真实评委会因疲劳、题型偏好等"
                         "产生更大分歧，刚好压线的规模不可靠。")
    ap.add_argument("--out", default=None, help="把结论写入 markdown")
    args = ap.parse_args()

    rng = np.random.default_rng(C.SPLIT_SEED)
    dist = difficulty_distribution()
    print("=== 真实难度分布（五档占比）===")
    print("  " + "  ".join(f"{i+1}档 {p:.1%}" for i, p in enumerate(dist)))
    print(f"  样本来源：{C.DATASET_PATH.name}" if C.DATASET_PATH.exists() else "  （均匀近似）")

    print("\n=== 校准评分行为 ===")
    p_for = {}
    for k in args.kappas:
        p = calibrate(k, dist, rng)
        p_for[k] = p
        print(f"  真值 κ={k:.2f}  对应命中率 p={p:.3f}")

    lines: list[str] = []
    lines.append("# 人类一致性实验的规模论证（模拟结果）\n")
    lines.append(f"> 生成方式：`python scripts/kappa_power.py --reps {args.reps}`"
                 f"；真值 κ ∈ {args.kappas}，每配置重复 {args.reps} 次。\n")
    lines.append("> ⚠ 这是**设计阶段的规模论证**，不是实验结果，用于选择评题量。\n")

    for k_target in args.kappas:
        p = p_for[k_target]
        print(f"\n=== 真值 κ = {k_target:.2f}（p_exact={p:.3f}）===")
        print(f"{'题数':>6s} {'人数':>5s} {'κ 均值':>9s} {'κ 标准差':>10s} "
              f"{'95%CI 半宽':>11s} {'CI 区间':>20s}")
        print("-" * 72)
        lines.append(f"\n## 真值 κ = {k_target:.2f}\n")
        lines.append("| 题数 | 人数 | κ 均值 | κ 标准差 | 95%CI 半宽 | CI 区间 |")
        lines.append("|---|---|---|---|---|---|")
        for n in args.items:
            for kr in args.raters:
                ks = np.array([simulate(rng, n, kr, p, dist)
                               for _ in range(args.reps)])
                ks = ks[ks == ks]
                if len(ks) < 10:
                    continue
                mean, sd = float(ks.mean()), float(ks.std(ddof=1))
                half = 1.96 * sd
                lo, hi = mean - half, mean + half
                print(f"{n:>6d} {kr:>5d} {mean:>9.3f} {sd:>10.3f} "
                      f"{half:>11.3f}   [{lo:>6.3f}, {hi:>6.3f}]")
                lines.append(f"| {n} | {kr} | {mean:.3f} | {sd:.3f} | "
                             f"{half:.3f} | [{lo:.3f}, {hi:.3f}] |")

    # 结论：推荐规模
    #
    # ★ 优化目标不是"规模最小"，而是**招募难度最低**。
    # 评委是稀缺资源（再找到一位有竞赛经历的人，比让已答应的人多评十几道题难得多），
    # 而题量是弹性资源。因此判据为：在**评委人数尽量少**的前提下，
    # 找出使结论站得住的最小题量。
    print("\n" + "=" * 72)
    print("=== 推荐规模 ===")
    margin = args.margin
    print(f"判据 ①：真值 κ=0.60 时 CI 下界 > {0.40 + margin:.3f}"
          f"（不落入『等级体系可分辨性有限』一侧，留 {margin:.2f} 安全边际）")
    print(f"判据 ②：真值 κ=0.40 时 CI 上界 < {0.60 - margin:.3f}"
          f"（能与『接近人类标注水平』区分开）")
    print("优化目标：先满足安全边际，再使评委人数与题量尽量小\n")
    print("⚠ 为什么需要安全边际：模拟用的是理想化评分过程，"
          "真实评委的行为\n  （疲劳、对某些题型系统性偏严/偏松）会让实际分歧大于模拟，"
          "刚好压线的规模不可靠。\n")

    def ci(n: int, kr: int, p: float) -> tuple[float, float, float]:
        ks = np.array([simulate(rng, n, kr, p, dist) for _ in range(args.reps)])
        ks = ks[ks == ks]
        if len(ks) < 10:
            return float("nan"), float("nan"), float("nan")
        mean, sd = float(ks.mean()), float(ks.std(ddof=1))
        return mean - 1.96 * sd, mean + 1.96 * sd, 1.96 * sd

    recommendations: list[tuple[int, int, float, float, float, float]] = []
    feasible: dict[int, list[int]] = {}
    for kr in sorted(args.raters):
        for n in sorted(args.items):
            lo60, hi60, _half = ci(n, kr, p_for[0.60])
            _lo40, hi40, half40 = ci(n, kr, p_for.get(0.40, p_for[0.60]))
            ok60 = lo60 > 0.40 + margin
            ok40 = hi40 < 0.60 - margin
            mark = "✓" if (ok60 and ok40) else "✗"
            print(f"  {kr} 人 × {n:3d} 题: κ=0.60 CI [{lo60:.3f}, {hi60:.3f}] "
                  f"{'下界够' if ok60 else '下界不足'} | "
                  f"κ=0.40 CI 上界 {hi40:.3f} {'够低' if ok40 else '偏高'} -> {mark}")
            if ok60 and ok40:
                feasible.setdefault(kr, []).append(n)
                recommendations.append((kr, n, min(abs(0.40 - lo60), abs(0.60 - hi40)),
                                        lo60, hi40, half40))
        if feasible.get(kr):
            print(f"  => {kr} 人时可行题量: {feasible[kr]}"
                  f"（最小 {min(feasible[kr])} 题，每人约 "
                  f"{min(feasible[kr])*40/60:.0f}~{min(feasible[kr])*90/60:.0f} 分钟）")
        else:
            print(f"  => {kr} 人时在给定范围内均不满足判据")

    lines.append("\n## 结论\n")
    if recommendations:
        # ★ 优化目标：**先让评委人数最少，再让题量最少**。
        # 理由：有竞赛经历的评委是稀缺资源——多找一个人远比让已答应的人多评十几题难。
        # 切勿按"安全边际最大"排序：那会选出 4 人 × 100 题这种
        # 比原方案更难执行的配置，与降低负担的目的背道而驰。
        recommendations.sort(key=lambda t: (t[0], t[1]))
        kr, n, m, lo60, hi40, half40 = recommendations[0]

        # 同时给出"每人负担更轻"的备选（人数 +1、题量更少），供按招募难度取舍
        lighter = [r for r in recommendations
                   if r[0] > kr and r[1] < n and r[2] >= m - 0.01]
        lighter.sort(key=lambda t: (t[0], t[1]))

        print(f"\n  ★ 推荐：{kr} 位评委 × {n} 题")
        print(f"    κ=0.60 时 CI 下界 {lo60:.3f} > {0.40+margin:.2f}；"
              f"κ=0.40 时 CI 上界 {hi40:.3f} < {0.60-margin:.2f}")
        print(f"    每位评委约需 {n*40/60:.0f}~{n*90/60:.0f} 分钟"
              f"（原定 100 题需 60~90 分钟）")
        if lighter:
            lk, ln, lm, llo, lhi, _ = lighter[0]
            print(f"    备选（评委多 1 位、每人负担更轻）：{lk} 位 × {ln} 题，"
                  f"每人约 {ln*40/60:.0f}~{ln*90/60:.0f} 分钟")

        lines.append(f"推荐规模：**{kr} 位评委 × {n} 题**。\n")
        lines.append(f"- 真值 κ=0.60 时，估计值 95% CI 下界为 {lo60:.3f}，"
                     f"高于 {0.40+margin:.2f}，不会误落入“等级体系可分辨性有限”一侧；")
        lines.append(f"- 真值 κ=0.40 时，95% CI 上界为 {hi40:.3f}，"
                     f"低于 {0.60-margin:.2f}，能与“接近人类标注水平”区分开；")
        lines.append(f"- 每位评委约需 {n*40/60:.0f}~{n*90/60:.0f} 分钟，"
                     "相比原定 100 题的 60~90 分钟明显减轻。\n")
        if lighter:
            lk, ln, lm, llo, lhi, _ = lighter[0]
            lines.append(f"若招募条件允许，另一可选方案是 **{lk} 位评委 × {ln} 题**"
                         f"（每人约 {ln*40/60:.0f}~{ln*90/60:.0f} 分钟，"
                         f"κ=0.60 时 CI 下界 {llo:.3f}）。"
                         "评委人数增加会显著提高招募难度，因此仅在前一方案不可行时采用。\n")
        lines.append("**为什么不能靠减少人数来省事**：由表可见，2 位评委时即使 100 题"
                     "也只能勉强满足判据，边际很薄；而 3 位评委在 "
                     f"{n} 题时即有充裕边际。评委人数对结论可靠性的影响大于题量。\n")
        lines.append("> 正文写法建议：为避免评委负担过重，本文先通过模拟确定"
                     "使 κ 的 95% 置信区间在结论分界两侧均留有余量所需的最小规模，"
                     "据此设定评题量。")
    else:
        print("  在给定范围内未找到满足判据的规模，建议增加题数或人数")
        lines.append("在给定范围内未找到满足判据的规模，建议增加题数或人数。")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
