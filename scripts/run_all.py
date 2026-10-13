"""全量运行编排器：一条命令跑完"建库 → 验证 → 调参 → 全部实验 → 出表出图"。

为什么需要它
------------
全部实验涉及 9 个阶段 × 10 个模型 × 多随机种子，手动逐条执行容易漏阶段、
也容易在中断后重复跑已完成的部分（BERT 单次微调实测约 3.7 分钟，
全链路 GPU 约 6 小时）。本编排器把整条链路串起来，并支持 **--from 从失败处续跑**：
已完成的阶段会被检测并跳过，超参文件已存在时不会重调。

顺序不可颠倒
------------
超参必须先在 A 格调好，B/C/D/S1' 才能复用；因子设计又依赖建库产出的数据集与划分索引。
**verify 必须紧跟 build**：完整实验要跑数小时，若数据本身有问题
（时间划分占比失真、某格样本量不足、S1' 列静默泄漏），把几小时压在坏数据上代价很大；
verify 在分钟级内完成关键检查并给出通过/不通过，不通过即中止后续阶段。

固定顺序为：
    build → verify → tune → factors → s1 → size_control
          → data_efficiency → ablation → tables → figures

用法
----
    python scripts/run_all.py                      # 全量、默认 5 个种子
    python scripts/run_all.py --to verify           # 抓取后只做重建+冒烟验证
    python scripts/run_all.py --from tune           # 验证已通过，从调参续跑
    python scripts/run_all.py --seeds 42 1337 2024  # 指定种子（默认取 config.SEEDS[:5]）
    python scripts/run_all.py --skip-build          # 数据集已最新，跳过建库
    python scripts/run_all.py --fast                # 快速档：3 种子 + max_length 192
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402

#: 阶段顺序（不可颠倒：见模块 docstring）
STAGES: tuple[str, ...] = (
    "build", "verify", "tune", "factors", "s1", "size_control",
    "data_efficiency", "ablation", "tables", "figures",
)

#: verify 是判定性阶段：不通过即中止整条链（其余阶段失败只是停在那里等续跑）
GATING_STAGES: tuple[str, ...] = ("verify",)


def log(msg: str) -> None:
    print(f"\n{'='*72}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'='*72}", flush=True)


def run(cmd: list[str]) -> int:
    """执行子命令并实时透传输出（不捕获，否则长任务看不到进度）。"""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def preflight_build() -> tuple[bool, str]:
    """建库前置检查：抓取是否已完成、是否有另一个进程正在写。

    ★ 为什么必须检查：`build_dataset.py` 与独立运行的抓取进程**写同一个**
    `data/raw/statements.jsonl`。若两者并发，追加写入会交错，可能产出半行 JSON，
    而 jsonl 的读取方（`fetchers.load_statements`）会**静默跳过**解析失败的行——
    表现为"抓了 11152 题但数据集只有 9000 题"，且不报错。
    因此这里同时检查两件事：完成度，以及文件是否正在被写入（mtime 很近）。
    """
    st = C.RAW_DIR / "statements.jsonl"
    if not st.exists():
        return True, "无题面缓存，build 将执行完整抓取（耗时数小时）"
    n = 0
    with st.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    age = time.time() - st.stat().st_mtime
    if age < 60:
        return False, (
            f"检测到 {st.name} 在 {age:.0f} 秒前仍在写入，说明有另一个抓取进程在运行。\n"
            f"    两个进程并发写同一文件会造成 JSON 行交错，读取方会**静默丢弃**这些行，\n"
            f"    导致数据集题数与抓取数不符且不报错。\n"
            f"    请等抓取任务结束后再运行本编排器（或在别的任务里确认它已退出）。"
        )
    meta = C.PROCESSED_DIR / "meta_joined.parquet"
    if meta.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(meta)
            need = int((df["rating"].notna() & df["start_time"].notna()).sum())
            if n < need:
                return False, (
                    f"题面缓存 {n} 条 < 需要 {need} 条，抓取尚未完成。\n"
                    f"    直接建库会得到残缺数据集（时间划分的验证/测试段可能近乎为空）。\n"
                    f"    请先补完抓取，或确认这是有意的部分数据实验。"
                )
            return True, f"题面缓存 {n} 条，覆盖所需 {need} 条（抓取已完成）"
        except Exception as e:
            return True, f"元数据读取异常（{type(e).__name__}），跳过完成度检查"
    return True, f"题面缓存 {n} 条（无元数据可比对，跳过完成度检查）"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="全量实验编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"阶段顺序：{' → '.join(STAGES)}")
    ap.add_argument("--from", dest="start", default="build", choices=STAGES,
                    help="从哪个阶段开始（之前的阶段跳过）")
    ap.add_argument("--to", dest="end", default="figures", choices=STAGES,
                    help="到哪个阶段结束（如 --to verify 只做重建与验证）")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help=f"随机种子；默认取 config.SEEDS[:5] = {C.SEEDS[:5]}")
    ap.add_argument("--models", nargs="*", default=None,
                    help="仅跑指定模型；默认全部主表模型")
    ap.add_argument("--skip-build", action="store_true",
                    help="数据集与划分索引已是最新时跳过建库")
    ap.add_argument("--force-tune", action="store_true",
                    help="强制重调超参（会覆盖已有 hyperparams_*.json）")
    ap.add_argument("--fast", action="store_true",
                    help="快速档：种子减为 3 个（大纲下限），GPU 时间约减少四成")
    args = ap.parse_args()

    py = sys.executable
    seeds = list(args.seeds) if args.seeds else list(C.SEEDS[:5])
    if args.fast and not args.seeds:
        seeds = list(C.SEEDS[:3])
    models = args.models if args.models else list(C.MAIN_TABLE_MODELS)

    si = STAGES.index(args.start)
    ei = STAGES.index(args.end)
    if si > ei:
        raise SystemExit(f"--from ({args.start}) 不能晚于 --to ({args.end})")
    plan = STAGES[si:ei + 1]

    issues = C.validate()
    if issues:
        log("口径自检未通过，已中止：")
        for i in issues:
            print("  ✗", i)
        return 1

    log(f"编排计划：{' → '.join(plan)}")
    log(f"模型 {len(models)} 个：{' '.join(models)}")
    log(f"种子 {len(seeds)} 个：{seeds}")
    if args.fast:
        print("  [fast] 种子已减为 3 个（大纲下限 3~5，合规）。"
              "BERT 的 max_length 保持 256——它由题面 token 分布决定（p95=174），"
              "不宜为省时间而降低，否则会截断部分题面并影响文本侧对比的公平性。")
    # 大纲要求深度模型 3~5 个种子；超出范围时提示，不静默接受
    if not (C.MIN_SEEDS <= len(seeds) <= 5):
        print(f"  [warn] 种子数 {len(seeds)} 不在大纲要求的 "
              f"{C.MIN_SEEDS}~5 之间，请在正文中说明理由")

    t_all = time.time()
    for stage in plan:
        if stage == "build":
            if args.skip_build:
                log("跳过 build（--skip-build）")
                continue
            ok, why = preflight_build()
            if not ok:
                log("✗ build 前置检查未通过，已中止：")
                print(f"    {why}")
                return 2
            log(f"阶段 build：数据管线（清洗/特征/建库/划分索引）")
            print(f"  前置检查：{why}")
            # ⚠ 这里**必须**带 --skip-crawl：题面抓取由独立的长任务负责，
            # 建库只做"用已有缓存重建"，不重复发起上万次网络请求。
            # 重复抓取既慢又可能与被并发的抓取进程冲突（见 preflight_build）。
            rc = run([py, "scripts/build_dataset.py", "--skip-crawl"])
        elif stage == "verify":
            log("阶段 verify：全量重建后的轻量验证（分钟级；不通过即中止）")
            rc = run([py, "scripts/run_experiments.py", "--stage", "verify"])
        elif stage == "tune":
            log("阶段 tune：**只在 A 格调参**（之后 B/C/D/S1' 复用，不再调）")
            cmd = [py, "scripts/run_experiments.py", "--stage", "tune",
                   "--models", *models]
            if args.force_tune:
                cmd.append("--force")
            rc = run(cmd)
        elif stage == "factors":
            log("阶段 factors：A/B/C/D 因子设计（主表 + 表 4）")
            rc = run([py, "scripts/run_experiments.py", "--stage", "factors",
                      "--models", *models, "--seeds", *map(str, seeds)])
        elif stage == "s1":
            log("阶段 s1：因子设计第五格 S1'（时间划分 + 赛时统计）")
            rc = run([py, "scripts/run_experiments.py", "--stage", "s1",
                      "--models", *models, "--seeds", *map(str, seeds)])
        elif stage == "size_control":
            log("阶段 size_control：B′ 规模对照（按**实测**比例对齐训练规模）")
            rc = run([py, "scripts/run_experiments.py", "--stage", "size_control",
                      "--seeds", *map(str, seeds)])
        elif stage == "data_efficiency":
            log("阶段 data_efficiency：实验 5 数据效率曲线（图 3 的数据来源）")
            rc = run([py, "scripts/run_experiments.py", "--stage", "data_efficiency"])
        elif stage == "ablation":
            log("阶段 ablation：provenance 消融（结构端 + 文本端）与弱基线自曝")
            rc = run([py, "scripts/run_experiments.py", "--stage", "ablation",
                      "--seeds", *map(str, seeds)])
        elif stage == "tables":
            log("阶段 tables：生成论文表 1～表 4 + 正文回填文本")
            rc = run([py, "scripts/make_tables.py"])
            if rc == 0:
                rc = run([py, "scripts/make_writeups.py"])
        elif stage == "figures":
            log("阶段 figures：生成论文图 1～图 4 + 表 1 配套分布图")
            rc = run([py, "scripts/make_figures.py"])
        else:
            raise AssertionError(stage)

        if rc != 0:
            if stage in GATING_STAGES:
                log(f"✗ 阶段 {stage} 未通过（exit={rc}）。**已中止后续阶段**——"
                    f"请先修复数据问题，再重跑 `python scripts/run_all.py "
                    f"--from {stage}`。切勿直接把完整实验压在有问题的数据上。")
            else:
                log(f"阶段 {stage} 失败（exit={rc}）。修复后可用 "
                    f"`python scripts/run_all.py --from {stage}` 从此处续跑。")
            return rc

    log(f"全部完成，总用时 {(time.time()-t_all)/60:.1f} 分钟")
    print("\n产出清单：")
    for d, pat in ((C.RESULTS_DIR, "*.csv"), (C.FIGURES_DIR, "*.png")):
        for p in sorted(d.glob(pat)):
            print(f"  {p.relative_to(ROOT)}")
    print("\n下一步（需人工/离线完成）：")
    print("  1. LLM 零样本实验：需各家 API 密钥与训练截止时间，编排脚本尚未实现")
    print("     （人类标注实验已按用户要求从论文中删去，不做、不写）")
    print("  2. 正文四·（二）(三) 的数字取 outputs/results/正文回填.md，"
          "勿手工抄写（限定语会漏）")
    print("  3. 按深度学习论文大纲三.md 成文；表 3/表 4 取 outputs/results/*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
