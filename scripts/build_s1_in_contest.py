"""S1' 数据管线：抓 contest.standings 重建赛时统计（大纲二·（七）可选中间态）。

S1' 的价值：官方 API 只有"当前累计"统计、无历史接口，因此时间划分下
测试集新题若用全量快照，属于"用最终统计预测最终标签"的乐观泄漏。
S1' 用**比赛结束时的榜单**重建统计——它是**历史固定、赛后可立即获得**的，
既非"全量累计"也非"无数据"，正好补上"新题→存量"的中间档。

产出：data/raw/in_contest_stats.jsonl（每行一个 题目×赛时统计）
用法：
    python scripts/build_s1_in_contest.py --limit 20   # 试跑
    python scripts/build_s1_in_contest.py              # 全量（约 5 小时）
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
from task2difficulty import fetchers as F         # noqa: E402

CACHE_DIR = C.RAW_DIR / "cache"
S1_JSONL = C.RAW_DIR / "in_contest_stats.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rated_contest_ids() -> list[int]:
    """取出"含 rating 题目"的比赛场次——S1' 只需覆盖这些场次。

    从已缓存的元数据读，避免重复请求；没有缓存则先跑 build_dataset.py。
    """
    meta = C.PROCESSED_DIR / "meta_joined.parquet"
    if not meta.exists():
        raise SystemExit(
            f"未找到元数据缓存 {meta}。请先运行 python scripts/build_dataset.py（会缓存元数据）。"
        )
    df = pd.read_parquet(meta)
    need = df[df["rating"].notna() & df["start_time"].notna()]
    ids = sorted({int(c) for c in need["contestId"].unique()})
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="S1' 赛时统计抓取")
    ap.add_argument("--limit", type=int, default=None, help="仅抓前 N 场（试跑）")
    ap.add_argument("--api-interval", type=float, default=F.DEFAULT_API_INTERVAL_S,
                    help=f"API 间隔秒；官方下限 {F.OFFICIAL_MIN_INTERVAL_S}s")
    args = ap.parse_args()

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = F.CodeforcesClient(CACHE_DIR, api_interval=args.api_interval, verbose=True)

    ids = rated_contest_ids()
    done = 0
    if S1_JSONL.exists():
        seen = set()
        with S1_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        import json
                        seen.add(int(json.loads(line)["contest_id"]))
                    except Exception:
                        pass
        done = len(seen)
    log(f"需覆盖比赛 {len(ids)} 场（大纲【十】记为 1976 场）；已完成 {done} 场")
    est_h = (len(ids) - done) * 9.7 / 3600
    log(f"按单场中位 9.7 s 估算，剩余约 {est_h:.1f} 小时；可随时中断，重跑自动续")

    t0 = time.time()
    n_rec = n_fail_contests = 0
    for rec in F.crawl_in_contest_stats(client, ids, S1_JSONL, max_items=args.limit):
        n_rec += 1
    # 统计失败场次
    try:
        import json
        with S1_JSONL.open("r", encoding="utf-8") as f:
            n_fail_contests = sum(1 for ln in f if ln.strip() and '"ok": false' in ln
                                  or ln.strip() and '"ok":false' in ln)
    except Exception:
        pass

    dt = time.time() - t0
    log(f"完成：写入 {n_rec} 条题目级赛时统计，失败场次 {n_fail_contests}，"
        f"用时 {dt/60:.1f} 分钟")
    log(f"HTTP 请求 {client.stats.requests}，重试 {client.stats.retries}，"
        f"最终失败 {client.stats.failures}")
    log(f"产出 -> {S1_JSONL}")

    stats = F.load_in_contest_stats(S1_JSONL)
    log(f"可读回 {len(stats)} 条；示例：")
    for k in list(stats)[:3]:
        r = stats[k]
        log(f"    {k}: 赛时通过 {r['solved_in_contest']}，通过率 "
            f"{float(r['acceptance_rate']):.3f}，首解 {r['first_solve_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
