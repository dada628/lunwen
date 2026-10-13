"""重建原始数据快照（供 GitHub 仓库使用者复现本文结果）。

━━ 为什么不在仓库里放原始数据 ━━
`data/raw/statements.jsonl` 单文件 735 MB，超过 GitHub 单文件 100 MB 的硬限制；
`data/raw/` 合计约 1.5 GB（含 11 154 个题面 HTML 缓存），也不适合放进仓库。
故本脚本用 **Codeforces 公开 API** 按论文所标快照日期重建。

━━ 数据来源 ━━
  · 题目与 rating：https://codeforces.com/api/problemset.problems
  · 比赛时间：    https://codeforces.com/api/contest.list
  · 赛时统计：    https://codeforces.com/api/contest.standings
  这些接口**无需密钥**，但有速率限制（建议每次请求间隔 ≥2 秒）。

━━ 用法 ━━
  python download_data.py                    # 全量（约 11 000 题，耗时较长）
  python download_data.py --limit 200        # 只取前 200 题（快速自测）
  python download_data.py --skip-statements  # 不抓题面（只要元数据）

━━ 产出 ━━
  data/raw/contest_list.json      比赛列表
  data/raw/problemset.json        题目 + rating + 标签
  data/raw/statements.jsonl       题面（每行一条 JSON）
  data/raw/in_contest_stats.jsonl 赛时统计（S1′ 档所需）

⚠ 快照日期：本文使用 2026-10-06。Codeforces 的 rating 与通过人数**会随时间演化**，
   晚于该日期重建的数据与论文不完全一致——这是排行榜类数据的固有性质，
   论文正文已标注快照日期。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
API = "https://codeforces.com/api"

#: 论文所用的快照日期（元数据标注用；接口本身不返回历史快照）
SNAPSHOT_DATE = "2026-10-06"
#: 请求间隔（秒）——Codeforces 限流较严，过快会被 429
SLEEP = 2.0


def fetch(url: str, retries: int = 3) -> dict:
    """带重试的 GET（Codeforces 偶发 5xx / 429）。"""
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            wait = SLEEP * (k + 1) * 2
            print(f"    [重试 {k+1}/{retries}] {type(e).__name__}: {e}"
                  f" → 等 {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"请求失败：{url}")


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 Codeforces 数据快照")
    ap.add_argument("--limit", type=int, default=0,
                    help="只取前 N 题（0=全部）")
    ap.add_argument("--skip-statements", action="store_true",
                    help="不抓题面（只要元数据，快很多）")
    ap.add_argument("--out", default=str(RAW))
    a = ap.parse_args()
    raw = Path(a.out)
    raw.mkdir(parents=True, exist_ok=True)

    print(f"快照日期：{SNAPSHOT_DATE}（论文标注值）")
    print(f"输出目录：{raw}")
    print()

    # ---------- 1. 比赛列表 ----------
    print("[1/4] 拉取比赛列表 contest.list …")
    cl = fetch(f"{API}/contest.list")
    if cl.get("status") != "OK":
        print(f"  ❌ 接口返回 {cl.get('status')}: {cl.get('comment')}")
        return 1
    contests = cl["result"]
    (raw / "contest_list.json").write_text(
        json.dumps(contests, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ {len(contests)} 场比赛 → contest_list.json")
    time.sleep(SLEEP)

    # ---------- 2. 题目 + rating + 标签 ----------
    print("[2/4] 拉取题目与 rating problemset.problems …")
    ps = fetch(f"{API}/problemset.problems")
    if ps.get("status") != "OK":
        print(f"  ❌ 接口返回 {ps.get('status')}: {ps.get('comment')}")
        return 1
    probs = ps["result"]["problems"]
    if a.limit:
        probs = probs[:a.limit]
    (raw / "problemset.json").write_text(
        json.dumps(ps["result"], ensure_ascii=False), encoding="utf-8")
    with_rating = [p for p in probs if p.get("rating")]
    print(f"  ✅ 题目 {len(probs)} 道，其中带 rating 的 {len(with_rating)} 道"
          f" → problemset.json")

    # ---------- 3. 题面 ----------
    if a.skip_statements:
        print("[3/4] 跳过题面（--skip-statements）")
    else:
        print(f"[3/4] 抓取题面（{len(with_rating)} 道，每题约 {SLEEP:.0f}s，"
              f"预计 {len(with_rating)*SLEEP/60:.0f} 分钟）…")
        out = raw / "statements.jsonl"
        done = 0
        with out.open("w", encoding="utf-8") as fh:
            for i, p in enumerate(with_rating, 1):
                cid, idx = p["contestId"], p["index"]
                url = f"https://codeforces.com/problemset/problem/{cid}/{idx}"
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        html = r.read().decode("utf-8", errors="ignore")
                    rec = {"contest_id": cid, "index": idx, "url": url,
                           "html": html, "snapshot_date": SNAPSHOT_DATE}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done += 1
                except Exception as e:                       # noqa: BLE001
                    print(f"    ⚠ {cid}{idx} 失败：{type(e).__name__}")
                if i % 50 == 0:
                    print(f"    进度 {i}/{len(with_rating)}（成功 {done}）")
                time.sleep(SLEEP)
        print(f"  ✅ 题面 {done} 条 → statements.jsonl")

    # ---------- 4. 赛时统计 ----------
    print("[4/4] 赛时统计 contest.standings（S1′ 档所需）…")
    print("  ⚠ 该接口需逐场拉取，全量耗时很长；此处只示范前 5 场，")
    print("    完整实现见 src/task2difficulty/fetchers.py。")
    out = raw / "in_contest_stats.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for c in contests[:5]:
            if c.get("phase") != "FINISHED":
                continue
            try:
                st = fetch(f"{API}/contest.standings?contestId={c['id']}"
                           f"&from=1&count=1")
                fh.write(json.dumps({"contest_id": c["id"],
                                     "result": st.get("result")},
                                    ensure_ascii=False) + "\n")
                n += 1
            except Exception as e:                           # noqa: BLE001
                print(f"    ⚠ 比赛 {c['id']} 失败：{type(e).__name__}")
            time.sleep(SLEEP)
    print(f"  ✅ 示范 {n} 场 → in_contest_stats.jsonl")

    print()
    print("完成。下一步：")
    print("  python src/task2difficulty/../scripts/build_dataset.py   # 构建数据集")
    print("  python scripts/run_experiments.py                        # 复现实验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
