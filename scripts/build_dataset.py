"""Codeforces 数据管线：抓取 → 清洗 → 特征 → 建库 → 划分索引（大纲二·（一）（二）（五））。

产出
----
  data/processed/dataset.csv         清洗后的建模数据集（含快照日期）
  data/processed/split_index.csv     两套划分索引（所有模型共用）
  outputs/reports/数据管线报告.md      抓取统计、约束解析率、分布核对

用法
----
  # 小样本试跑（推荐先做，验证全链路再全量）
  python scripts/build_dataset.py --limit 300

  # 全量
  python scripts/build_dataset.py

  # 跳过抓取，仅用已有缓存重建（改了清洗/特征逻辑时用）
  python scripts/build_dataset.py --skip-crawl

可随时 Ctrl+C 中断；题面抓取写 jsonl，重跑会**跳过已完成**的题目（断点续爬）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import dataset as D          # noqa: E402
from task2difficulty import fetchers as F         # noqa: E402
from task2difficulty import vectorizers as V      # noqa: E402

CACHE_DIR = C.RAW_DIR / "cache"
STATEMENTS_JSONL = C.RAW_DIR / "statements.jsonl"
#: S1' 赛时统计（可选，由 build_s1_in_contest.py 产出）
S1_JSONL = C.RAW_DIR / "in_contest_stats.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================ 1. 元数据

def fetch_metadata(client: F.CodeforcesClient) -> pd.DataFrame:
    """`problemset.problems` + `contest.list` join。

    题面文本 API 不提供、题目发布日期 API 也不提供——**必须 join contest.list
    才能拿到 startTimeSeconds**，而时间划分完全依赖它。
    """
    log("抓取 problemset.problems")
    ps = client.fetch_problemset()
    problems = ps["result"]["problems"]
    stats = ps["result"]["problemStatistics"]
    log(f"  题目 {len(problems)} 条，统计 {len(stats)} 条")

    df_p = pd.DataFrame(problems)
    df_s = pd.DataFrame(stats)
    df = df_p.merge(df_s, on=["contestId", "index"], how="left", validate="one_to_one")

    log("抓取 contest.list（用于 join 获取比赛时间与 Div 类型）")
    contests = client.fetch_contest_list()
    df_c = pd.DataFrame(contests)[["id", "name", "startTimeSeconds", "phase", "type"]]
    df_c = df_c.rename(columns={"id": "contestId", "name": "contest_name",
                                "startTimeSeconds": "start_time", "type": "contest_type"})
    df = df.merge(df_c, on="contestId", how="left", validate="many_to_one")

    df["year"] = pd.to_datetime(df["start_time"], unit="s", utc=True,
                                errors="coerce").dt.year
    df["div_type"] = df["contest_name"].map(F.parse_div_type)
    df["problem_key"] = df["contestId"].astype(str) + "-" + df["index"].astype(str)
    return df


# ================================================================ 2. 抓题面

def crawl_statements(client: F.CodeforcesClient, keys: list[tuple[int, str]],
                     limit: int | None) -> None:
    log("抓取题面（HTML，官方 API 不提供；限速 + 重试 + 断点续爬）")
    t0 = time.time()
    n_ok = n_fail = 0
    for rec in F.iter_statements(client, keys, STATEMENTS_JSONL, max_items=limit):
        if rec.get("ok"):
            n_ok += 1
        else:
            n_fail += 1
    dt = time.time() - t0
    log(f"  完成：成功 {n_ok}，失败 {n_fail}，用时 {dt/60:.1f} 分钟")
    log(f"  HTTP 请求 {client.stats.requests}，缓存命中 {client.stats.cache_hits}，"
        f"重试 {client.stats.retries}，失败 {client.stats.failures}")


# ================================================================ 3. 组装数据集

def build_dataframe(df_meta: pd.DataFrame, statements: dict[str, dict],
                    snapshot: str, s1_stats: dict[str, dict] | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    """把元数据与题面合并成建模数据集，并生成全部特征列。

    `statements[key]` 需含 `raw_html`（只缓存原始 HTML，解析在此处进行），
    以便解析口径修正后无需重新抓取（见 fetchers.iter_statements 的说明）。

    `s1_stats[key]`（可选）：S1' 赛时统计。存在时写入 `*__s1` 后缀列，
    供"中间档 S1'"条件使用（它才是可回溯、无泄漏的那一档）。
    """
    rows: list[dict] = []
    diag = {
        "n_meta": int(len(df_meta)),
        "n_with_rating": 0,
        "n_with_year": 0,
        "n_with_statement": 0,
        "n_parse_ok": 0,
        "n_n_upper_found": 0,
        "n_n_upper_imputed": 0,
        "n_tag_columns": 0,
        "n_with_s1": 0,
        "template_hits": 0,
        "interactive": 0,
    }

    for r in df_meta.itertuples(index=False):
        if r.rating is None or (isinstance(r.rating, float) and np.isnan(r.rating)):
            continue
        if pd.isna(r.year) or pd.isna(r.start_time):
            # 拿不到比赛开始时间就无法做时间划分，也不满足"上线天数"的计算前提
            continue
        diag["n_with_rating"] += 1
        st = statements.get(r.problem_key)
        if not st or not st.get("ok"):
            continue
        diag["n_with_statement"] += 1

        # ---- 解析原始 HTML（缓存里只有 raw_html，解析在此处进行）----
        raw_html = st.get("raw_html") or ""
        if not raw_html:
            continue
        parts = F.extract_statement_parts(raw_html)

        # ---- 题面清洗（大纲二·（二））----
        # drop_samples=False：HTML 抽取阶段已按配平方式整块删掉 sample-tests/note，
        # 这里若再删一遍，正则的 `.*$` 会从正文中任何 "note" 处一路吃到文末。
        raw_text = "\n".join(x for x in (parts["preamble"], parts["input_spec"]) if x)
        cleaned = V.clean_statement(raw_text, drop_samples=False)
        if not cleaned:
            continue
        diag["n_parse_ok"] += 1

        # ---- 约束规模特征（冷启动核心）----
        cons = F.parse_constraints(parts["input_spec"] or parts["preamble"])
        n_upper = cons["constraint_n_upper"]
        imputed = 0
        if n_upper is None:
            # n 未在约束中显式给出（题面把范围写在散文里或只约束了 t）。
            # 用约束段的最大上界作代理，并**显式打标**以便做敏感性分析。
            n_upper = cons["constraint_max_upper"]
            imputed = 1 if n_upper is not None else 0
            if imputed:
                diag["n_n_upper_imputed"] += 1
        else:
            diag["n_n_upper_found"] += 1

        if V.count_template_hits(cleaned):
            diag["template_hits"] += 1
        if parts["is_interactive"] == "1":
            diag["interactive"] += 1

        # ---- 统计特征（新题不可得）----
        start = pd.Timestamp(r.start_time, unit="s", tz="UTC")
        snap_dt = pd.Timestamp(snapshot, tz="UTC")
        days_since = max(0.0, (snap_dt - start).total_seconds() / 86400.0)

        # ⚠ 标签在 API 里经 pandas 处理后是 **numpy.ndarray**，不是 list/tuple。
        # 只判断 isinstance(x, (list, tuple)) 会让 numpy 数组静默走进 else 分支，
        # 导致 tag_count 全为 0、multi-hot 全缺——且不报任何错（实测踩过）。
        try:
            tags = [str(t) for t in list(r.tags)]
        except TypeError:
            tags = []
        tags = [t for t in tags if t and t.lower() != "nan"]
        row = {
            "contest_id": int(r.contestId),
            # ⚠ 列名必须是 problem_index（而非 *_raw）：dataset.load_dataset 会按
            # "problem_index" 找来源特征并做 A→0、B→1 的有序编码。
            # 若改了名字，provenance 组会静默少一列（实测踩过）。
            "problem_index": str(r.index),
            "problem_key": r.problem_key,
            "name": str(r.name),
            "year": int(r.year),
            "start_time": int(r.start_time),
            "div_type": int(r.div_type),
            "contest_type": str(r.contest_type),
            "statement": cleaned,
            "statement_len": len(cleaned),
            # 统计特征（冷启动时整组被移除）
            "solved_count": float(r.solvedCount) if pd.notna(r.solvedCount) else np.nan,
            "days_since_publish": float(days_since),
            # 约束规模特征
            "constraint_n_upper": float(n_upper) if n_upper is not None else np.nan,
            "multi_test_count": int(cons["multi_test_count"]),
            "is_interactive": int(parts["is_interactive"]),
            # 标签特征
            "tag_count": len(tags),
            # 来源特征（高危，单列；config 里默认剔除）
            "contest_id_feat": int(r.contestId),
            # 元信息（供诊断，不进模型）
            "constraint_has_n": int(cons["constraint_has_n"]),
            "constraint_n_imputed": imputed,
            "constraint_hits": int(cons["constraint_hits"]),
            "constraint_max_upper": (float(cons["constraint_max_upper"])
                                     if cons["constraint_max_upper"] is not None else np.nan),
            "template_hit": int(V.count_template_hits(cleaned) > 0),
            "rating": int(r.rating),
            "snapshot_date": snapshot,
        }
        for t in tags:
            row[f"{C.TAG_MULTIHOT_PREFIX}{str(t).replace(' ', '_')}"] = 1

        # ---- S1' 赛时统计（可选）：写成 *__s1 列，供中间档条件使用 ----
        # 口径：这些是"比赛结束那一刻"的固定值，历史可回溯、赛后立即可得，
        # 既不是"全量累计"也不是"无数据"——正是大纲要的中间档。
        s1 = (s1_stats or {}).get(str(r.problem_key))
        if s1:
            diag["n_with_s1"] += 1
            row["solved_count__s1"] = float(s1.get("solved_in_contest") or 0)
            row["acceptance_rate__s1"] = float(s1.get("acceptance_rate") or 0.0)
            # 上线天数的赛时对应量：比赛当天即为 0 天，这里用 1 表示"刚上线"
            row["days_since_publish__s1"] = 1.0
            row["attempts_in_contest__s1"] = float(s1.get("attempts_in_contest") or 0)
            row["first_solve_seconds__s1"] = (
                float(s1["first_solve_seconds"])
                if s1.get("first_solve_seconds") is not None else np.nan
            )
        else:
            for col in ("solved_count__s1", "acceptance_rate__s1",
                        "days_since_publish__s1", "attempts_in_contest__s1",
                        "first_solve_seconds__s1"):
                row[col] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("数据集为空：请检查抓取是否成功")

    # multi-hot 的缺失位置补 0
    tag_cols = [c for c in df.columns if c.startswith(C.TAG_MULTIHOT_PREFIX)]
    df[tag_cols] = df[tag_cols].fillna(0).astype(int)

    # ---- 结构自检：这些失败会**静默产出坏数据**，必须在这里拦住 ----
    problems: list[str] = []
    if not tag_cols:
        problems.append(
            "没有任何 tag__* 列：算法标签特征整组缺失。"
            "常见原因是从 API 读回的 tags 是 numpy.ndarray 而类型判断只接受 list/tuple。"
        )
    if int(df["tag_count"].sum()) == 0:
        problems.append("tag_count 全为 0：标签未被读入。")
    for feat, group in (("problem_index", "来源特征"),):
        if feat not in df.columns:
            problems.append(f"{group} 缺少列 {feat}（列名须与 config 声明一致）")
    if df["constraint_n_upper"].isna().all():
        problems.append("constraint_n_upper 全为缺失：约束解析可能整体失效。")
    if problems:
        raise SystemExit("数据自检未通过：\n  - " + "\n  - ".join(problems))

    diag["n_tag_columns"] = len(tag_cols)
    return df, diag


# ================================================================ 4. 划分索引

def make_splits(df: pd.DataFrame) -> tuple[D.SplitIndex, list[str]]:
    si = D.make_split_index(df["year"], df["problem_key"])
    warns = D.check_split_ratios(si)
    return si, warns


# ================================================================ 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="Codeforces 数据管线")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅处理前 N 题（试跑用）；不填为全量")
    ap.add_argument("--skip-crawl", action="store_true",
                    help="跳过题面抓取，只用已有 jsonl 重建数据集")
    ap.add_argument("--api-interval", type=float, default=F.DEFAULT_API_INTERVAL_S)
    ap.add_argument("--html-interval", type=float, default=F.DEFAULT_HTML_INTERVAL_S)
    ap.add_argument("--snapshot", default=None,
                    help="数据快照日期 YYYY-MM-DD；默认取当前 UTC 日期")
    args = ap.parse_args()

    for d in (C.DATA_DIR, C.RAW_DIR, C.PROCESSED_DIR, C.RESULTS_DIR, C.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    snapshot = args.snapshot or date.today().isoformat()
    log(f"数据快照日期：{snapshot}（可复现性的关键，必须写入正文二·（一））")

    client = F.CodeforcesClient(
        CACHE_DIR, api_interval=args.api_interval,
        html_interval=args.html_interval, verbose=True,
    )

    # ---- 元数据（有磁盘缓存，重跑很快）----
    meta_cache = C.PROCESSED_DIR / "meta_joined.parquet"
    if meta_cache.exists():
        log(f"载入元数据缓存 {meta_cache.name}")
        df_meta = pd.read_parquet(meta_cache)
    else:
        df_meta = fetch_metadata(client)
        try:
            df_meta.to_parquet(meta_cache, index=False)
            log(f"  元数据已缓存 -> {meta_cache.name}")
        except Exception as e:
            log(f"  [warn] parquet 写入失败（{type(e).__name__}），跳过缓存")
    log(f"  元数据 {len(df_meta)} 题，含 rating {int(df_meta['rating'].notna().sum())} 题，"
        f"含比赛时间 {int(df_meta['start_time'].notna().sum())} 题")

    # ---- 待抓题面：只抓"有 rating 且能拿到比赛时间"的题 ----
    need = df_meta[df_meta["rating"].notna() & df_meta["start_time"].notna()].copy()
    need = need.sort_values(["year", "contestId", "index"]).reset_index(drop=True)
    if args.limit:
        # ⚠ 不能直接 head(N)：按年份排序后取前 N 会全是早年题目，
        # 导致时间划分的验证段（2022 年）为空，划分直接报错。
        # 改为**跨年份比例抽样**，保证三段都有样本，试跑结果才有代表性。
        per_year = max(1, args.limit // max(1, need["year"].nunique()))
        parts = []
        for _, g in need.groupby("year", sort=True):
            parts.append(g.head(per_year))
        need = pd.concat(parts, ignore_index=True)
        rng = np.random.default_rng(C.SPLIT_SEED)
        if len(need) > args.limit:
            keep = rng.choice(len(need), size=args.limit, replace=False)
            need = need.iloc[np.sort(keep)].reset_index(drop=True)
        log(f"  --limit 生效：跨年份抽样 {len(need)} 题，"
            f"年份跨度 {need['year'].min()}~{need['year'].max()}，"
            f"2022 年 {int((need['year'] == 2022).sum())} 题（验证集来源）")

    keys = list(zip(need["contestId"].astype(int), need["index"].astype(str)))
    if not args.skip_crawl:
        crawl_statements(client, keys, None)

    statements = F.load_statements(STATEMENTS_JSONL)
    log(f"载入题面记录 {len(statements)} 条")

    # ---- S1' 赛时统计（可选；未抓取时为 0 条，不影响主流程）----
    s1_stats = F.load_in_contest_stats(S1_JSONL)
    if s1_stats:
        log(f"载入 S1' 赛时统计 {len(s1_stats)} 条（可支撑中间档条件 S1'）")
    else:
        log("未发现 S1' 赛时统计（可选）；如需中间档请先运行 "
            "python scripts/build_s1_in_contest.py")

    # ---- 组装数据集 ----
    log("组装数据集与特征")
    df, diag = build_dataframe(need, statements, snapshot, s1_stats)
    log(f"  最终数据集 {len(df)} 题 × {df.shape[1]} 列")

    # ---- 划分索引 ----
    log("生成两套划分索引（时间划分为三段式）")
    si, warns = make_splits(df)
    D.save_split_index(si, C.SPLIT_INDEX_PATH)
    for kind in (C.SPLIT_RANDOM, C.SPLIT_TEMPORAL):
        r = si.ratios(kind)
        log(f"  {kind:9s} 训练 {r['train']:.1%} / 验证 {r['valid']:.1%} / 测试 {r['test']:.1%}")
    for w in warns:
        log(f"  [warn] {w}")

    # ---- 落盘 ----
    df.to_csv(C.DATASET_PATH, index=False, encoding="utf-8-sig")
    log(f"数据集 -> {C.DATASET_PATH}")

    # ---- 报告 ----
    write_report(df, diag, si, warns, snapshot, client, need)

    # ---- 自检：能否被下游正确读入 ----
    log("下游自检：用 load_dataset 读回数据集")
    try:
        ds = D.load_dataset(C.DATASET_PATH)
        log(f"  PASS：{len(ds)} 题，特征组 "
            f"{ {g: len(v) for g, v in ds.feature_groups.items()} }")
    except Exception as e:
        log(f"  FAIL：load_dataset 读回失败 {type(e).__name__}: {e}")
        return 2
    return 0


def write_report(df: pd.DataFrame, diag: dict, si: D.SplitIndex,
                 warns: list[str], snapshot: str,
                 client: F.CodeforcesClient, need: pd.DataFrame) -> None:
    rating = df["rating"]
    lines: list[str] = []
    A = lines.append
    A("# 数据管线报告\n")
    A(f"> 数据快照日期：**{snapshot}**（正文二·（一）必须注明）  ")
    A(f"> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

    A("## 一、抓取统计\n")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 元数据题目总数 | {diag['n_meta']} |")
    A(f"| 含 rating 且可关联比赛开始时间 | {diag['n_with_rating']} |")
    A(f"| 成功取得题面 | {diag['n_with_statement']} |")
    A(f"| **最终入库题目数** | **{len(df)}** |")
    A(f"| 题面抓取 HTTP 请求 | {client.stats.requests} |")
    A(f"| 缓存命中 | {client.stats.cache_hits} |")
    A(f"| 重试次数 | {client.stats.retries} |")
    A(f"| 最终失败 | {client.stats.failures} |")
    A("")

    A("## 二、约束解析质量（冷启动核心特征）\n")
    A("| 指标 | 数值 | 占比 |")
    A("|---|---|---|")
    n_tot = max(1, len(df))
    A(f"| 显式解析到 n 上界 | {diag['n_n_upper_found']} | {diag['n_n_upper_found']/n_tot:.1%} |")
    A(f"| 回退到约束段最大上界 | {diag['n_n_upper_imputed']} | {diag['n_n_upper_imputed']/n_tot:.1%} |")
    A(f"| 完全未解析到（缺失） | {int(df['constraint_n_upper'].isna().sum())} | "
      f"{df['constraint_n_upper'].isna().mean():.1%} |")
    A(f"| 题面含模板句 | {diag['template_hits']} | {diag['template_hits']/n_tot:.1%} |")
    A(f"| 交互题 | {diag['interactive']} | {diag['interactive']/n_tot:.1%} |")
    A("")
    A("> `constraint_n_upper` 缺失是正常现象：部分题面只约束组数 t，"
      "把 n 的范围写在散文里。`constraint_n_imputed` 列标记了回退取值，"
      "正文若使用该特征，应报告缺失率或做敏感性分析。\n")

    A("## 三、rating 分布（表 1 素材）\n")
    A("| 统计量 | 值 |")
    A("|---|---|")
    A(f"| 题目数 | {len(df)} |")
    A(f"| rating 范围 | {rating.min():.0f} ~ {rating.max():.0f} |")
    A(f"| 均值 / 标准差 | {rating.mean():.1f} / {rating.std():.1f} |")
    A(f"| **等于下限 800 的题目占比** | {(rating <= C.RATING_FLOOR).mean():.1%} |")
    A("")
    A("> 800 下钳位占比越大，最低档内部噪声越大——该档构成回归 MAE 的系统性下限，"
      "须写入误差分析（大纲二·（三）标签事实之一）。\n")
    A("按年份：\n")
    A("| 年份 | 题目数 |")
    A("|---|---|")
    for y, n in df["year"].value_counts().sort_index().items():
        A(f"| {y} | {n} |")
    A("")

    A("## 四、文本长度（决定 BERT 的 max_length）\n")
    A("| 统计量 | 值 |")
    A("|---|---|")
    A(f"| 字符均值 / 中位数 / p95 | {df['statement_len'].mean():.0f} / "
      f"{df['statement_len'].median():.0f} / {df['statement_len'].quantile(0.95):.0f} |")
    _split_len = df["statement"].str.split().str.len()
    A(f"| 词数均值 / 中位数 / p95 | {_split_len.mean():.0f} / "
      f"{_split_len.median():.0f} / {_split_len.quantile(0.95):.0f} |")
    A("")
    A("> ⚠ **口径说明（与表 1 必须一致）**：这里统计的是 `dataset.csv` 的 `statement` 列，"
      "内容为【题面正文 + Input 规范】合并后的清洗文本。"
      "注意它与【仅题面正文、不含 Input 规范】是不同口径，后者会短约 3 倍——"
      "表 1 与本报告使用同一列，两处数字必须一致，不可混用（实测踩过）。\n")

    # token 长度分布：BERT max_length 的直接依据，必须与上面同一口径
    try:
        _tok = V.BertTokenizerWrapper(model_name="bert-base-uncased", max_length=512)
        _sample = df["statement"].head(400).tolist()
        _tok.fit(_sample)
        _stats = V.choose_max_length(_sample, _tok._tok, cover=0.95, cap=512)
        A("| BERT token 统计量（抽样 400 题） | 值 |")
        A("|---|---|")
        A(f"| 均值 / 中位数 | {_stats['mean']:.0f} / {_stats['median']:.0f} |")
        A(f"| p90 / p95 / p99 | {_stats['p90']:.0f} / {_stats['p95']:.0f} / {_stats['p99']:.0f} |")
        A(f"| **覆盖 95% 所需 max_length** | **{int(_stats['chosen_max_length_for_95pct'])}** |")
        A("")
    except Exception as e:
        A(f"> [warn] token 统计跳过：{type(e).__name__}: {e}\n")

    A("## 五、标签 Top10\n")
    tag_cols = [c for c in df.columns if c.startswith(C.TAG_MULTIHOT_PREFIX)]
    if tag_cols:
        top = df[tag_cols].sum().sort_values(ascending=False).head(10)
        A("| 算法标签 | 题目数 |")
        A("|---|---|")
        for c, n in top.items():
            A(f"| {c[len(C.TAG_MULTIHOT_PREFIX):]} | {int(n)} |")
    A("")

    A("## 六、两套划分的三段规模（大纲四·（二）要求全报）\n")
    A("| 划分 | 训练 | 验证 | 测试 |")
    A("|---|---|---|---|")
    for kind in (C.SPLIT_RANDOM, C.SPLIT_TEMPORAL):
        r = si.ratios(kind)
        A(f"| {kind} | {int(round(r['train']*len(df)))} ({r['train']:.1%}) | "
          f"{int(round(r['valid']*len(df)))} ({r['valid']:.1%}) | "
          f"{int(round(r['test']*len(df)))} ({r['test']:.1%}) |")
    A("")
    if warns:
        A("**占比自检告警**（与大纲【十】实测值的差异）：\n")
        for w in warns:
            A(f"- {w}")
        A("")
    A("> 时间划分固定为 训练 ≤2021 / 验证 =2022 / 测试 ≥2023，**验证集按时间切**，"
      "分箱阈值同样只在该验证集上确定（大纲红线）。\n")

    out = C.REPORTS_DIR / "数据管线报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"报告 -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
