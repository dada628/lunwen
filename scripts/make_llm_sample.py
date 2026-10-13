"""生成四·（五）大模型零样本实验的**输入文件**（题面 + 标签 + 提示词）。

设计要点（每一条都对应正文里必须交代的口径）
------------------------------------------------
1. **可读题面**：给模型的不是 `dataset.csv` 里那份小写化、LaTeX 归一化过的
   训练用文本，而是由缓存原始网页重建的可读版本（保留大小写、`\\le`→`≤`、
   `\\frac{a}{b}`→`(a)/(b)`），并删掉题号标题——题号字母本身携带难度序，
   保留即构成泄漏。实现复用 `scripts/human_kappa.py` 的 `readable_statements`。
2. **只给题面 + 算法标签**，不给任何提交统计，与 D 格（时间划分 + 统计置空）
   的特征条件对齐；否则一致率与训练模型不可比。
3. **分层抽样**：按 rating 分低/中/高三层尽量等分，层内再按新题（≥2023）/
   旧题（<2023）分配。若纯随机抽，样本会被 1200–2000 的中间段主导，
   一致率只代表中间段共识，而本文关心的恰是两端。
4. **item_id 与难度无关**：抽完后整体打乱再编号（K001…K500）。
   若按难度排序编号，模型可以从编号里"读出"难度，整节结论作废。
5. **键文件与输入文件分离**：`items_key.csv` 含 problem_key / rating / year，
   是研究者留档，**不得**发给模型；`prompt_items*.jsonl` 只含 item_id、
   题面与标签，可以直接投喂。

用法
----
    python scripts/make_llm_sample.py --n-items 500 --seed 20260106
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task2difficulty import config as C  # noqa: E402

import human_kappa as HK  # noqa: E402  复用抽样与可读题面（同一份实现，避免两套口径）

OUT_DIR = C.OUTPUTS_DIR / "llm"


def build_prompt(item: dict, *, with_tags: bool = True, role: bool = False) -> str:
    """把一条题目渲染成完整提示词。

    ★ 三个开关对应主实验与两组消融，**除被消融的那一处外逐字相同**——
    提示词消融只有在"只改一个变量"时才可解释。
    """
    head = ("You are an experienced competitive programmer who has solved thousands "
            "of problems.\n\n" if role else "")
    tag_line = f"tags: {item['tags']}\n" if (with_tags and item.get("tags")) else ""
    return (
        f"{head}You are estimating how difficult a competitive programming problem is.\n\n"
        "Below is the statement of one problem from the Codeforces problemset"
        + (", together with its algorithm tags" if tag_line else "") + ".\n"
        "Estimate the difficulty rating that Codeforces would assign to this problem.\n\n"
        "The rating scale (integer, in steps of 100):\n\n"
        "  800-1200   very easy - basic implementation, no algorithm knowledge required\n"
        "  1300-1600  easy - simple greedy / sorting / math, straightforward once read\n"
        "  1700-2000  medium - one standard algorithm or data structure, or a non-obvious observation\n"
        "  2100-2400  hard - advanced algorithms, or several insights combined\n"
        "  2500-3500  very hard - solved in contest only by strong contestants\n\n"
        "Rules:\n"
        "1. Judge only from the text given below. Do not search for the problem, and do not\n"
        "   try to recall its identity or its official rating. If it looks familiar, ignore\n"
        "   that and base your answer on the content.\n"
        "2. Answer with an integer between 800 and 3500, in multiples of 100.\n"
        "   The minimum possible rating is 800.\n"
        "3. Never refuse and never ask questions. Always produce the required output.\n"
        "4. Output exactly one JSON object on one line, and nothing else - no explanation\n"
        "   before or after it.\n\n"
        '{"item_id": "<the id given below>", "rating": <integer>, '
        '"confidence": <number between 0 and 1>, "reason": "<at most 25 words>"}\n\n'
        "--- PROBLEM ---\n"
        f"item_id: {item['item_id']}\n"
        f"{tag_line}"
        "statement:\n"
        f"{item['statement']}\n"
        "--- END ---\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 LLM 零样本实验的输入文件")
    ap.add_argument("--n-items", type=int, default=500, help="总题量（大纲上限 500）")
    ap.add_argument("--seed", type=int, default=20260106)
    ap.add_argument("--new-frac", type=float, default=0.4,
                    help="每层中新题（year ≥ 2023）的目标比例")
    ap.add_argument("--dataset", default=str(C.DATASET_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) 读数据并重建可读题面（复用 human_kappa，含"取不到就回退并告警"）----
    frame = HK.load_working_frame(args.dataset)
    if args.n_items > len(frame):
        raise SystemExit(f"需要 {args.n_items} 题，数据集只有 {len(frame)} 题")

    # ---- 2) 分层 × 新旧 配额 ----
    labels = HK.stratum_labels()
    caps = [int((frame["stratum"] == i).sum()) for i in range(len(labels))]
    quotas = HK.allocate_even(args.n_items, caps)
    picks: list[pd.DataFrame] = []
    for s in range(len(labels)):
        sub = frame[frame["stratum"] == s]
        want_new, want_old = HK.split_new_old(
            int(quotas[s]), int(sub["is_new"].sum()),
            int(len(sub) - sub["is_new"].sum()), float(args.new_frac))
        for era_i, want in enumerate((want_new, want_old)):
            if want == 0:
                continue
            pool = sub[sub["is_new"] == (1 - era_i)]
            if len(pool) < want:
                raise SystemExit(f"分层 {labels[s]} 的题目不足：需要 {want}，只有 {len(pool)}")
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, s, era_i]))
            idx = np.sort(rng.choice(len(pool), size=want, replace=False))
            picks.append(pool.iloc[idx])
        print(f"  层 {labels[s]}: 抽 {quotas[s]} 题（新 {want_new} / 旧 {want_old}）")

    sample = pd.concat(picks, ignore_index=True)
    if len(sample) != args.n_items or sample["problem_key"].duplicated().any():
        raise SystemExit("抽样数量不符或出现重复题目")

    # ---- 3) 打乱后编号：item_id 不得携带任何难度信息 ----
    order = np.random.default_rng(
        np.random.SeedSequence([args.seed, 0x5DEECE66D])).permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample["item_id"] = [f"K{i + 1:04d}" for i in range(len(sample))]

    # ---- 4) 键文件（研究者留档，不可发给模型）----
    key = sample[["item_id", "problem_key", "rating", "year", "is_new", "tags"]].copy()
    key.to_csv(out / "items_key.csv", index=False, encoding="utf-8-sig")

    # ---- 5) 输入文件（可直接投喂）----
    items = sample[["item_id", "statement", "tags"]].to_dict("records")
    with (out / "prompt_items.jsonl").open("w", encoding="utf-8") as f:      # 主实验：题面+标签
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "prompt_items_no_tags.jsonl").open("w", encoding="utf-8") as f:  # 消融①
        for it in items:
            f.write(json.dumps({"item_id": it["item_id"], "statement": it["statement"]},
                               ensure_ascii=False) + "\n")

    # ---- 6) 三种提示词各存一份样例，便于肉眼核对 ----
    variants = {
        "prompt_baseline.txt": dict(with_tags=True, role=False),
        "prompt_no_tags.txt": dict(with_tags=False, role=False),
        "prompt_role.txt": dict(with_tags=True, role=True),
    }
    for fn, kw in variants.items():
        (out / fn).write_text(build_prompt(items[0], **kw), encoding="utf-8")

    # ---- 7) 抽样说明 ----
    meta = {
        "snapshot_date": str(frame.attrs.get("snapshot_date", "")),
        "n_items": int(len(sample)),
        "seed": int(args.seed),
        "strata": labels,
        "by_stratum": {lab: int((sample["stratum"] == i).sum())
                       for i, lab in enumerate(labels)},
        "n_new": int(sample["is_new"].sum()),
        "n_old": int((1 - sample["is_new"]).sum()),
        "rating_range": [float(sample["rating"].min()), float(sample["rating"].max())],
        "readable_statement_fallback": int(len(frame) - frame.attrs.get("n_readable_statement", 0)),
        "disclaimer": "items_key.csv 含真实 rating 与年份，不得随题面一起交给模型",
    }
    (out / "sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 -> {out}")
    print(f"  题量 {len(sample)}（新 {meta['n_new']} / 旧 {meta['n_old']}），"
          f"rating {meta['rating_range'][0]:.0f}–{meta['rating_range'][1]:.0f}")
    print("  items_key.csv            键文件（含真实 rating，**不得发给模型**）")
    print("  prompt_items.jsonl       主实验输入（题面 + 算法标签）")
    print("  prompt_items_no_tags.jsonl  消融①输入（去了标签）")
    print("  prompt_*.txt             三种提示词样例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
