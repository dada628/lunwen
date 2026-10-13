"""把 LLM 实验改成**批处理**：一次请求出多题，500 题只需 2 次请求。

⚠ 这个改动会削弱四·（五）原本的独立性假设，必须在正文如实交代（见文末"代价"）。
本脚本只负责按新协议生成任务，判据不因省调用而改变。

协议（每个模型各跑一遍）
------------------------
    B1  baseline  前 250 题   repeat=1    主实验前半
    B2  baseline  后 250 题   repeat=1    主实验后半
    B3  no_tags   前 200 题   repeat=1    消融①（与 B1 的前 200 题同题同序）
    B4  role      前 200 题   repeat=1    消融②（同上）
    B5  baseline  前 250 题   repeat=2    采样噪声基线
    B6  baseline  后 250 题   repeat=2    采样噪声基线
    => 6 次请求/模型 × 5 模型 = 30 次

用法
----
    python scripts/make_llm_batches.py                    # 默认 250 题/批（500 题 = 2 批）
    python scripts/make_llm_batches.py --per-batch 125    # 上下文吃紧时改小（500 题 = 4 批）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C  # noqa: E402

LLM_DIR = C.OUTPUTS_DIR / "llm"
ARCHIVE = LLM_DIR / "archive"

N_ABLATION = 200
NOISE_REPEAT = 2

BATCH_TEMPLATE = """\
{role_line}You are estimating how difficult competitive programming problems are.

Below are {n_items} problems from the Codeforces problemset. For each problem, estimate the difficulty rating that Codeforces would assign to it.

The rating scale (integer, in steps of 100):

  800-1200   very easy - basic implementation, no algorithm knowledge required
  1300-1600  easy - simple greedy / sorting / math, straightforward once read
  1700-2000  medium - one standard algorithm or data structure, or a non-obvious observation
  2100-2400  hard - advanced algorithms, or several insights combined
  2500-3500  very hard - solved in contest only by strong contestants

Rules:
1. Judge each problem independently, from its own statement alone. Do not compare the
   problems with one another, and do not let an answer for one problem influence
   another.
2. Do not search for the problems, and do not try to recall their identities or their
   official ratings. If a problem looks familiar, ignore that and judge it on content.
3. Every rating must be an integer between 800 and 3500, in multiples of 100.
   The minimum possible rating is 800.
4. Output exactly one JSON object per problem, one object per line, in the same order
   as the problems appear below, and nothing else - no headings, no counts, no
   commentary:
   {"item_id": "<the id>", "rating": <integer>}
5. Produce a line for every problem. Do not skip any problem, do not summarize,
   do not refuse.

=== PROBLEMS ===
{problems_block}=== END OF PROBLEMS ===
"""

ROLE_LINE = ("You are an experienced competitive programmer who has solved "
             "thousands of problems.\n\n")


def block(it: dict, *, with_tags: bool) -> str:
    tag_line = f"tags: {it.get('tags', '')}\n" if (with_tags and it.get("tags")) else ""
    return (f"item_id: {it['item_id']}\n"
            f"{tag_line}"
            f"statement:\n{it['statement']}\n"
            f"=== END {it['item_id']} ===\n")


def render(items: list[dict], *, variant: str) -> str:
    blocks = "".join(block(it, with_tags=(variant != "no_tags")) for it in items)
    return (BATCH_TEMPLATE
            .replace("{role_line}", ROLE_LINE if variant == "role" else "")
            .replace("{n_items}", str(len(items)))
            .replace("{problems_block}", blocks))


def main() -> int:
    ap = argparse.ArgumentParser(description="生成批处理的 LLM 实验任务包")
    ap.add_argument("--per-batch", type=int, default=250,
                    help="每批题数（默认 250，即 500 题 = 2 次请求）")
    args = ap.parse_args()

    src = LLM_DIR / "items.jsonl"
    if not src.exists():
        fallback = ARCHIVE / "prompt_items.jsonl"
        if not fallback.exists():
            raise SystemExit("缺少题面数据；请先运行 scripts/make_llm_sample.py")
        src.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")
    items = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    n = len(items)
    k = args.per_batch

    batches: list[dict] = []

    def add(batch_id: str, variant: str, subset: list[dict], repeat: int) -> None:
        batches.append({
            "batch_id": batch_id,
            "variant": variant,
            "repeat": repeat,
            "n_items": len(subset),
            "item_ids": [it["item_id"] for it in subset],
            "prompt": render(subset, variant=variant),
        })

    # 主实验：按 --per-batch 切批
    for i in range(0, n, k):
        part = items[i:i + k]
        add(f"B{len(batches) + 1}-baseline-r1", "baseline", part, 1)
    # 采样噪声基线：把主实验的每一批原样再跑一遍
    for b in list(batches):
        ids = set(b["item_ids"])
        subset = [it for it in items if it["item_id"] in ids]
        add(f"{b['batch_id'].split('-')[0]}-baseline-r{NOISE_REPEAT}",
            "baseline", subset, NOISE_REPEAT)
    # 两组消融：固定的前 200 题，题序一致，只改一处；同样按 --per-batch 切块，
    # 保证与主实验的"每请求题数"完全一致（否则消融与主实验的批大小不同，
    # 内锚定强度不同，对比里就混进了协议差异）。
    for prefix, variant in (("A1", "no_tags"), ("A2", "role")):
        sub = items[:N_ABLATION]
        for j in range(0, len(sub), k):
            add(f"{prefix}-{variant}-r1-{j // k + 1}", variant, sub[j:j + k], 1)

    (LLM_DIR / "prompt_batch.txt").write_text(BATCH_TEMPLATE, encoding="utf-8")
    with (LLM_DIR / "llm_batches.jsonl").open("w", encoding="utf-8") as f:
        for b in batches:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    # 旧的逐题任务表归档，保证工作目录里只有一份 jsonl
    for name in ("llm_tasks.jsonl", "prompt.txt"):
        p = LLM_DIR / name
        if p.exists():
            ARCHIVE.mkdir(exist_ok=True)
            p.replace(ARCHIVE / name)

    print(f"批处理任务包 -> {LLM_DIR}")
    print(f"  prompt_batch.txt   唯一提示词模板（占位符 role_line / n_items / problems_block）")
    print(f"  llm_batches.jsonl  {len(batches)} 个批次，每批 {k} 题以内")
    for b in batches:
        print(f"      {b['batch_id']:<20s} {b['variant']:<9s} {b['n_items']:>4d} 题  "
              f"prompt {len(b['prompt'])/1024:>6.0f} KB  "
              f"≈{len(b['prompt'])/4/1000:>5.0f}k token")
    print(f"  每个模型跑完这一份 = {len(batches)} 次请求；5 个模型共 "
          f"{len(batches) * 5} 次（原来的逐题方案是 5500 次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
