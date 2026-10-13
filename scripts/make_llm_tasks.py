"""生成四·（五）LLM 实验的**统一任务包**：一段提示词 + 一份 jsonl + 一个输出文件。

为什么要改成这个结构
--------------------
早先的做法是"三个模板 × 两份 jsonl"（主实验 / 无标签 / 角色扮演各一套），
跑起来要切换文件、容易串味。现在改为：

- **一段提示词**：`prompt.txt`，含 `{role_line}` `{tags_line}` `{item_id}` `{statement}`
  四个占位符；三个变体**只差数据行里填不填那两行**，提示词本身逐字不变。
  这比"改句子"的消融更干净——单变量原则要求除被消融的那一处外一切相同。
- **一份 jsonl**：`llm_tasks.jsonl`，一行一个**任务**（题目 × 变体 × 重复），
  每行自带填好占位符后的完整提示词 `prompt` 字段；调用方既可以自己替换模板，
  也可以直接把 `prompt` 发出去，两种用法都不会跑偏。
- **一个输出文件**：`llm_results_<模型>.jsonl`（同目录），字段见 `实验操作说明.md`。

⚠ 为什么不做成"一次调用让模型同时给三个变体的答案"：那样主实验的答案会先出现在
上下文里，"无标签"的那一答已经不是无标签条件——标签已经被模型看到了。
消融的整个意义就在于对比"给没给"这一处差异，折叠成一次调用会直接废掉这个对比。

用法
----
    python scripts/make_llm_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C  # noqa: E402

LLM_DIR = C.OUTPUTS_DIR / "llm"
ARCHIVE = LLM_DIR / "archive"

#: 唯一的提示词模板。`{role_line}` 与 `{tags_line}` 为空串时不留空行，
#: 因此"给/不给"这两处只体现为**多一行或少一行**，句法完全不变。
PROMPT_TEMPLATE = """\
{role_line}You are estimating how difficult a competitive programming problem is.

Below is the statement of one problem from the Codeforces problemset.
Estimate the difficulty rating that Codeforces would assign to this problem.

The rating scale (integer, in steps of 100):

  800-1200   very easy - basic implementation, no algorithm knowledge required
  1300-1600  easy - simple greedy / sorting / math, straightforward once read
  1700-2000  medium - one standard algorithm or data structure, or a non-obvious observation
  2100-2400  hard - advanced algorithms, or several insights combined
  2500-3500  very hard - solved in contest only by strong contestants

Rules:
1. Judge only from the text given below. Do not search for the problem, and do not
   try to recall its identity or its official rating. If it looks familiar, ignore
   that and base your answer on the content.
2. Answer with an integer between 800 and 3500, in multiples of 100.
   The minimum possible rating is 800.
3. Never refuse and never ask questions. Always produce the required output.
4. Output exactly one JSON object on one line, and nothing else - no explanation
   before or after it.

{"item_id": "<the id given below>", "rating": <integer>, "confidence": <number between 0 and 1>, "reason": "<at most 25 words>"}

--- PROBLEM ---
item_id: {item_id}
{tags_line}statement:
{statement}
--- END ---
"""

ROLE_LINE = ("You are an experienced competitive programmer who has solved "
             "thousands of problems.\n\n")

#: 消融子集：固定的前 N 题（item_id 已是打乱编号，与难度无关）
N_ABLATION = 200
N_NOISE = 100
NOISE_REPEATS = (2, 3)


def render(item_id: str, statement: str, tags: str, *, variant: str) -> str:
    tags_line = f"tags: {tags}\n" if (variant != "no_tags" and tags) else ""
    role_line = ROLE_LINE if variant == "role" else ""
    return (PROMPT_TEMPLATE
            .replace("{role_line}", role_line)
            .replace("{tags_line}", tags_line)
            .replace("{item_id}", item_id)
            .replace("{statement}", statement))


def main() -> int:
    src = LLM_DIR / "prompt_items.jsonl"
    if not src.exists():
        raise SystemExit(f"缺少 {src}；请先运行 scripts/make_llm_sample.py")
    items = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    if len(items) != 500:
        raise SystemExit(f"期望 500 题，实际 {len(items)} 题")

    tasks: list[dict] = []

    def add(it: dict, variant: str, repeat: int) -> None:
        tasks.append({
            "task_id": f"{it['item_id']}-{variant}-r{repeat}",
            "item_id": it["item_id"],
            "variant": variant,
            "repeat": repeat,
            "prompt": render(it["item_id"], it["statement"], it.get("tags", ""),
                             variant=variant),
        })

    # 主实验：500 题，每模型各 1 次
    for it in items:
        add(it, "baseline", 1)
    # 消融①：前 200 题，不给算法标签
    for it in items[:N_ABLATION]:
        add(it, "no_tags", 1)
    # 消融②：前 200 题，加角色设定
    for it in items[:N_ABLATION]:
        add(it, "role", 1)
    # 采样噪声基线：前 100 题额外重复 2 次（合计每题 3 次）
    for r in NOISE_REPEATS:
        for it in items[:N_NOISE]:
            add(it, "baseline", r)

    # ---- 写入 ----
    (LLM_DIR / "prompt.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    with (LLM_DIR / "llm_tasks.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ---- 旧文件归档，保证工作目录里只有一份 jsonl ----
    ARCHIVE.mkdir(exist_ok=True)
    moved = []
    for name in ("prompt_items.jsonl", "prompt_items_no_tags.jsonl",
                 "prompt_baseline.txt", "prompt_no_tags.txt", "prompt_role.txt",
                 "prompt_template_baseline.txt", "prompt_template_no_tags.txt",
                 "prompt_template_role.txt"):
        p = LLM_DIR / name
        if p.exists():
            p.replace(ARCHIVE / name)
            moved.append(name)

    print(f"统一任务包 -> {LLM_DIR}")
    print(f"  prompt.txt        唯一提示词模板（{len(PROMPT_TEMPLATE)} 字符，"
          f"占位符 role_line / tags_line / item_id / statement）")
    print(f"  llm_tasks.jsonl   {len(tasks)} 个任务（每个模型都要跑完这一份）")
    print(f"      主实验 500 + 消融①无标签 {N_ABLATION} + 消融②角色 {N_ABLATION}"
          f" + 噪声基线 {N_NOISE * len(NOISE_REPEATS)} = {len(tasks)}")
    print(f"    预计调用量：{len(tasks)} 任务 × 5 模型 = {len(tasks) * 5} 次")
    if moved:
        print(f"  已归档旧文件 {len(moved)} 个 -> archive/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
