"""导出带占位符的提示词模板（供批量调用时做字符串替换）。

为什么不直接让使用者改 `prompt_baseline.txt`：那三份 `.txt` 是**用 K0001 渲染好的样例**，
里面已经填进了真实题面，容易误当成"模板文件加上题目"而重复粘贴；
一旦有人手工删改，就会与主实验口径不一致（提示词消融要求除被消融处外逐字相同）。
这里从同一个 `build_prompt` 导出占位符版本，保证模板与代码永远同源。

    python scripts/make_llm_prompt_templates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task2difficulty import config as C  # noqa: E402

from make_llm_sample import build_prompt  # noqa: E402  与抽样脚本共用同一实现

PLACEHOLDER = {
    "item_id": "{item_id}",
    "tags": "{tags}",
    "statement": "{statement}",
}

VARIANTS = {
    "prompt_template_baseline.txt": dict(with_tags=True, role=False),
    "prompt_template_no_tags.txt": dict(with_tags=False, role=False),
    "prompt_template_role.txt": dict(with_tags=True, role=True),
}


def main() -> int:
    out = C.OUTPUTS_DIR / "llm"
    out.mkdir(parents=True, exist_ok=True)
    for fn, kw in VARIANTS.items():
        text = build_prompt(PLACEHOLDER, **kw)
        (out / fn).write_text(text, encoding="utf-8")
        print(f"  {fn}  {len(text)} 字符")
    print("\n占位符：{item_id} / {tags} / {statement}"
          "\n（无标签模板里没有 {tags}，替换时该处直接省略）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
