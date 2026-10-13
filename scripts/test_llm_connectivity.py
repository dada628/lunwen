"""连通性探测：确认三个端点可达、密钥有效、模型名正确、能返回可解析的 JSON。

⚠ 本脚本**只做探测，不跑实验**。它发出的请求都是 1 题的小请求，用于确定：
  1. 网络与密钥是否可用；
  2. `model` 参数该写什么（很多网关的模型名与展示名不同，必须实测）；
  3. 上下文长度上限（用一条长输入探边界）；
  4. 返回是否稳定为一行 JSON、是否会被思考过程/围栏污染。

用法：
    python scripts/test_llm_connectivity.py
    python scripts/test_llm_connectivity.py --long     # 追加长上下文探测（约 25 题的量）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_client import Endpoint, chat, list_models, load_endpoints  # noqa: E402

LLM_DIR = ROOT / "outputs" / "llm"

#: 探测用的一句话请求：要求输出 JSON，用来验证返回格式是否可解析
PROBE_PROMPT = (
    'Reply with exactly one JSON object on one line and nothing else: '
    '{"ok": true, "n": 42}'
)


def probe_basic(ep: Endpoint, timeout: float) -> dict:
    """基础探测：/models 列表 + 一次最小 chat 请求。"""
    out: dict = {"name": ep.name, "base_url": ep.base_url,
                 "key": ep.masked_key(), "model_id": ep.model_id}
    code, models = list_models(ep, timeout=min(timeout, 60))
    if isinstance(models, list):
        out["models_http"] = code
        out["n_models"] = len(models)
        out["models_sample"] = models[:12]
        # 若声明的 model_id 不在列表里，尝试在列表中找到最接近的写法
        if ep.model_id not in models:
            cand = [m for m in models if ep.model_id.lower().replace("-", "") in
                    m.lower().replace("-", "").replace("_", "")]
            out["model_id_match"] = cand[:5]
    else:
        out["models_http"] = code
        out["models_error"] = str(models)[:200]

    r = chat(ep, [{"role": "user", "content": PROBE_PROMPT}],
             max_tokens=64, temperature=0.0, timeout=timeout)
    out["chat"] = r.summary()
    out["chat_text"] = r.text.strip()[:200]
    out["chat_parsed"] = _try_parse_json(r.text)
    return out


def _try_parse_json(text: str) -> object:
    """从返回文本里尽力提取 JSON（容忍 ```json 围栏与前后缀）。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s.lstrip("`")
        s = s.removeprefix("json").strip()
    for cand in (s, *[ln for ln in s.splitlines() if ln.strip().startswith("{")]):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def probe_long(ep: Endpoint, timeout: float, per_item_chars: int = 1530) -> dict:
    """长上下文探测：塞入约 25 题体量的文本，看是否报超长、能否完整作答。

    只放 25 条极简"题"（每条是填充文本 + 一个 id），要求回 25 行 JSON——
    目的不是评估能力，而是量出该端点的输入/输出上限。
    """
    lines = []
    for i in range(25):
        body = ("This is a filler statement used only to measure the context limit. "
                * (per_item_chars // 60))
        lines.append(f"item_id: P{i + 1:03d}\nstatement:\n{body}\n=== END P{i + 1:03d} ===")
    prompt = (
        "For each of the 25 problems below, output one JSON object per line, in order:\n"
        '{"item_id": "<the id>", "rating": <integer 800-3500>}\n'
        "Output nothing else.\n\n=== PROBLEMS ===\n" + "\n".join(lines) + "\n=== END OF PROBLEMS ===\n"
    )
    t0 = time.time()
    r = chat(ep, [{"role": "user", "content": prompt}],
             max_tokens=2048, temperature=0.0, timeout=timeout, retries=0)
    n_lines = len([l for l in r.text.splitlines() if l.strip().startswith("{")])
    return {
        "name": ep.name,
        "prompt_chars": len(prompt),
        "est_tokens": len(prompt) // 4,
        "seconds": round(time.time() - t0, 1),
        "ok": r.ok,
        "status": r.status,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "json_lines": n_lines,
        "expected_lines": 25,
        "error": r.error[:200],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 端点连通性探测（不跑实验）")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--long", action="store_true", help="追加长上下文探测")
    ap.add_argument("--only", nargs="*", default=None, help="只测指定模型")
    args = ap.parse_args()

    eps = load_endpoints(only=args.only)
    print("=" * 78)
    print("LLM 端点连通性探测（只探测，不跑实验）")
    print("=" * 78)
    print(f"授权模型（来自 接口与key.txt）：{list(eps)}")
    print()

    report: dict = {"probes": [], "long": [], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name, ep in eps.items():
        print(f"--- {name} ---")
        print(f"    base_url : {ep.base_url}")
        print(f"    key      : {ep.masked_key()}")
        info = probe_basic(ep, args.timeout)
        if "models_sample" in info:
            print(f"    /models  : HTTP {info['models_http']}，{info['n_models']} 个模型")
            print(f"    样例     : {info['models_sample']}")
            if info.get("model_id_match"):
                print(f"    ⚠ 声明的 model_id 不在列表中，近似匹配：{info['model_id_match']}")
        else:
            print(f"    /models  : HTTP {info['models_http']}  {info.get('models_error','')}")
        print(f"    chat     : {info['chat']}")
        if info["chat_text"]:
            print(f"    返回原文 : {info['chat_text'][:120]!r}")
        print(f"    JSON 可解析: {info['chat_parsed'] is not None}"
              f"{'  -> ' + json.dumps(info['chat_parsed'], ensure_ascii=False) if info['chat_parsed'] else ''}")
        report["probes"].append(info)
        print()

    if args.long:
        print("=" * 78)
        print("长上下文探测（约 25 题体量，输入 ≈ 9.5k token）")
        print("=" * 78)
        for name, ep in eps.items():
            r = probe_long(ep, args.timeout)
            print(f"  {name:<22s} {r['prompt_chars']:>7,} 字符 ≈{r['est_tokens']:>6,} tok  "
                  f"HTTP {r['status']:<4} {r['seconds']:>6.1f}s  "
                  f"返回 JSON 行 {r['json_lines']}/{r['expected_lines']}"
                  + (f"  错误: {r['error']}" if r["error"] else ""))
            report["long"].append(r)

    out = LLM_DIR / "connectivity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
