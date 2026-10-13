"""四·（五）LLM 零样本实验的执行器：**一次请求只发一块**，逐次串行。

⚠ 本脚本只允许使用 `接口与key.txt` 里列出的三个模型（白名单在 `llm_client` 里强制），
禁止换成别的模型。

协议
----
读 `outputs/llm/llm_batches.jsonl`（56 块，每块 25 题），对每个模型逐块发送：
主实验 20 块 → 采样噪声基线 20 块（同块重发）→ 无标签消融 8 块 → 角色消融 8 块。
每块一次请求，**不并行、不合并**；块与块之间留 `--sleep` 秒。

产出
----
- `outputs/llm/llm_results_<模型>.jsonl`  一行一次**请求**（含原文与用量，可审计）
- `outputs/llm/ratings_<模型>.jsonl`      一行一个**题**（已解析的 rating，供分析）
- `outputs/llm/llm_run_summary.json`      运行汇总

用法
----
    # 先小规模验证（每个模型只跑第 1 块，结果写到 _dryrun 后缀，不污染正式结果）
    python scripts/run_llm_experiment.py --limit 1 --out-suffix _dryrun

    # 正式全量
    python scripts/run_llm_experiment.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_client import Endpoint, chat, load_endpoints  # noqa: E402

LLM_DIR = ROOT / "outputs" / "llm"
BATCH_FILE = LLM_DIR / "llm_batches.jsonl"

#: 每题允许的 rating 区间与步长（与 config.DIFFICULTY_BINS 的口径一致）
RATING_MIN, RATING_MAX, RATING_STEP = 800.0, 3500.0, 100.0

#: 匹配一行 JSON 对象（含 item_id 与 rating 两个键）
_OBJ_RE = re.compile(r'\{[^{}]*"item_id"[^{}]*\}', re.S)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_ratings(text: str) -> dict[str, float]:
    """从返回原文里提取 `{item_id: rating}`。

    容忍：```json 围栏、前后说明文字、JSON 数组、单行多个对象。
    ⚠ 只取 `content`（调用方已剥离 reasoning_content），否则思考文本里的
    示例 JSON 会被误当成答案。
    """
    s = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", text or "", flags=re.M)
    s = s.replace("```", "")
    out: dict[str, float] = {}
    for m in _OBJ_RE.finditer(s):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        iid, r = obj.get("item_id"), obj.get("rating")
        if iid is None or r is None:
            continue
        try:
            out[str(iid).strip()] = float(r)
        except Exception:
            continue
    if not out:                       # 兜底：整个响应是一个 JSON 数组
        try:
            arr = json.loads(s.strip())
        except Exception:
            return out
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict) and obj.get("item_id") is not None \
                        and obj.get("rating") is not None:
                    try:
                        out[str(obj["item_id"]).strip()] = float(obj["rating"])
                    except Exception:
                        pass
    return out


def validate_ratings(found: dict[str, float], expected_ids: list[str]) -> dict:
    """核对解析结果：缺哪些题、哪些 rating 不合规。"""
    missing = [i for i in expected_ids if i not in found]
    extra = [i for i in found if i not in set(expected_ids)]
    bad = {i: found[i] for i in found
           if not (RATING_MIN <= found[i] <= RATING_MAX)
           or abs(found[i] / RATING_STEP - round(found[i] / RATING_STEP)) > 1e-9}
    return {"n_found": len(found), "missing": missing, "extra": extra, "invalid": bad}


def load_batches(limit: int | None = None) -> list[dict]:
    if not BATCH_FILE.exists():
        raise SystemExit(f"缺少 {BATCH_FILE}；请先运行 scripts/make_llm_batches.py --per-batch 25")
    bs = [json.loads(l) for l in BATCH_FILE.open(encoding="utf-8") if l.strip()]
    if limit:
        bs = bs[:limit]
    return bs


def load_done(results_path: Path) -> set[str]:
    """已完成（ok=true）的 batch_id 集合，用于断点续跑。"""
    if not results_path.exists():
        return set()
    done: set[str] = set()
    for line in results_path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ok"):
            done.add(r["batch_id"])
    return done


def run_model(ep: Endpoint, batches: list[dict], args) -> dict:
    suffix = args.out_suffix
    res_path = LLM_DIR / f"llm_results_{ep.name}{suffix}.jsonl"
    rat_path = LLM_DIR / f"ratings_{ep.name}{suffix}.jsonl"
    done = set() if args.force else load_done(res_path)

    log(f"=== {ep.name}（接口 model={ep.model_id}）===")
    if done:
        log(f"  已完成 {len(done)} 块，跳过（--force 可重跑）")

    n_req = n_ok = n_trunc = 0
    t_start = time.time()
    with res_path.open("a", encoding="utf-8") as fres, \
            rat_path.open("a", encoding="utf-8") as frat:
        for b in batches:
            if b["batch_id"] in done:
                continue
            expected = b["item_ids"]
            attempt = 0
            while True:
                attempt += 1
                t0 = time.time()
                r = chat(ep, [{"role": "user", "content": b["prompt"]}],
                         max_tokens=args.max_tokens, temperature=args.temperature,
                         timeout=args.timeout, retries=args.retries)
                found = extract_ratings(r.text) if r.ok else {}
                chk = validate_ratings(found, expected)
                complete = r.ok and not chk["missing"]
                n_req += 1

                rec = {
                    "batch_id": b["batch_id"], "variant": b["variant"],
                    "repeat": b["repeat"], "n_items": b["n_items"],
                    "attempt": attempt,
                    "model": ep.name, "model_id": ep.model_id,
                    "model_version": r.model_returned,
                    "temperature": args.temperature, "max_tokens": args.max_tokens,
                    "finish_reason": r.finish_reason,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "reasoning_chars": r.reasoning_chars,
                    "seconds": round(r.seconds, 2),
                    "ok": bool(complete), "http_status": r.status,
                    "n_parsed": chk["n_found"], "n_expected": len(expected),
                    "n_missing": len(chk["missing"]), "n_invalid": len(chk["invalid"]),
                    "missing_ids": chk["missing"][:10],
                    "invalid_examples": dict(list(chk["invalid"].items())[:5]),
                    "error": r.error[:300],
                    "raw_response": r.text,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                }
                fres.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fres.flush()

                if complete:
                    for iid in expected:
                        frat.write(json.dumps({
                            "item_id": iid, "rating": found[iid],
                            "batch_id": b["batch_id"], "variant": b["variant"],
                            "repeat": b["repeat"], "model": ep.name,
                        }, ensure_ascii=False) + "\n")
                    frat.flush()
                    n_ok += 1
                    log(f"  {b['batch_id']:<22s} {chk['n_found']:>3d}/{len(expected)} "
                        f"{r.seconds:>6.1f}s finish={r.finish_reason} "
                        f"tok={r.prompt_tokens}+{r.completion_tokens}"
                        f"(思考{r.reasoning_chars}字)"
                        + (f"  ⚠ 不合规 {len(chk['invalid'])} 个" if chk["invalid"] else ""))
                    break
                # 不完整：重试一次（多半是输出被截断）
                n_trunc += 1
                log(f"  {b['batch_id']:<22s} 不完整 {chk['n_found']}/{len(expected)} "
                    f"finish={r.finish_reason} status={r.status} "
                    f"{('错误: ' + r.error[:90]) if r.error else ''}")
                if attempt > args.max_attempts:
                    log(f"  {b['batch_id']:<22s} 重试 {attempt - 1} 次仍不完整，记为该块失败")
                    break
                time.sleep(args.retry_sleep)
            if args.sleep:
                time.sleep(args.sleep)

    return {
        "model": ep.name, "model_id": ep.model_id,
        "n_requests": n_req, "n_complete": n_ok, "n_incomplete": n_trunc,
        "seconds": round(time.time() - t_start, 1),
        "results_file": str(res_path.relative_to(ROOT)),
        "ratings_file": str(rat_path.relative_to(ROOT)),
    }


def noise_check(ep: Endpoint, batches: list[dict], args) -> dict:
    """噪声基线有效性检查：同一块连发两次，看两次答案是否**真的不同**。

    若完全相同，说明该端点对相同请求做了缓存（或 temperature 未生效），
    那么"采样噪声基线"就退化成"重复读取同一答案"，跨模型一致率的下限参照失效，
    必须改用别的温度设置或换参数重测。
    """
    b = batches[0]
    log(f"  噪声检查：{b['batch_id']} 连发两次，比对差异…")
    outs = []
    for k in (1, 2):
        r = chat(ep, [{"role": "user", "content": b["prompt"]}],
                 max_tokens=args.max_tokens, temperature=args.temperature,
                 timeout=args.timeout, retries=args.retries)
        outs.append(extract_ratings(r.text) if r.ok else {})
        time.sleep(1.0)
    common = [i for i in b["item_ids"] if i in outs[0] and i in outs[1]]
    diff = [i for i in common if abs(outs[0][i] - outs[1][i]) > 1e-9]
    same = len(common) - len(diff)
    return {"model": ep.name, "batch_id": b["batch_id"], "n_common": len(common),
            "n_same": same, "n_diff": len(diff),
            "identical": bool(common) and not diff}


def main() -> int:
    ap = argparse.ArgumentParser(description="四·（五）LLM 零样本实验执行器（单次请求、串行）")
    ap.add_argument("--models", nargs="*", default=None,
                    help="只跑指定模型（默认三个全跑，且只能是 key 文件里的三个）")
    ap.add_argument("--limit", type=int, default=None, help="每个模型只跑前 N 块（验证用）")
    ap.add_argument("--out-suffix", default="", help="结果文件名后缀（验证用 _dryrun）")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="留足余量：25 行 JSON 约 500 token；关掉思考后 4096 已很宽松")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=3, help="同一块最多发几次")
    ap.add_argument("--sleep", type=float, default=1.0, help="块与块之间的间隔秒数")
    ap.add_argument("--retry-sleep", type=float, default=5.0)
    ap.add_argument("--force", action="store_true", help="忽略已完成记录，重跑")
    ap.add_argument("--noise-check", action="store_true", help="额外做一次噪声基线有效性检查")
    args = ap.parse_args()

    eps = load_endpoints(only=args.models)
    batches = load_batches(args.limit)
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    log(f"模型 {list(eps)}；块 {len(batches)}（每块 {batches[0]['n_items']} 题）")
    log(f"预计请求数：{len(batches)} 块 × {len(eps)} 模型 = {len(batches) * len(eps)}"
        + (f"（含噪声检查另加 {len(eps)} 次）" if args.noise_check else ""))
    if args.out_suffix:
        log(f"⚠ 结果写入后缀 {args.out_suffix} 的文件，不覆盖正式结果")

    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
               "n_batches": len(batches), "models": [], "noise": []}
    for name, ep in eps.items():
        try:
            summary["models"].append(run_model(ep, batches, args))
        except KeyboardInterrupt:
            log("收到中断，已保存已完成的部分。")
            break
        except Exception as e:
            log(f"  ✗ {name} 异常：{type(e).__name__}: {e}")
            summary["models"].append({"model": name, "error": f"{type(e).__name__}: {e}"})
        if args.noise_check:
            try:
                summary["noise"].append(noise_check(ep, batches, args))
            except Exception as e:
                summary["noise"].append({"model": name, "error": str(e)})

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    out = LLM_DIR / f"llm_run_summary{args.out_suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    log("=" * 70)
    for m in summary["models"]:
        if "error" in m:
            log(f"  {m['model']:<22s} ✗ {m['error'][:70]}")
        else:
            log(f"  {m['model']:<22s} 请求 {m['n_requests']:>3d}  "
                f"完整 {m['n_complete']:>3d}  不完整 {m['n_incomplete']:>3d}  "
                f"{m['seconds']:>7.1f}s")
    for n in summary["noise"]:
        if "error" in n:
            log(f"  噪声检查 {n['model']}: ✗ {n['error'][:60]}")
        else:
            log(f"  噪声检查 {n['model']:<22s} 两次相同 {n['n_same']}/{n['n_common']} "
                f"不同 {n['n_diff']}"
                + ("  ⚠ 完全相同，可能被缓存，噪声基线失效！" if n["identical"] else ""))
    log(f"汇总 -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
