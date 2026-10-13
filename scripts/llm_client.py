"""OpenAI 兼容接口的最小客户端（标准库实现，不引入新依赖）。

配置从项目根目录的 `接口与key.txt` 读取——**密钥不写进代码、不写进日志**。
该文件格式：

    1、mimo-v2.6-flash
    key：tp-xxxx
    接口：https://token-plan-cn.xiaomimimo.com/v1

    2、DeepSeek-V4.1-Flash
    key：sk-xxxx
    接口：https://api.deepseek.com

用法：
    from llm_client import load_endpoints, chat, list_models
    eps = load_endpoints()          # {显示名: Endpoint}
    chat(eps["mimo-v2.6-flash"], [{"role": "user", "content": "hi"}])
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "接口与key.txt"

#: 只允许使用 `接口与key.txt` 里列出的模型。这里再放一份白名单常量，
#: 是为了让"用了没被授权的模型"在代码层面直接报错，而不是靠人记得。
#: 若 key 文件内容变化，白名单会在 load_endpoints 时校验并报错。
ALLOWED_MODELS: tuple[str, ...] = (
    "mimo-v2.6-flash",
    "DeepSeek-V4.1-Flash",
    "qwen3.8-flash",
)


#: 显示名 → 接口实际接受的 `model` 参数。
#:
#: ⚠ 这张表是**实测**结果，不是推测：DeepSeek 官方端点只接受
#: `deepseek-flash` / `deepseek-v4-pro`，直接传 key 文件里的显示名
#: `DeepSeek-V4.1-Flash` 会返回 400
#: （"The supported API model names are deepseek-flash, deepseek-v4-pro,
#: but you passed DeepSeek-V4.1-Flash"）。
#: 其余两家显示名与 model id 一致，无需映射。
#: 正文报告模型时仍用**显示名**（即用户指定的名称），并注明接口 id。
MODEL_ID_OVERRIDE: dict[str, str] = {
    "DeepSeek-V4.1-Flash": "deepseek-flash",
}

#: 关闭"思考过程"的请求参数（**按模型分别实测**，不可混用）。
#:
#: 为什么要关：这三家都会先输出一段 `reasoning_content`，它**与正式答案共享
#: completion token 预算**。实测同一块 25 题：
#:
#: | 模型 | 参数 | 结果 |
#: |---|---|---|
#: | mimo-v2.6-flash | 默认（思考开） | 272 s，思考 26550 字，**8192 token 全被吃光，content 为空** |
#: | mimo-v2.6-flash | `reasoning_effort=none` | 6.9 s，思考 0 字，25/25 解析成功 |
#: | DeepSeek-V4.1-Flash | 默认 | 思考 31547 字，**content 为空** |
#: | DeepSeek-V4.1-Flash | `reasoning_effort=none` | 1.6 s，思考 0 字，25/25 成功 |
#: | qwen3.8-flash | 默认 | 思考 9137 字，2048 token 用尽，0/3 成功 |
#: | qwen3.8-flash | `enable_thinking=False` | 2.2 s，思考 0 字，25/25 成功 |
#:
#: ⚠ 参数名**不通用**：`reasoning_effort=none` 对 qwen 无效（照样思考），
#: `enable_thinking=False` 对 mimo/DeepSeek 只是缩短思考、并未关掉。
#: 因此必须按模型分别设置，且已用真实 25 题块逐家验证。
#:
#: 对实验口径的影响：关闭思考**不改变任务与输出格式**，只是不生成中间推理文本；
#: 五个模型里三家都关，条件一致。正文须写明这一点（否则"零样本能力"会被
#: 误读为"在受抑制的推理下测得"）。
DISABLE_THINKING: dict[str, dict[str, Any]] = {
    "mimo-v2.6-flash": {"reasoning_effort": "none"},
    "DeepSeek-V4.1-Flash": {"reasoning_effort": "none"},
    "qwen3.8-flash": {"enable_thinking": False},
}


@dataclass
class Endpoint:
    """一个模型端点。"""

    name: str                 # 用户在 key 文件里写的显示名（也是实验报告里的模型名）
    base_url: str
    api_key: str
    #: 实际传给接口的 `model` 参数。**必须与显示名分开**：实测 DeepSeek 官方端点
    #: 只接受 `deepseek-flash` / `deepseek-v4-pro`，传显示名 `DeepSeek-V4.1-Flash`
    #: 会被 400 拒绝。映射表见 `MODEL_ID_OVERRIDE`。
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = MODEL_ID_OVERRIDE.get(self.name, self.name)
        # 默认关闭思考；调用方显式传入的 extra 优先（便于做"开/关思考"的探测）
        for k, v in DISABLE_THINKING.get(self.name, {}).items():
            self.extra.setdefault(k, v)

    @property
    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        # DeepSeek 官方端点两种写法都接受；统一补 /chat/completions
        return f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"

    def masked_key(self) -> str:
        k = self.api_key
        return k[:6] + "…" + k[-4:] if len(k) > 12 else "…"


def parse_key_file(path: Path | str = KEY_FILE) -> list[Endpoint]:
    """解析 `接口与key.txt`。

    格式并不规整（全角顿号、全角冒号、条目间空行），因此用"遇到 key 行/接口行就
    归到当前条目"的状态机解析，而不是按固定行数切。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"未找到 {p}；请把接口与 key 放在项目根目录")
    text = p.read_text(encoding="utf-8-sig")

    eps: list[Endpoint] = []
    cur_name: str | None = None
    cur_key: str | None = None
    cur_base: str | None = None

    def flush() -> None:
        nonlocal cur_name, cur_key, cur_base
        if cur_name and cur_key and cur_base:
            eps.append(Endpoint(name=cur_name, base_url=cur_base, api_key=cur_key))
        cur_name, cur_key, cur_base = None, None, None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # 统一全角标点，简化匹配
        norm = line.replace("：", ":").replace("、", ".").replace("，", ",")
        m_key = re.match(r"^key\s*:\s*(\S+)", norm, re.IGNORECASE)
        m_url = re.match(r"^(?:接口|base_?url|url)\s*:\s*(\S+)", norm, re.IGNORECASE)
        if m_key:
            cur_key = m_key.group(1)
            continue
        if m_url:
            cur_base = m_url.group(1)
            continue
        # 形如 "1.mimo-v2.6-flash" / "2、DeepSeek-V4.1-Flash"
        m_name = re.match(r"^\d+\s*[.、)]\s*(.+)$", norm)
        if m_name:
            flush()
            cur_name = m_name.group(1).strip()
            continue
        # 兜底：不以数字开头的裸模型名行
        if not cur_name and not norm.startswith(("key", "接口")):
            cur_name = norm
    flush()

    if not eps:
        raise ValueError(f"{p} 未解析出任何端点；请检查格式（模型名 / key：/ 接口：）")
    return eps


def load_endpoints(path: Path | str = KEY_FILE,
                   only: Sequence[str] | None = None) -> dict[str, Endpoint]:
    """读取端点，并**强制白名单校验**：只允许 `接口与key.txt` 中列出的模型。"""
    eps = {e.name: e for e in parse_key_file(path)}
    unauthorized = [n for n in eps if n not in ALLOWED_MODELS]
    if unauthorized:
        raise ValueError(
            f"发现未授权的模型 {unauthorized}；本实验只允许使用 "
            f"{list(ALLOWED_MODELS)}。若确实要新增，请同时更新 ALLOWED_MODELS 与正文口径。"
        )
    missing = [m for m in ALLOWED_MODELS if m not in eps]
    if missing:
        raise ValueError(f"key 文件里缺少这些模型：{missing}（三者都应存在）")
    if only:
        bad = [m for m in only if m not in eps]
        if bad:
            raise ValueError(f"请求的模型不在 key 文件中：{bad}")
        return {m: eps[m] for m in only}
    # 固定顺序，保证实验可复现（不依赖字典顺序）
    return {m: eps[m] for m in ALLOWED_MODELS}


# ---------------------------------------------------------------- HTTP


def _post_json(url: str, key: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return e.code, detail
    except Exception as e:                      # 网络层失败（超时/DNS/连接被拒）
        return 0, f"{type(e).__name__}: {e}"


def _get_json(url: str, key: str, timeout: float) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def list_models(ep: Endpoint, timeout: float = 30.0) -> tuple[int, list[str] | str]:
    """列出该端点可用模型（用于确认 model_id 的正确写法）。"""
    code, body = _get_json(ep.models_url, ep.api_key, timeout)
    if code != 200 or not isinstance(body, dict):
        return code, str(body)[:300]
    ids = [m.get("id", "?") for m in body.get("data", [])]
    return code, ids


@dataclass
class ChatResult:
    ok: bool
    status: int
    text: str = ""
    error: str = ""
    model_returned: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_chars: int = 0
    finish_reason: str = ""
    seconds: float = 0.0
    raw: dict | None = None

    def summary(self) -> str:
        if self.ok:
            return (f"OK {self.status} {self.seconds:.1f}s "
                    f"model={self.model_returned} "
                    f"tok={self.prompt_tokens}+{self.completion_tokens} "
                    f"finish={self.finish_reason} "
                    f"reasoning={self.reasoning_chars}字符")
        return f"FAIL status={self.status} {self.error[:160]}"


#: 这些端点在正式回复前会先输出一段思考过程（`reasoning_content`），
#: 它**同样消耗 completion token**。实测把 max_tokens 设成 64 时，Qwen 的 64 个
#: token 全被思考吃光、`content` 返回空串，看起来像"模型不听话"，
#: 其实是预算不够。因此：
#:   1. 正式实验的 max_tokens 必须留足（25 行 JSON 约 500 token，思考另计）；
#:   2. 解析时**只取 `content`**，`reasoning_content` 单独计数供诊断。
REASONING_FIELD = "reasoning_content"


def chat(ep: Endpoint, messages: list[dict],
         max_tokens: int = 2048, temperature: float = 1.0,
         timeout: float = 600.0, retries: int = 2,
         model_id: str | None = None) -> ChatResult:
    """发一次对话请求。**一次调用 = 一次请求**，不做任何批量合并。

    重试只针对网络层失败与 429/5xx；4xx（除 429）直接返回，避免把
    "参数写错"当成"服务不稳"而反复重试。
    """
    payload = {
        "model": model_id or ep.model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(ep.extra)
    last = ChatResult(ok=False, status=0, error="未执行")
    for attempt in range(retries + 1):
        t0 = time.time()
        code, body = _post_json(ep.chat_url, ep.api_key, payload, timeout)
        dt = time.time() - t0
        if code == 200 and isinstance(body, dict):
            try:
                choice = body["choices"][0]
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                reasoning = msg.get(REASONING_FIELD) or ""
                usage = body.get("usage") or {}
                return ChatResult(
                    ok=True, status=code, text=text, seconds=dt,
                    model_returned=str(body.get("model", "")),
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    reasoning_chars=len(reasoning),
                    finish_reason=str(choice.get("finish_reason", "")),
                    raw=body,
                )
            except Exception as e:
                return ChatResult(ok=False, status=code, seconds=dt,
                                  error=f"响应结构异常: {type(e).__name__}: {e}",
                                  raw=body if isinstance(body, dict) else None)
        err = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)[:400]
        last = ChatResult(ok=False, status=code, error=err, seconds=dt)
        retryable = code in (0, 429, 500, 502, 503, 504)
        if attempt < retries and retryable:
            time.sleep(2.0 * (attempt + 1))
            continue
        return last
    return last
