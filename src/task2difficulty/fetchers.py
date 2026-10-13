"""Codeforces 数据抓取与题面解析（大纲二·（一）（二））。

本模块只依赖标准库 + bs4/lxml，把"从官方接口与题面页拿到可用数据"这件事隔离出来，
便于在抓取失败的任意点重跑而不影响下游。

★ 接口边界（实测确认，直接决定哪些实验可做）
-------------------------------------------
- `problemset.problems`：
    * `result.problems[]` 含 `contestId`/`index`/`name`/`tags`/`rating`/`type`/`points`；
    * `result.problemStatistics[]` **只有** `contestId`/`index`/`solvedCount`；
    * **只有当前累计值，没有任何历史/时点信息**，也**不提供题目发布日期**。
- `contest.list`：含 `id`/`name`/`startTimeSeconds`/`phase`/`durationSeconds`，
  **按 contestId 与上表 join 才能得到年份**，时间划分依赖这一步。
- 题面：API 不提供，须抓 `problemset/problem/{contestId}/{index}` 页面。

★ 限速
------
官方 API 文档（https://codeforces.com/apiHelp Introduction）规定
"may be requested at most 1 time per two seconds"，超限返回 `Call limit exceeded`。
**这是官方规定而非社区惯例**。本文档仍自设更保守的间隔以降低被拒风险，
两者在正文中必须分清表述。
"""

from __future__ import annotations

import html as html_mod
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# ================================================================ 常量

API_BASE = "https://codeforces.com/api"
PROBLEM_URL = "https://codeforces.com/problemset/problem/{cid}/{index}"

#: 官方限速下限（1 次 / 2 秒），**仅约束 API 端点**。
#: 依据：https://codeforces.com/apiHelp 的 Introduction
#: "may be requested at most 1 time per two seconds"，超限返回 `Call limit exceeded`。
OFFICIAL_MIN_INTERVAL_S = 2.0

#: 默认自设间隔（比官方保守），写进正文时须注明"自设"。
DEFAULT_API_INTERVAL_S = 2.2
#: 题面页（HTML）不属于 API，官方限速条款不覆盖它。
#: 但仍自设保守间隔 + 串行抓取，避免给站点造成压力；正文须写明这是自设策略。
DEFAULT_HTML_INTERVAL_S = 1.5
#: HTML 端的最低间隔（自设的礼貌下限，非官方规定）
MIN_HTML_INTERVAL_S = 1.0

USER_AGENT = ("Mozilla/5.0 (compatible; difficulty-prediction-research/0.1; "
              "academic use; contact via paper corresponding author)")


# ================================================================ 基础工具


def split_problem_key(key: str) -> tuple[int, str]:
    """把 '1582A' / '4-A' 之类的键拆成 (contestId, index)。

    注意 index 可能是多字符（如 'A1'、'B2'），不能用单字符切片。
    """
    s = str(key).strip().upper()
    if "-" in s:
        cid, idx = s.split("-", 1)
        return int(cid), idx
    m = re.match(r"^(\d+)([A-Z][A-Z0-9]*)$", s)
    if not m:
        raise ValueError(f"无法解析题目键: {key!r}")
    return int(m.group(1)), m.group(2)


def parse_div_type(contest_name: str) -> int:
    """从比赛名解析 Div 类型（来源特征，高危，须单列）。

    返回值含义：1 = Div.1（最难），2 = Div.2，3 = Div.3，4 = Div.4（最易），
    0 = 无法识别（Educational / Global Round / 特殊赛制等）。

    ⚠ 该特征是**标签的强代理**：Div.1 必偏难、Div.4 必简单。
    大纲二·（四）要求单列成组，并在四·（四）做 provenance 消融 + 弱基线自曝，
    **不得作为普通特征混入**。
    """
    if not isinstance(contest_name, str):
        return 0
    for d in (1, 2, 3, 4):
        if re.search(rf"\bDiv\.?\s*{d}\b", contest_name, re.IGNORECASE):
            return d
    return 0


# ================================================================ 解析题面 HTML

#: 题面正文容器。页面结构为 <div class="problem-statement">…</div>，
#: 内部按 class 分为 header / 正文 / input-specification / output-specification /
#: sample-tests / note 等段落（见 `extract_statement_parts` 的说明）。
_RE_PROBLEM_STATEMENT = re.compile(
    r'<div\s+class="problem-statement"[^>]*>(.*)', re.S | re.I)
_RE_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
_RE_TAG_ANY = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"[ \t\r\f\v]+")
_RE_NL = re.compile(r"\n{3,}")


def _strip_balanced_div(html: str, class_name: str) -> str:
    """删除 `<div class="class_name">…</div>`，正确处理嵌套 div。

    天真正则 `<div class=X>.*?</div>` 会在第一个 `</div>` 处截断，
    而 sample-test / note 内部都有嵌套 div，因此必须做括号配平。

    实测要点：CF 的 div 常有**复合 class**，例如
    `<div class="input-file input-standard">`、`<div class="output-file output-standard">`。
    若要求 class 值**精确等于** `"input-file"`，这些块就删不掉（实测踩过：
    `stdin`/`stdout` 因此残留在正文里）。因此这里用"class 属性中**包含**该词"匹配。
    """
    out: list[str] = []
    i = 0
    n = len(html)
    # (?<![\w-]) 避免 "note" 命中 "footnote"；class 属性内允许其它词共存
    open_pat = re.compile(
        rf'<div\s+class="[^"]*(?<![\w-]){re.escape(class_name)}(?![\w-])[^"]*"[^>]*>',
        re.I)
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)

    while i < n:
        m = open_pat.search(html, i)
        if not m:
            out.append(html[i:])
            break
        out.append(html[i:m.start()])
        depth = 1
        j = m.end()
        while depth > 0:
            dm = div_pat.search(html, j)
            if not dm:
                j = n
                break
            depth += -1 if dm.group(1) else 1
            j = dm.end()
        i = j
    return "".join(out)


def _inner_of_div(html: str, start: int) -> str:
    """从 `<div …>` 的起始位置开始，返回该 div 的**内容**（配平嵌套）。

    比 `(.*)` 贪婪匹配可靠得多：CF 的 `problem-statement` 之后还有
    大段 JS 与 `<div id="footer">`，贪婪匹配会把它们全部吞进来。
    """
    m = re.compile(r"<div\b[^>]*>", re.I).match(html, start)
    if not m:
        return ""
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)
    depth = 1
    j = m.end()
    while depth > 0:
        dm = div_pat.search(html, j)
        if not dm:
            return html[m.end():]
        depth += -1 if dm.group(1) else 1
        j = dm.end()
    return html[m.end():dm.start()]


def _extract_div_by_class(html: str, class_name: str) -> str:
    """取出 `<div class="…class_name…">…</div>` 的**内容**（不含外层标签）。

    与 `_strip_balanced_div` 同一套配平逻辑，用于按 section 精确切分题面。
    """
    open_pat = re.compile(
        rf'<div\s+class="[^"]*(?<![\w-]){re.escape(class_name)}(?![\w-])[^"]*"[^>]*>',
        re.I)
    m = open_pat.search(html)
    if not m:
        return ""
    div_pat = re.compile(r"<(/?)div\b[^>]*>", re.I)
    depth = 1
    j = m.end()
    while depth > 0:
        dm = div_pat.search(html, j)
        if not dm:
            return html[m.end():]
        depth += -1 if dm.group(1) else 1
        j = dm.end()
    return html[m.end():dm.start()]


_RE_SUP = re.compile(r"(?is)<sup[^>]*>(.*?)</sup>")
_RE_SUB = re.compile(r"(?is)<sub[^>]*>(.*?)</sub>")


def _html_to_text(html: str) -> str:
    """HTML → 纯文本，保留段落与行结构。

    ★ 上标必须转成 `^{k}` 再剥标签，不能直接拍平。
    实测：CF 把 10 的幂写成 `10<sup class="upper-index">9</sup>`。
    若直接去标签会得到 `10 9`，约束解析就只能读到 `10`——
    导致 n 上界被系统性低估 9 个数量级（实测 1A 的 1≤n,m,a≤10^9 曾被解析为 10）。
    `_to_number` 能解析 `10^{9}`，因此这里统一输出成 `^{…}` 形式。

    所有块级标签（含无 class 的裸 `<div>`，CF 的正文段落正是用它包的）
    都转成换行，避免相邻段落粘连。
    """
    s = _RE_SCRIPT_STYLE.sub(" ", html)
    s = _RE_SUP.sub(lambda m: "^{" + _RE_TAG_ANY.sub("", m.group(1)).strip() + "}", s)
    s = _RE_SUB.sub(lambda m: "_{" + _RE_TAG_ANY.sub("", m.group(1)).strip() + "}", s)
    s = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|pre|section|article)>", "\n", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = _RE_TAG_ANY.sub(" ", s)
    s = html_mod.unescape(s)
    # $$$…$$$ 是 Codeforces 的行内公式定界符，替换为空格但**保留内部 LaTeX**
    # （约束解析依赖 \le 这类命令，因此不能在这一步把 LaTeX 去掉）
    s = s.replace("$$$", " ")
    s = _RE_WS.sub(" ", s)
    s = _RE_NL.sub("\n\n", s)
    return s.strip()



def extract_statement_parts(html: str) -> dict[str, str]:
    """从题目页 HTML 抽出题面各部分。

    ★ 方法：按 **CSS 类**逐段切片，不用文本启发式。
    实测 CF 题面容器的结构（以 1A 为例）：

        <div class="problem-statement">
          <div class="header">…标题 / 时限 / 内存限制 / stdin / stdout…</div>
          <div>…题面正文…</div>                        ← 无 class，须按位置切
          <div class="input-specification">…</div>
          <div class="output-specification">…</div>
          <div class="sample-tests">…</div>            ← 注意是复数 sample-tests
          <div class="note">…</div>
        </div>

    之前用"按行匹配小节标题"的文本启发式切分，实测有三处会出错：
    1. 头部块（`header` 内的 title / time-limit / memory-limit / input-file /
       output-file）会整段并入正文——实测 391 题里 100% 混入了 "time limit"；
    2. `input-file input-standard` 这类**复合 class** 若按精确匹配删不掉，
       `stdin`/`stdout` 会残留；
    3. 只匹配 "input"/"output" 而漏掉 "standard input"/"standard output"，
       Input 规范整段并入正文，同时污染文本特征与约束特征。
    改为按 class 切片后，以上问题一次性消失。

    返回 raw_html（正文，供约束解析与模板句消融）、preamble、input_spec、
    output_spec、is_interactive。
    """
    m = _RE_PROBLEM_STATEMENT.search(html)
    if not m:
        return {"raw_html": "", "preamble": "", "input_spec": "", "output_spec": "",
                "interaction_spec": "", "is_interactive": "0"}
    # 按配平取容器**内容**，不可用贪婪 (.*)：CF 的 problem-statement 之后还有
    # JS 脚本、`<div id="footer">` 等大段内容。若一路吃到页面末尾，
    # 页脚 "Codeforces (c) Copyright 2010-2026 Mike Mirzayanov" 与脚本残渣
    # 会混进题面（实测 1791A 的正文尾部就带上了版权行）。
    body = _inner_of_div(html, m.start())

    # ---- 按 class 精确取各 section（取不到则回退为空串）----
    header = _extract_div_by_class(body, "header")
    input_spec = _extract_div_by_class(body, "input-specification")
    output_spec = _extract_div_by_class(body, "output-specification")
    interaction = (_extract_div_by_class(body, "interaction-specification")
                   or _extract_div_by_class(body, "interaction"))

    # ---- 正文 = 容器去掉 header / 各 section / 样例 / note 后剩下的部分 ----
    prose = body
    for cls in ("header", "input-specification", "output-specification",
                "interaction-specification", "interaction",
                "sample-tests", "sample-test", "note"):
        prose = _strip_balanced_div(prose, cls)
    prose_text = _html_to_text(prose)

    # 清理各 section 开头的 section-title（"Input" / "Output" / "Interaction"）
    def drop_title(seg: str) -> str:
        seg = re.sub(r"(?is)<div\s+class=\"section-title\"[^>]*>.*?</div>", " ", seg)
        txt = _html_to_text(seg)
        # 首行若只是节名，去掉
        first, _, rest = txt.partition("\n")
        if first.strip().lower().strip(" .:：") in (
                "input", "output", "interaction", "note", "example", "examples"):
            return rest.strip()
        return txt.strip()

    input_text = drop_title(input_spec)
    output_text = drop_title(output_spec)
    interaction_text = drop_title(interaction)

    # ---- 交互题判定 ----
    # 优先用官方标注的 interaction 段落（最可靠）；否则退回文本模式。
    # 实测：1286C2 写的是 "It is an interactive problem"，而只匹配
    # "This is an interactive problem" 会漏判，因此这里用较宽的短语集合。
    low_all = (html or "").lower()
    interactive = bool(
        interaction.strip()
        or re.search(r"hunk-title[^>]*>\s*interaction", low_all)
        or re.search(r"interactive\s+problem", low_all)
        or re.search(r"this\s+is\s+an?\s+interactive", low_all)
    )

    return {
        "raw_html": prose,
        "preamble": prose_text,
        "input_spec": input_text,
        "output_spec": output_text,
        "interaction_spec": interaction_text,
        "is_interactive": "1" if interactive else "0",
    }


# ================================================================ 约束解析

_NUM = r"(?:\d+(?:\.\d+)?(?:\s*[eE]\s*[+-]?\d+)?)"
#: 10 的幂，覆盖四种实测写法 10^5 / 10^{5} / 10^ 5 / 10 ^ { 5 }
_POW10 = r"10\s*\^\s*\{?\s*(\d+)\s*\}?"
#: 科学计数法的乘数前缀，覆盖 2·10^5 / 2\times10^5 / 2*10^5
_MULT = r"(?:(\d+(?:\.\d+)?)\s*(?:\\cdot|\\times|\*|·)?\s*)?"
#: 单个界 = （可选乘数 × 10 的幂）或纯数字。
#: ⚠ 幂分支必须排在纯数字分支**之前**，否则 10^9 会先被纯数字分支匹配成 10，
#: 导致 n 上界被系统性低估（实测 1582A 的 1<=a,b,c<=10^9 曾被解析为 10）。
_BOUND = rf"(?:{_MULT}{_POW10}|{_NUM})"

#: 变量名（允许下标，如 a_i、b_1）
_VAR = r"[a-zA-Z](?:_\{?\w+\}?)?"
#: 变量列表：a, b, c / a_i / n
_VARLIST = rf"{_VAR}(?:\s*,\s*{_VAR})*"

_SUP = r"(?:\\le|\\leq|<=|≤|\\lt|<)"
_INF = r"(?:\\ge|\\geq|>=|≥|\\gt|>|\\lt|<)"   # 左右界在同一表达式中只出现一次时用它

#: 形如  1 \le n, t \le 10^5   或   1 \le a_i \le 10^9
_RE_CHAIN = re.compile(
    rf"(?P<lo>{_BOUND})\s*{_SUP}\s*(?P<vars>{_VARLIST})\s*{_SUP}\s*(?P<hi>{_BOUND})"
)
#: 形如  n \le 10^5 （只有上界）
_RE_UPPER_ONLY = re.compile(rf"(?P<vars>{_VARLIST})\s*{_SUP}\s*(?P<hi>{_BOUND})")


#: 指数的合理上限。真实算法题的规模上界不会超过 10^18 量级；
#: 出现更大的指数必然是解析误匹配（实测：某题面含 10^{1000} 之类的数学说明，
#: 直接做 10.0 ** 1000 会抛 OverflowError 中断整条管线）。
#: 超过上限的匹配按"无效"处理，丢弃而不是崩溃。
_MAX_EXPONENT = 64
_MAX_ABS_VALUE = 1e30


def _to_number(token: str) -> float | None:
    """把界 token 转成数值；无效或超出合理量级时返回 None。

    覆盖实测写法：10^5 / 10^{9} / 2\\cdot10^5 / 2\\times10^5 / 2*10^5 / 1e5 / 100000
    """
    t = token.strip()
    if not t:
        return None
    # 先试"乘数 × 10 的幂"
    m = re.match(rf"^{_MULT}{_POW10}$", t)
    if m:
        exp = int(m.group(2))
        if exp > _MAX_EXPONENT:
            return None
        mult = float(m.group(1)) if m.group(1) else 1.0
        return mult * (10.0 ** exp)
    # 纯 10 的幂
    m = re.match(rf"^{_POW10}$", t)
    if m:
        exp = int(m.group(1))
        if exp > _MAX_EXPONENT:
            return None
        return 10.0 ** exp
    # 纯数字（含 1e5）
    t2 = t.replace(" ", "")
    m = re.match(rf"^{_NUM}$", t2)
    if m:
        try:
            v = float(m.group(0))
        except (ValueError, OverflowError):
            return None
        if not np_isfinite(v) or abs(v) > _MAX_ABS_VALUE:
            return None
        return v
    return None


def np_isfinite(v: float) -> bool:
    """不引入 numpy 依赖的有限性判断。"""
    return v == v and v not in (float("inf"), float("-inf"))


def _clean_var(v: str) -> str:
    """a_{i} / a_i -> a_i；用于匹配 n 与 t。"""
    v = v.strip()
    v = re.sub(r"_\{(\w+)\}", r"_\1", v)
    return v


def parse_constraints(text: str) -> dict[str, Any]:
    """从（Input 规范）文本解析约束规模特征——冷启动场景的核心特征。

    实测记法（Codeforces 现代题面）：
        `$$$ 1 \\le t \\le 1000 $$$`
        `$$$ (1 \\le a, b, c \\le 10^9) $$$`
        `$$$ 1 \\le n \\le 2 \\cdot 10^5 $$$`

    返回：
        constraint_n_upper   所有非 t 变量上界的**最大值**（最保守的规模估计）
        multi_test_count     t 的上界；无 t 时置 1（表示单组）
        has_multi_test       是否出现 t
        constraint_raw       命中的约束片段（供人工复核，写入报告）
    """
    if not text:
        return {"constraint_n_upper": None, "constraint_max_upper": None,
                "constraint_has_n": 0, "constraint_hits": 0,
                "multi_test_count": 1, "has_multi_test": 0, "constraint_raw": ""}

    matches: list[tuple[float, list[str], str]] = []
    for m in _RE_CHAIN.finditer(text):
        hi = _to_number(m.group("hi"))
        if hi is None:
            continue
        vars_ = [_clean_var(v) for v in m.group("vars").split(",")]
        matches.append((hi, vars_, m.group(0)))
    if not matches:
        for m in _RE_UPPER_ONLY.finditer(text):
            hi = _to_number(m.group("hi"))
            if hi is None:
                continue
            vars_ = [_clean_var(v) for v in m.group("vars").split(",")]
            matches.append((hi, vars_, m.group(0)))

    if not matches:
        return {"constraint_n_upper": None, "constraint_max_upper": None,
                "constraint_has_n": 0, "constraint_hits": 0,
                "multi_test_count": 1, "has_multi_test": 0, "constraint_raw": ""}

    # t 的上界 → multi_test_count
    t_vals = [hi for hi, vs, _ in matches if any(v == "t" for v in vs)]
    has_t = 1 if t_vals else 0
    multi_test = int(max(t_vals)) if t_vals else 1

    # n 的上界：优先取变量含 n 的；否则取所有非 t 变量上界的最大值
    n_candidates = [hi for hi, vs, _ in matches if any(v.startswith("n") for v in vs)]
    if n_candidates:
        n_upper = max(n_candidates)
        has_n = 1
    else:
        others = [hi for hi, vs, _ in matches if not any(v == "t" for v in vs)]
        n_upper = max(others) if others else None
        has_n = 0

    # 兜底规模：约束段中出现的所有上界的最大值（不区分变量名）。
    # 用途：有的题面只写 "1 ≤ t ≤ 1000" 而把 n 的范围写在散文里，
    # 此时 constraint_n_upper 为 None，但"题目规模量级"仍然有信息。
    # 保留该列是为了让特征工程阶段能自行取舍，而不是在这里替它决定。
    all_upper = max(hi for hi, _, _ in matches)

    raw = " | ".join(r for _, _, r in matches[:6])
    return {
        "constraint_n_upper": float(n_upper) if n_upper is not None else None,
        "constraint_max_upper": float(all_upper),
        "constraint_has_n": has_n,
        "constraint_hits": len(matches),
        "multi_test_count": multi_test,
        "has_multi_test": has_t,
        "constraint_raw": raw[:500],
    }


# ================================================================ 题面清洗

#: ⚠ 题面清洗的唯一实现在 `vectorizers.clean_statement`。
#: 这里**不再重复实现一份**——两处实现迟早会漂移，而"清洗口径"是要写进正文的。
#: `fetchers` 只负责把 HTML 拆成 raw_html / preamble / input_spec / output_spec，
#: 以及删除 sample-test 与 note 块；LaTeX 归一化与大小写统一交给 vectorizers。


# ================================================================ HTTP 客户端


@dataclass
class FetchStats:
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    seconds: float = 0.0


class RateLimiter:
    """线程安全的固定间隔限速器。

    `min_interval` 是**调用方给定的礼貌下限**（不是官方值）——
    官方"1 次/2 秒"只约束 API，HTML 端点不适用，因此下限由构造方按端点传入。
    """

    def __init__(self, min_interval: float, jitter: float = 0.25,
                 floor: float = 0.5):
        if min_interval < floor:
            raise ValueError(
                f"自设间隔 {min_interval}s 低于容忍下限 {floor}s，会危及站点与自身"
            )
        self.min_interval = min_interval
        self.jitter = jitter
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            need = self.min_interval - elapsed
            if need > 0:
                time.sleep(need + random.uniform(0, self.jitter))
            self._last = time.monotonic()


class CodeforcesClient:
    """Codeforces 抓取客户端：限速 + 重试 + 磁盘缓存 + 断点续爬。"""

    def __init__(
        self,
        cache_dir: Path | str,
        api_interval: float = DEFAULT_API_INTERVAL_S,
        html_interval: float = DEFAULT_HTML_INTERVAL_S,
        max_retries: int = 5,
        timeout: float = 45.0,
        verbose: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # API 端必须遵守官方 1 次/2 秒；此处再断言一次，防止调用方传入更激进的间隔
        if api_interval < OFFICIAL_MIN_INTERVAL_S:
            raise ValueError(
                f"API 自设间隔 {api_interval}s 小于官方下限 "
                f"{OFFICIAL_MIN_INTERVAL_S}s（apiHelp Introduction）"
            )
        self.api_limiter = RateLimiter(api_interval, floor=OFFICIAL_MIN_INTERVAL_S)
        self.html_limiter = RateLimiter(html_interval, floor=MIN_HTML_INTERVAL_S)
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.stats = FetchStats()

    # ------------------------------------------------------------ 底层

    def _cached_path(self, kind: str, name: str) -> Path:
        d = self.cache_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        return d / f"{safe}.json"

    def _http_get(self, url: str, limiter: RateLimiter) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            limiter.wait()
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9",
                })
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                self.stats.requests += 1
                return raw.decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                last_err = e
                # 429 / 5xx 值得退避重试；4xx（除 429）多为永久错误，快速失败
                if e.code == 429 or 500 <= e.code < 600:
                    backoff = min(60.0, 2.0 ** attempt * 3.0) + random.uniform(0, 2)
                    if self.verbose:
                        print(f"    [HTTP {e.code}] 退避 {backoff:.1f}s 后重试 "
                              f"({attempt+1}/{self.max_retries})")
                    time.sleep(backoff)
                    self.stats.retries += 1
                    continue
                raise
            except Exception as e:  # 网络中断/超时
                last_err = e
                backoff = min(60.0, 2.0 ** attempt * 2.0) + random.uniform(0, 1.5)
                if self.verbose:
                    print(f"    [{type(e).__name__}] 退避 {backoff:.1f}s 后重试 "
                          f"({attempt+1}/{self.max_retries})")
                time.sleep(backoff)
                self.stats.retries += 1
        self.stats.failures += 1
        raise RuntimeError(f"重试 {self.max_retries} 次仍失败: {url}") from last_err

    def _api(self, method: str, **params: Any) -> Any:
        q = urllib.parse.urlencode(params, doseq=True)
        url = f"{API_BASE}/{method}" + (f"?{q}" if q else "")
        cache = self._cached_path("api", f"{method}_{q}" if q else method)
        if cache.exists():
            self.stats.cache_hits += 1
            return json.loads(cache.read_text(encoding="utf-8"))
        text = self._http_get(url, self.api_limiter)
        data = json.loads(text)
        if data.get("status") != "OK":
            raise RuntimeError(f"API 返回非 OK: {data.get('comment', data)}")
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    # ------------------------------------------------------------ 公开接口

    def fetch_problemset(self) -> dict[str, Any]:
        """`problemset.problems`：题目列表 + 累计统计。"""
        return self._api("problemset.problems")

    def fetch_contest_list(self) -> list[dict[str, Any]]:
        """`contest.list`：取 startTimeSeconds 以支撑时间划分。"""
        return self._api("contest.list")["result"]

    def fetch_statement_html(self, contest_id: int, index: str,
                             use_cache: bool = True) -> str | None:
        """抓题目页 HTML（题面 API 不提供）。失败返回 None，由调用方决定是否跳过。"""
        cache = self._cached_path("html", f"{contest_id}_{index}")
        if use_cache and cache.exists():
            self.stats.cache_hits += 1
            return cache.read_text(encoding="utf-8")
        url = PROBLEM_URL.format(cid=contest_id, index=index)
        try:
            html = self._http_get(url, self.html_limiter)
        except Exception as e:
            if self.verbose:
                print(f"    [skip] {contest_id}{index} 抓取失败: {type(e).__name__}")
            return None
        cache.write_text(html, encoding="utf-8")
        return html

    def fetch_contest_standings(self, contest_id: int,
                                use_disk_cache: bool = False) -> dict[str, Any] | None:
        """`contest.standings`（仅 S1' 需要）。

        ⚠ 实测结论（大纲二·（七））：
        1. **常规比赛无需翻页**——匿名 GET 且仅带 contestId 时一次返回完整榜单
           （实测 contestId=2253 单次 13709 行未截断）。
           官方文档明确：常规比赛的公开模式只接受且仅接受 contestId 一个查询参数，
           **多传参数会失败**，因此这里只传 contestId。
        2. `problemResults[]` **没有 `solved` 布尔字段**，字段为
           `points`/`rejectedAttemptCount`/`type`/`bestSubmissionTimeSeconds`。
           ICPC 计分下**通过判定应写为 `points > 0`**，
           **不要用 `points == 该题满分`**——CF 存在按时间递减的部分分
           （实测 contestId=2269 的 A 题出现 203 档不同分值、恰得满分者 0 人）。

        ⚠ `use_disk_cache` 默认 **False**：单场响应实测最大 8.13 MB，
        全量 1976 场的原始 JSON 可达数 GB，而本机 C 盘余量紧张（约 19 GB）。
        因此 S1' 走**流式聚合**路径，不落原始响应（见 `aggregate_standings`）。
        注意这与"题面缓存只存原始 HTML"的取舍不同：题面必须留原文以便改解析口径，
        而榜单的价值只在聚合量，留原文反而有撑爆磁盘的风险。
        """
        method = "contest.standings"
        q = f"contestId={contest_id}"
        url = f"{API_BASE}/{method}?{q}"
        if use_disk_cache:
            cache = self._cached_path("api", f"{method}_{q}")
            if cache.exists():
                self.stats.cache_hits += 1
                return json.loads(cache.read_text(encoding="utf-8"))
        text = self._http_get(url, self.api_limiter)
        data = json.loads(text)
        if data.get("status") != "OK":
            raise RuntimeError(f"API 返回非 OK: {data.get('comment', data)}")
        if use_disk_cache:
            cache = self._cached_path("api", f"{method}_{q}")
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data


# ================================================================ S1' 赛时统计聚合


def aggregate_standings(payload: dict[str, Any],
                        contest_id: int) -> dict[str, dict[str, Any]]:
    """把一场比赛的榜单压成"每题一条赛时统计"。

    为什么必须流式聚合、不留原始响应
    --------------------------------
    实测单场响应 2.8–19.9 s、体积最大 **8.13 MB**；全量 1976 场的原始 JSON
    可达数 GB，而本机 C 盘余量仅约 19 GB。榜单的唯一价值就是这个聚合量，
    因此抓完即弃原始响应。

    产出字段（按 problem index）：
        solved_in_contest        赛时通过人数（判据 `points > 0`）
        attempts_in_contest      赛时总尝试（提交）次数
        rejected_attempts        赛时被拒次数之和
        first_solve_seconds      首解时间（秒，该题最早 `bestSubmissionTimeSeconds`）
        acceptance_rate          通过人数 / 参赛人数（真·通过率，主实验无法派生）
        n_participants           参赛人数（有任一题目结果的选手数）

    ⚠ 通过判定必须用 `points > 0`：CF 存在按时间递减的部分分，
    用 `points == 满分` 会漏掉绝大多数通过（实测某题恰得满分者 0 人）。
    """
    result = payload.get("result", {})
    rows = result.get("rows", [])
    problems = result.get("problems", [])
    n_participants = len(rows)

    acc: dict[str, dict[str, Any]] = {}
    for j, p in enumerate(problems):
        idx = str(p.get("index", ""))
        if not idx:
            continue
        acc[idx] = {
            "contest_id": int(contest_id),
            "problem_index": idx,
            "solved_in_contest": 0,
            "attempts_in_contest": 0,
            "rejected_attempts": 0,
            "first_solve_seconds": None,
            "n_participants": n_participants,
        }

    for r in rows:
        for j, pr in enumerate(r.get("problemResults", [])):
            if j >= len(problems):
                continue
            idx = str(problems[j].get("index", ""))
            if idx not in acc:
                continue
            a = acc[idx]
            pts = pr.get("points", 0) or 0
            rej = int(pr.get("rejectedAttemptCount", 0) or 0)
            bst = pr.get("bestSubmissionTimeSeconds")
            # 尝试次数 = 通过次数(0/1) + 被拒次数；无任何提交时 pts==0 且 rej==0
            if pts > 0:
                a["solved_in_contest"] += 1
                a["attempts_in_contest"] += 1 + rej
                if bst is not None:
                    bst = int(bst)
                    if a["first_solve_seconds"] is None or bst < a["first_solve_seconds"]:
                        a["first_solve_seconds"] = bst
            elif rej > 0:
                a["attempts_in_contest"] += rej
            a["rejected_attempts"] += rej

    for idx, a in acc.items():
        denom = max(1, a["n_participants"])
        a["acceptance_rate"] = a["solved_in_contest"] / denom
        a["problem_key"] = f"{contest_id}-{idx}"
    return acc


def crawl_in_contest_stats(
    client: CodeforcesClient,
    contest_ids: Sequence[int],
    out_path: Path | str,
    flush_every: int = 10,
    max_items: int | None = None,
) -> Iterator[dict[str, Any]]:
    """逐场抓 `contest.standings` 并**流式聚合**成 S1' 赛时统计（可断点续爬）。

    产出 jsonl 每行一个 (contest_id, problem_index) 的赛时统计。
    已在 out_path 中的场次会被跳过，中断后重跑不会重复抓。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["contest_id"]))
                except Exception:
                    continue

    todo = [c for c in contest_ids if c not in done]
    if max_items is not None:
        todo = todo[:max_items]
    print(f"  S1' 抓取：总场次 {len(contest_ids)}，已完成 {len(done)}，本次待抓 {len(todo)}")

    with out_path.open("a", encoding="utf-8") as f:
        for n, cid in enumerate(todo, 1):
            try:
                payload = client.fetch_contest_standings(cid, use_disk_cache=False)
                agg = aggregate_standings(payload, cid)
            except Exception as e:
                # 单场失败（如 gym 场次返回 400）不中断全线，记一条错误行后继续
                f.write(json.dumps({"contest_id": int(cid), "ok": False,
                                    "error": f"{type(e).__name__}: {str(e)[:120]}"},
                                   ensure_ascii=False) + "\n")
                if n % flush_every == 0:
                    f.flush(); os.fsync(f.fileno())
                continue
            for idx in sorted(agg):
                rec = dict(agg[idx]); rec["ok"] = True
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                yield rec
            if n % flush_every == 0:
                f.flush(); os.fsync(f.fileno())
                print(f"    {n}/{len(todo)} 场已抓（请求 {client.stats.requests}，"
                      f"重试 {client.stats.retries}，失败 {client.stats.failures}）")
        f.flush(); os.fsync(f.fileno())


def load_in_contest_stats(path: Path | str) -> dict[str, dict[str, Any]]:
    """读回 S1' 赛时统计，键为 problem_key。"""
    out: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("ok") and rec.get("problem_key"):
                out[rec["problem_key"]] = rec
    return out


# ================================================================ 题面抓取编排


def iter_statements(
    client: CodeforcesClient,
    problems: Sequence[tuple[int, str]],
    disk_path: Path | str,
    flush_every: int = 20,
    max_items: int | None = None,
) -> Iterator[dict[str, Any]]:
    """逐题抓题面并追加写入 jsonl，支持断点续爬。

    ★ 设计决定：jsonl **只缓存原始 HTML**，不在抓取阶段做解析。
    理由：解析口径（section 切分、上标转换、模板短语）会随研究推进而修正，
    若把解析结果写死进缓存，每改一次解析逻辑都得**重新抓一遍上万个页面**。
    只存原始 HTML 后，改解析只需重跑建库阶段（`--skip-crawl`）。

    已存在于 `disk_path` 的键会被跳过（进程可随时中断后重跑）。
    每 `flush_every` 条 fsync 一次，避免中断时丢进度。
    """
    disk_path = Path(disk_path)
    disk_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if disk_path.exists():
        with disk_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("ok"):
                    done.add(rec["problem_key"])

    todo = [(c, i) for c, i in problems if f"{c}-{i}" not in done]
    if max_items is not None:
        todo = todo[:max_items]
    print(f"  题面抓取：总计 {len(problems)} 题，已完成 {len(done)} 题，"
          f"本次待抓 {len(todo)} 题")

    with disk_path.open("a", encoding="utf-8") as f:
        for n, (cid, idx) in enumerate(todo, 1):
            html_doc = client.fetch_statement_html(cid, idx)
            if html_doc is None:
                rec: dict[str, Any] = {"problem_key": f"{cid}-{idx}", "ok": False}
            else:
                rec = {"problem_key": f"{cid}-{idx}", "ok": True, "raw_html": html_doc}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % flush_every == 0:
                f.flush()
                os.fsync(f.fileno())
                print(f"    {n}/{len(todo)} 已抓（累计请求 {client.stats.requests}，"
                      f"重试 {client.stats.retries}，失败 {client.stats.failures}）")
            yield rec
        f.flush()
        os.fsync(f.fileno())


def load_statements(disk_path: Path | str) -> dict[str, dict[str, Any]]:
    """读回 jsonl 抓取结果，返回 {problem_key: record}。"""
    out: dict[str, dict[str, Any]] = {}
    p = Path(disk_path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out[rec["problem_key"]] = rec
    return out
