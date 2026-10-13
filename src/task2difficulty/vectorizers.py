"""文本向量化与文本特征（大纲二·（二））。

铁律：所有 vectorizer / 词表**只在训练集上 fit**，再 transform 验证/测试集。
任何在全量数据上 fit 的写法（包括 `fit_transform(all_text)`）都是泄漏。

包含
----
- `TfidfTextVectorizer`：传统侧文本输入；
- `BertTokenizerWrapper`：深度侧输入（定长 + attention mask）；
- `strip_template_phrases`：文本端 provenance 剥离（大纲四·（四））。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config as C

# ================================================================ 文本清洗


#: LaTeX 归一化映射（大纲二·（二）：\le→<=、\times→*，保留数字）
LATEX_MAP: dict[str, str] = {
    r"\leq": "<=", r"\le": "<=", r"\geq": ">=", r"\ge": ">=",
    r"\times": "*", r"\cdot": "*", r"\div": "/",
    r"\lt": "<", r"\gt": ">", r"\neq": "!=", r"\ne": "!=",
    r"\pm": "+-", r"\sum": "sum", r"\prod": "prod",
    r"\infty": "infinity", r"\ldots": "...", r"\dots": "...",
    r"\rightarrow": "->", r"\to": "->", r"\Rightarrow": "=>",
    r"\mod": "mod", r"\pmod": "mod", r"\bmod": "mod",
    r"\lfloor": "floor(", r"\rfloor": ")", r"\lceil": "ceil(", r"\rceil": ")",
    r"\sqrt": "sqrt", r"\log": "log", r"\ln": "ln",
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\theta": "theta",
    r"\pi": "pi", r"\epsilon": "epsilon", r"\phi": "phi", r"\lambda": "lambda",
    r"\mu": "mu", r"\sigma": "sigma", r"\Delta": "Delta", r"\Omega": "Omega",
}
#: LaTeX 命令与花括号（保留括号内内容）
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_RE_BRACES = re.compile(r"[{}]")
_RE_DOLLAR = re.compile(r"\$+")
#: 样例输入/输出与 Note 段（大纲：纯噪声，必须删除）
_RE_SAMPLE = re.compile(
    r"(?is)(example\s*\d*\s*)?(input|output)\s*(copy)?\s*:.*?(?=(example\s*\d|note\s*:|$))"
)
_RE_NOTE = re.compile(r"(?is)\bnote\s*:.*$")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")


def clean_statement(raw: str, drop_samples: bool = True) -> str:
    """题面清洗流水线（大纲二·（二））——**全文唯一的清洗实现**。

    顺序：剥 HTML → LaTeX 归一化 → 删样例与 Note → 统一小写。

    `drop_samples=False` 时只做 LaTeX/HTML/大小写处理，不删样例段与 Note。
    数据管线应传 `False`：因为 HTML 抽取阶段（`fetchers.extract_statement_parts`）
    已经用配平的方式整块删除了 `sample-test` 与 `note`，此处若再删一遍，
    正则的 `.*$` 会从正文中任何出现 "note" 的地方一路吃到文末，误删正文。
    """
    if not isinstance(raw, str):
        return ""
    s = _RE_HTML_TAG.sub(" ", raw)
    # LaTeX：先处理已知命令，再删剩余命令名，最后去花括号与 $ 符号
    for k, v in LATEX_MAP.items():
        s = s.replace(k, v)
    s = _RE_LATEX_CMD.sub(" ", s)
    s = _RE_BRACES.sub(" ", s)
    s = _RE_DOLLAR.sub(" ", s)
    if drop_samples:
        s = _RE_SAMPLE.sub(" ", s)
        s = _RE_NOTE.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip().lower()
    return s


def strip_template_phrases(text: str, extra: Sequence[str] = ()) -> str:
    """删除题面模板句（大纲四·（四）文本端 provenance 剥离）。

    ⚠ 口径必须写明：这是**"捷径依赖度诊断"，不是泄漏剔除**。
    模板句对新题同样可得，不属于泄漏；但它决定模型是否真在"读懂题面"
    而非抓"easy version / interactive"这类模板。
    """
    phrases = list(C.TEMPLATE_PHRASES) + [p.lower() for p in extra]
    out = text
    for p in phrases:
        out = re.sub(re.escape(p), " ", out, flags=re.IGNORECASE)
    return _RE_WS.sub(" ", out).strip()


def count_template_hits(text: str) -> int:
    """统计题面命中模板句的次数（用于量化"捷径"的存在程度）。"""
    low = text.lower()
    return sum(1 for p in C.TEMPLATE_PHRASES if p.lower() in low)


# ================================================================ TF-IDF


@dataclass
class TfidfTextVectorizer:
    """传统侧文本输入。严格只在训练集 fit。"""

    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, texts: Sequence[str]) -> "TfidfTextVectorizer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._v = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
            **self.params,
        )
        self._v.fit(list(texts))
        self.n_features_ = len(self._v.vocabulary_)
        return self

    def transform(self, texts: Sequence[str]):
        if not hasattr(self, "_v"):
            raise RuntimeError("必须先 fit（且只能用训练集）")
        return self._v.transform(list(texts))

    def fit_transform(self, texts: Sequence[str]):
        return self.fit(texts).transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return getattr(self, "n_features_", 0)


# ================================================================ BERT tokenizer


@dataclass
class BertTokenizerWrapper:
    """深度侧输入：bert-base-uncased 定长编码。

    `max_length` 必须由 `choose_max_length` 依据训练集 token 长度分布决定，
    不能凭感觉定——大纲要求"题面文本长度"进入表 1，
    而 max_length 直接决定显存与可训性（本机 8 GB 显存，桌面已占 5.4 GB）。
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 256

    def fit(self, texts: Sequence[str]) -> "BertTokenizerWrapper":
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        return self

    def encode(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        if not hasattr(self, "_tok"):
            raise RuntimeError("必须先 fit")
        enc = self._tok(
            list(texts), padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="np",
        )
        return {k: np.asarray(v) for k, v in enc.items()}

    @property
    def vocab_size(self) -> int:
        return len(getattr(self, "_tok", None) or [])


@dataclass
class SimpleVocab:
    """TextCNN / BiLSTM 用的朴素词表（从训练集构建）。

    不引入 torchtext（已停止维护），自建词表更可控也更少依赖。
    """

    max_vocab: int = 60000
    min_freq: int = 2
    max_length: int = 400

    def fit(self, texts: Sequence[str]) -> "SimpleVocab":
        from collections import Counter

        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(str(t).split())
        self.token2idx: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for w, c in cnt.most_common(self.max_vocab):
            if c < self.min_freq:
                break
            self.token2idx[w] = len(self.token2idx)
        self.vocab_size_ = len(self.token2idx)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "token2idx"):
            raise RuntimeError("必须先 fit")
        unk = self.token2idx["<unk>"]
        out = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.token2idx.get(w, unk) for w in str(t).split()][: self.max_length]
            if ids:
                out[i, : len(ids)] = ids
        return out

    @property
    def vocab_size(self) -> int:
        return getattr(self, "vocab_size_", 0)


# ================================================================ 辅助


def choose_max_length(texts: Sequence[str], tokenizer: Any,
                      cover: float = 0.95, cap: int = 512) -> dict[str, float]:
    """用训练集题面的 token 长度分布决定 max_length。

    返回分位数统计，供正文说明"为何取该 max_length"。
    """
    lengths: list[int] = []
    sample = list(texts)
    for t in sample:
        lengths.append(len(tokenizer(str(t), truncation=False)["input_ids"]))
    arr = np.asarray(lengths, dtype=np.float64)
    p = float(np.quantile(arr, cover))
    chosen = int(min(cap, max(64, 2 ** int(np.ceil(np.log2(max(64.0, p)))))))
    return {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": p,
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        f"chosen_max_length_for_{int(cover*100)}pct": float(chosen),
    }


def text_length_stats(texts: Sequence[str]) -> dict[str, float]:
    """表 1 的"题均文本长度"：字符数与词数双口径。"""
    chars = np.asarray([len(t) for t in texts], dtype=np.float64)
    words = np.asarray([len(str(t).split()) for t in texts], dtype=np.float64)
    return {
        "char_mean": float(chars.mean()),
        "char_median": float(np.median(chars)),
        "char_p95": float(np.quantile(chars, 0.95)),
        "word_mean": float(words.mean()),
        "word_median": float(np.median(words)),
        "word_p95": float(np.quantile(words, 0.95)),
    }
