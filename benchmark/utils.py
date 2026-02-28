import re, string
from typing import Any, Dict, Iterable
from collections import deque
import threading
from threading import Thread
from math import floor
import json


def tok_cnt(text: str) -> int:
    """
    Approximate token count using whitespace-based word splitting,
    scaled by a token inflation factor (default 1.2).

    Returns at least 1.
    """
    if not text.strip():
        return 1
    word_count = len([tok for tok in text.split() if tok])
    token_estimate = floor(word_count * 1.2)
    return max(token_estimate, 1)


# ── Sentence/Statement Counters ──

def sent_cnt(text: str, mode: str = "qa") -> int:
    """
    Count “sentences” in `text`.
      • mode="sql": count SQL statements separated by semicolons.
      • mode="qa" (default): count punctuation-based sentence boundaries.
    Returns at least 1.
    """
    if mode == "sql":
        # In SQL mode, assume minimum 1 statement even if no semicolon
        return 1
    else:
        count = len(re.findall(r"[.!?…]+", text))
        return max(count, 1)


def chunker(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

def normalize_answer(s: str) -> str:
    """
    Lowercase, remove punctuation/articles, collapse whitespace.
    Handles None inputs gracefully.
    """
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)

def _to_text(x: Any) -> str:
    # Accept either strings or objects like ChatCompletionMessage(content=...)
    if isinstance(x, str):
        return x
    content = getattr(x, "content", None)
    if isinstance(content, str):
        return content
    # Fallback: last resort string conversion
    return "" if x is None else str(x)

def _strip_think_blocks(text: str) -> str:
    # Remove all <think>...</think> blocks (including multiline)
    return _THINK_BLOCK_RE.sub("", text)

def clean_prediction(prediction: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []

    for item in prediction:
        raw = _to_text(item)

        # 0) Remove <think>...</think> blocks + trim again
        ans = _strip_think_blocks(raw).strip()

        # 1) Remove anything after the first '###'
        ans = ans.split("###", 1)[0].strip()

        # 2) If the whole thing is valid JSON, return a one-line JSON string
        try:
            obj = json.loads(ans)
            ans = json.dumps(obj, separators=(",", ":"))  # minified, single line
        except json.JSONDecodeError:
            # 3) Not JSON: remove anything after the first newline
            ans = ans.split("\n", 1)[0].strip()

        cleaned.append(ans)

    return cleaned


def _start_log_tailer(self, max_lines: int = 30):
    """Spawn a daemon thread that reads the process’ output and
    stores the latest `max_lines` in self._log_buf (deque[str])."""
    self._log_buf = deque(maxlen=max_lines)

    def _tail():
        for line in self._launcher_proc.stdout:
            self._log_buf.append(line.rstrip("\n"))
            if self._stop_tail.is_set():
                break

    self._stop_tail = threading.Event()
    self._tail_thread = Thread(target=_tail, daemon=True)
    self._tail_thread.start()
