"""Context budgeting helpers: keep the model aware of the WHOLE conversation inside a small window.

Nothing here calls a model. Everything is deterministic text work, so it costs no extra latency:

* ``sanitize_chart_state`` / ``chart_summary`` / ``chart_tool_messages`` – a chart the app drew is
  remembered as structured facts (symbol, timeframe, overlays, the exact levels that were drawn).
  The model sees ONE short factual line (or a tool-call/tool-result pair) instead of the long status
  notice that used to sit in the history and that models copied back as their «answer».
* ``last_chart_state`` – the chart the user is most probably still talking about («ادیتش کن»).
* ``history_digest`` – older turns that no longer fit are condensed into a few lines instead of being
  deleted, so the model keeps the thread of the conversation.
* ``estimate_tokens`` / ``ContextBudget`` / ``guard_request`` – real accounting of what a request costs
  (system prompt incl. memory, tool definitions, history, files, reserved output) against the model's
  ACTUAL window, with compression starting at ~75 % utilisation and a hard stop instead of silently
  sending a request Ollama would truncate.
* ``pinned_digest`` / ``make_checkpoint`` – standing instructions and unfinished work survive compression
  and can be restored after an interruption.
"""
import json
import math
import re

MAX_CHART_STATE_CHARS = 3000
OVERLAY_ORDER = ("scenario", "swings", "fvg")
_SYMBOL = re.compile(r"^[A-Za-z0-9._#-]{1,20}$")
_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
_NAMES = {"scenario": "فلش سناریو", "swings": "سقف/کف", "fvg": "FVG"}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _numbers(values, limit):
    if not isinstance(values, (list, tuple)):
        return []
    found = [_number(v) for v in values[:limit]]
    return [v for v in found if v is not None]


def sanitize_chart_state(raw):
    """Whitelist what the browser sends back about a chart (it is only a memory aid, never trusted)."""
    if not isinstance(raw, dict):
        return None
    symbol = raw.get("symbol")
    timeframe = str(raw.get("timeframe", "")).upper()
    if not isinstance(symbol, str) or not _SYMBOL.match(symbol) or timeframe not in _TIMEFRAMES:
        return None
    state = {"symbol": symbol.upper(), "timeframe": timeframe,
             "theme": "light" if raw.get("theme") == "light" else "dark"}
    wanted = raw.get("overlays")
    overlays = [name for name in OVERLAY_ORDER if isinstance(wanted, (list, tuple)) and name in wanted]
    scenario = raw.get("scenario")
    if isinstance(scenario, dict) and scenario.get("direction") in ("long", "short"):
        clean = {"direction": scenario["direction"]}
        for key in ("prob", "entry", "stop", "target1", "target2"):
            value = _number(scenario.get(key))
            if value is not None:
                clean[key] = value
        state["scenario"] = clean
        if "scenario" not in overlays:
            overlays.append("scenario")
    swings = raw.get("swings")
    if isinstance(swings, dict):
        clean = {"highs": _numbers(swings.get("highs"), 6), "lows": _numbers(swings.get("lows"), 6)}
        if clean["highs"] or clean["lows"]:
            state["swings"] = clean
    gaps = raw.get("fvgs")
    if isinstance(gaps, list):
        clean_gaps = []
        for gap in gaps[:6]:
            if isinstance(gap, dict) and gap.get("kind") in ("bull", "bear"):
                top, bottom = _number(gap.get("top")), _number(gap.get("bottom"))
                if top is not None and bottom is not None:
                    clean_gaps.append({"kind": gap["kind"], "top": top, "bottom": bottom,
                                       "filled": bool(gap.get("filled"))})
        if clean_gaps:
            state["fvgs"] = clean_gaps
    if raw.get("market") in ("open", "closed", "stale", "unknown"):
        state["market"] = raw["market"]
    state["overlays"] = [name for name in OVERLAY_ORDER if name in overlays]
    return state


def _price(value):
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text or "0"


def chart_summary(state, limit=520):
    """One factual Persian paragraph describing what the app drew (what the model remembers)."""
    if not state:
        return ""
    parts = [f"تصویر نمودار {state['symbol']} ({state['timeframe']}، زمینهٔ "
             f"{'سفید' if state.get('theme') == 'light' else 'تیره'}) توسط برنامه از دادهٔ MT5 ساخته و به کاربر نمایش داده شد."]
    drawn = []
    scenario = state.get("scenario")
    if scenario:
        side = "صعودی" if scenario["direction"] == "long" else "نزولی"
        levels = "، ".join(f"{label} {_price(scenario[key])}" for key, label in
                           (("entry", "ورود"), ("stop", "SL"), ("target1", "TP1"), ("target2", "TP2")) if key in scenario)
        prob = f" احتمال تجربی {scenario['prob'] * 100:.0f}٪" if "prob" in scenario else ""
        drawn.append(f"فلش سناریوی {side}{prob}" + (f" ({levels})" if levels else ""))
    elif "scenario" in state.get("overlays", ()):
        drawn.append("فلش سناریو (جزئیات ثبت نشد)")
    swings = state.get("swings")
    if swings:
        if swings.get("highs"):
            drawn.append("سقف‌ها " + "، ".join(_price(v) for v in swings["highs"]))
        if swings.get("lows"):
            drawn.append("کف‌ها " + "، ".join(_price(v) for v in swings["lows"]))
    elif "swings" in state.get("overlays", ()):
        drawn.append("سقف/کف (موردی پیدا نشد)")
    gaps = state.get("fvgs")
    if gaps:
        drawn.append("FVG " + "، ".join(
            f"{'صعودی' if g['kind'] == 'bull' else 'نزولی'} {_price(g['bottom'])}–{_price(g['top'])}" for g in gaps[:3]))
    elif "fvg" in state.get("overlays", ()):
        drawn.append("FVG (موردی پیدا نشد)")
    if drawn:
        parts.append("روی تصویر کشیده شد: " + "؛ ".join(drawn) + ".")
    if state.get("market") == "closed":
        parts.append("بازار بسته بود.")
    text = " ".join(parts)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def chart_tool_messages(state):
    """The chart as an Ollama tool call + tool result, so tool-calling models see «chart requests ->
    call the tool» in their own history instead of a text answer they can imitate without drawing."""
    arguments = {"symbol": state["symbol"], "timeframe": state["timeframe"], "theme": state.get("theme", "dark")}
    if state.get("overlays"):
        arguments["overlays"] = list(state["overlays"])
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "render_forex_chart", "arguments": arguments}}]},
        {"role": "tool", "tool_name": "render_forex_chart", "content": chart_summary(state)},
    ]


def last_chart_state(messages, max_gap=2):
    """State of the chart the user is still talking about.

    Only the newest assistant message counts, and only when nothing but at most ``max_gap - 1``
    other messages follow it; an analysis several turns later is not an «edit the image» request.
    """
    for back, message in enumerate(reversed(messages), 1):
        if message.get("role") != "assistant":
            continue
        return sanitize_chart_state(message.get("chart")) if back <= max_gap else None
    return None


def _one_line(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# (user chars, assistant chars) per digest line, tried from most to least detailed before any line is dropped
_DIGEST_LEVELS = ((150, 120), (110, 70), (80, 40), (60, 0))


def history_digest(messages, cap):
    """Condense turns that no longer fit into a short list instead of deleting them.

    Zero model calls: the first characters of each turn, charts as their one-line summary. What the user
    said or asked matters most, so when the digest is too long the assistant lines are shortened (then left
    out, charts excepted) and the user lines clipped BEFORE any turn is dropped; only then do the oldest
    lines give way to the newest.
    """
    if cap < 200:
        return ""
    header = "خلاصهٔ خودکار بخش‌های قدیمی‌تر همین گفت‌وگو (فقط برای حفظ پیوستگی؛ آنچه کاربر قبلاً خواسته یا گفته):\n"
    room = cap - len(header)

    def build(user_chars, assistant_chars):
        lines = []
        for message in messages:
            role = message.get("role")
            if role == "system" or message.get("tool_calls"):
                continue
            content = message.get("content")
            if role == "user":
                text = _one_line(content, user_chars)
                if text:
                    lines.append("• کاربر: " + text)
            elif role in ("assistant", "tool"):
                state = sanitize_chart_state(message.get("chart"))
                if state:
                    text = _one_line(chart_summary(state, 260), max(100, assistant_chars * 2))
                elif role == "tool":
                    text = _one_line(content, max(100, assistant_chars * 2))
                else:
                    text = _one_line(content, assistant_chars) if assistant_chars else ""
                if text:
                    lines.append("• دستیار: " + text)
        return lines

    lines = []
    for user_chars, assistant_chars in _DIGEST_LEVELS:
        lines = build(user_chars, assistant_chars)
        if sum(len(line) + 1 for line in lines) <= room:
            break
    if not lines:
        return ""
    kept, used = [], 0
    for line in reversed(lines):
        if used + len(line) + 1 > room:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return ""
    omitted = len(lines) - len(kept)
    prefix = f"(+{omitted} پیام قدیمی‌تر)\n" if omitted and used + 30 <= room else ""
    return header + prefix + "\n".join(reversed(kept))


# ----------------------------------------------------------------------------- token accounting
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_CJK = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")
_CODEISH = re.compile(r"[{}\[\]();=<>/\\_#*`|]")
COMPRESS_AT = 0.75        # start condensing history when the window is ~75 % used (range asked for: 70-80 %)
COMPRESS_TO = 0.60        # ... and condense down to about this level
HARD_LIMIT = 0.97         # never send a request estimated above this share of the window
MIN_CTX = 1024


def estimate_tokens(text):
    """Pessimistic token estimate without a tokenizer (Persian ≈ 2 chars/token, code ≈ 3, Latin prose ≈ 3.8).

    It deliberately over-counts a little: over-estimating only costs a slightly earlier compression, while
    under-estimating makes Ollama silently cut the start of the prompt.
    """
    if not text:
        return 0
    text = str(text)
    arabic = len(_ARABIC_SCRIPT.findall(text))
    cjk = len(_CJK.findall(text))
    rest = len(text) - arabic - cjk
    code = len(_CODEISH.findall(text))
    tokens = arabic / 2.0 + cjk / 1.1 + (rest - code) / 3.8 + code / 1.6
    return int(math.ceil(tokens)) + 1


def estimate_messages_tokens(messages, per_message=6):
    total = 0
    for message in messages or ():
        total += per_message + estimate_tokens(message.get("content", ""))
        for call in message.get("tool_calls") or ():
            total += estimate_tokens(json.dumps(call, ensure_ascii=False))
        total += 800 * len(message.get("images") or ())     # an image costs a roughly fixed number of tokens
    return total


def estimate_tools_tokens(tools):
    return estimate_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0


def chars_per_token(sample):
    """Measured characters-per-token of ``sample`` (used to turn a token budget into the char budget the
    prompt fitter works with, so a Persian chat and an English code chat get different char budgets)."""
    sample = str(sample or "")
    if len(sample) < 40:
        return 2.1
    return max(1.4, min(4.0, len(sample) / max(1, estimate_tokens(sample) - 1)))


def capacity_from_show(info):
    """Context window a model supports, from an Ollama ``/api/show`` response (None when unknown)."""
    if not isinstance(info, dict):
        return None
    model_info = info.get("model_info")
    best = None
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if str(key).endswith(".context_length") and isinstance(value, int) and not isinstance(value, bool) and value > 0:
                best = value if best is None else min(best, value)
    return best


def effective_context(requested_ctx, capacity):
    """The ``num_ctx`` to ask Ollama for: what the app wants, but never more than the model can hold."""
    requested = max(MIN_CTX, int(requested_ctx))
    if capacity:
        requested = min(requested, max(MIN_CTX, int(capacity)))
    return requested


class ContextBudget:
    """What one request costs inside a window of ``num_ctx`` tokens."""

    def __init__(self, num_ctx, num_predict, system_tokens=0, tools_tokens=0, history_tokens=0, file_tokens=0,
                 memory_tokens=0):
        self.num_ctx = max(MIN_CTX, int(num_ctx))
        self.num_predict = max(0, int(num_predict))
        self.system_tokens = int(system_tokens)
        self.tools_tokens = int(tools_tokens)
        self.history_tokens = int(history_tokens)
        self.file_tokens = int(file_tokens)
        self.memory_tokens = int(memory_tokens)

    @property
    def input_tokens(self):
        return self.system_tokens + self.tools_tokens + self.history_tokens + self.file_tokens + self.memory_tokens

    @property
    def used(self):
        return self.input_tokens + self.num_predict

    @property
    def utilization(self):
        return self.used / self.num_ctx

    @property
    def should_compress(self):
        return self.utilization >= COMPRESS_AT

    @property
    def fits(self):
        return self.used <= self.num_ctx * HARD_LIMIT

    @property
    def input_limit(self):
        """Input tokens that may be used before compression must start (soft limit)."""
        return max(0, min(self.num_ctx - self.num_predict, int(self.num_ctx * COMPRESS_AT)))

    @property
    def input_target(self):
        return max(0, min(self.num_ctx - self.num_predict, int(self.num_ctx * COMPRESS_TO)))

    def as_dict(self):
        return {"num_ctx": self.num_ctx, "reserved_output": self.num_predict, "system": self.system_tokens,
                "tools": self.tools_tokens, "history": self.history_tokens, "files": self.file_tokens,
                "memory": self.memory_tokens, "used": self.used, "utilization": round(self.utilization, 3),
                "compress": self.should_compress, "fits": self.fits}


def guard_request(messages, num_ctx, num_predict, tools=None):
    """Final check before a request goes to Ollama.

    Returns ``(ok, budget)``.  ``ok`` is False when the request would not fit the window even though Ollama
    would still accept it (and quietly drop the oldest part of the prompt – an incomplete request).
    """
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    budget = ContextBudget(num_ctx, num_predict, system_tokens=estimate_messages_tokens(system),
                           tools_tokens=estimate_tools_tokens(tools), history_tokens=estimate_messages_tokens(rest))
    return budget.fits, budget


# ----------------------------------------------------------------------------- what must never be forgotten
_INSTRUCTION_CUE = re.compile(
    r"(?:همیشه|هرگز|هیچ\s*وقت|هیچ‌وقت|از\s*این\s*به\s*بعد|حتماً|حتما|نباید|فقط\s+(?:با|به|از)|قانون|دستور|"
    r"یادت\s*باشه|\balways\b|\bnever\b|\bmust\b|\bdon't\b|\bdo not\b)", re.I)
_TASK_CUE = re.compile(
    r"(?:ادامه\s*بده|هنوز|بعداً|بعدا|مرحلهٔ?\s*بعد|باقی[\u200c]?مانده|ناتمام|بعدش|\bTODO\b|\bnext step|\bremaining\b|\bstill\b|"
    r"قدم\s*بعد|کار\s*بعدی|پروژه|فایل\s+\S+|\b\w+\.(?:py|js|ts|dart|kt|java|html|css|md|docx|pdf)\b)", re.I)


def pinned_digest(messages, cap=700, max_lines=12):
    """Standing instructions and unfinished work found in ``messages`` (newest wins), as one short block.

    History compression drops/condenses old turns; this keeps the few sentences that change how the model must
    behave («همیشه کوتاه جواب بده») or what is still open («فایل app.py هنوز باقی مانده») verbatim.
    """
    if cap < 120:
        return ""
    found, seen = [], set()
    for message in reversed(messages or ()):
        if message.get("role") != "user":
            continue
        for sentence in re.split(r"(?<=[!؟?؛])\s+|(?<=\.)\s+|\n+", str(message.get("content") or "")):
            line = _one_line(sentence, 220)
            if len(line) < 8 or line in seen:
                continue
            if _INSTRUCTION_CUE.search(line):
                found.append(("دستور", line))
            elif _TASK_CUE.search(line) and len(line) <= 220:
                found.append(("کار", line))
            else:
                continue
            seen.add(line)
            if len(found) >= max_lines * 2:
                break
    header = "دستورها و کارهای باز گفت‌وگوی قبلی که نباید فراموش شوند:\n"
    room = cap - len(header)
    kept, used = [], 0
    for kind, line in found:                        # newest first, instructions before open tasks
        for wanted in ("دستور", "کار"):
            if kind != wanted:
                continue
            entry = f"• {line}"
            if used + len(entry) + 1 <= room and len(kept) < max_lines:
                kept.append((wanted, entry))
                used += len(entry) + 1
    if not kept:
        return ""
    kept.sort(key=lambda item: 0 if item[0] == "دستور" else 1)
    return header + "\n".join(entry for _, entry in kept)


def make_checkpoint(messages, goal="", max_pinned_chars=1200):
    """A small JSON-safe snapshot that lets a conversation/task continue after an interruption."""
    users = [m for m in messages or () if m.get("role") == "user"]
    return {
        "version": 1,
        "goal": _one_line(goal or (users[0].get("content") if users else ""), 300),
        "last_user": _one_line(users[-1].get("content") if users else "", 600),
        "pinned": pinned_digest(messages, max_pinned_chars),
        "digest": history_digest(list(messages or ())[:-2], 1400),
        "turns": len(list(messages or ())),
    }


def checkpoint_system_text(checkpoint):
    """The system-prompt block that restores a checkpoint (empty when there is nothing to restore)."""
    if not isinstance(checkpoint, dict):
        return ""
    parts = ["[ادامهٔ کار بعد از وقفه — خلاصهٔ حالت قبلی]"]
    if checkpoint.get("goal"):
        parts.append("هدف: " + str(checkpoint["goal"]))
    if checkpoint.get("pinned"):
        parts.append(str(checkpoint["pinned"]))
    if checkpoint.get("digest"):
        parts.append(str(checkpoint["digest"]))
    if checkpoint.get("last_user"):
        parts.append("آخرین درخواست کاربر: " + str(checkpoint["last_user"]))
    return "\n".join(parts) if len(parts) > 1 else ""
