"""Context budgeting helpers: keep the model aware of the WHOLE conversation inside a small window.

Nothing here calls a model. Everything is deterministic text work, so it costs no extra latency:

* ``sanitize_chart_state`` / ``chart_summary`` / ``chart_tool_messages`` – a chart the app drew is
  remembered as structured facts (symbol, timeframe, overlays, the exact levels that were drawn).
  The model sees ONE short factual line (or a tool-call/tool-result pair) instead of the long status
  notice that used to sit in the history and that models copied back as their «answer».
* ``last_chart_state`` – the chart the user is most probably still talking about («ادیتش کن»).
* ``history_digest`` – older turns that no longer fit are condensed into a few lines instead of being
  deleted, so the model keeps the thread of the conversation.
"""
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
