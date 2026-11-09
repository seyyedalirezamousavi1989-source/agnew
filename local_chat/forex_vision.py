"""Visual read of the chart by the local vision model (Ollama).

The vision model is a *second opinion*, not the source of the numbers: it sees the
rendered candlestick picture and answers a fixed JSON form (structure, patterns, bias).
Its bias is compared with the quantitative lean; agreement or conflict is reported, but
it never changes the percentages (a small local model is far less reliable than the
walk-forward-tested statistics).
"""
import json
import re

COLOR_KEY = (
    "Colour key: green/red candles = up/down; blue line = EMA20; purple line = EMA50; "
    "green dashed lines = support; red dashed lines = resistance; amber dotted line = daily pivot; "
    "white dashed line = current price (its value is printed on the right scale)."
)

VISION_PROMPT = (
    "You are a professional price-action analyst. Look ONLY at this candlestick chart of {symbol} "
    "on the {tf} timeframe (last ~110 closed candles, oldest on the left). " + COLOR_KEY + "\n"
    "Answer with ONE JSON object and nothing else, with exactly these keys:\n"
    '{{"trend": "up|down|range", "structure": "one short sentence on swing highs/lows (HH/HL, LH/LL, channel, range)", '
    '"patterns": ["0-3 chart patterns you can actually see, e.g. double top, triangle, flag, breakout-retest, wedge"], '
    '"price_vs_emas": "above both|below both|between|tangled", '
    '"nearest_reaction_zone": "where price most recently reacted, in words and approximate price", '
    '"bias": "bull|bear|neutral", "confidence": "low|medium|high"}}\n'
    "Do not invent patterns; if the picture is unclear use neutral/low. Do not give trade advice."
)

_JSON = re.compile(r"\{.*\}", re.S)
_ALLOWED = {
    "trend": {"up", "down", "range"},
    "bias": {"bull", "bear", "neutral"},
    "confidence": {"low", "medium", "high"},
    "price_vs_emas": {"above both", "below both", "between", "tangled"},
}


def parse_reply(text):
    """Dict of the allowed fields, or None when the reply holds no usable JSON."""
    if not isinstance(text, str):
        return None
    match = _JSON.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    result = {}
    for key, allowed in _ALLOWED.items():
        value = str(data.get(key, "")).strip().lower()
        result[key] = value if value in allowed else ("neutral" if key == "bias" else
                                                       "low" if key == "confidence" else "")
    for key, limit in (("structure", 220), ("nearest_reaction_zone", 160)):
        value = data.get(key)
        result[key] = value.strip()[:limit] if isinstance(value, str) else ""
    patterns = data.get("patterns")
    result["patterns"] = [str(p).strip()[:60] for p in patterns[:3] if str(p).strip()] if isinstance(patterns, list) else []
    return result


def agreement(quant_lean, vision_bias):
    """'agree' / 'conflict' / 'neutral' between the quantitative lean and the visual bias."""
    mapping = {"up": "bull", "down": "bear"}
    wanted = mapping.get(quant_lean)
    if wanted is None or vision_bias == "neutral":
        return "neutral"
    return "agree" if wanted == vision_bias else "conflict"


def read_chart(chat, model, png_b64, symbol, timeframe):
    """Ask ``model`` to read the chart. ``chat(model, messages, options)`` returns the reply text.

    Returns ``{"ok": True, "read": {...}}`` or ``{"ok": False, "error": "..."}``; never raises.
    """
    messages = [{
        "role": "user",
        "content": VISION_PROMPT.format(symbol=symbol, tf=timeframe),
        "images": [png_b64],
    }]
    try:
        reply = chat(model, messages, {"temperature": 0.1, "num_predict": 450})
    except Exception as exc:  # noqa: BLE001 - the desk must still answer without the picture
        return {"ok": False, "error": f"مدل بینایی پاسخ نداد ({type(exc).__name__})."}
    parsed = parse_reply(reply)
    if parsed is None:
        return {"ok": False, "error": "پاسخ مدل بینایی قابل خواندن نبود؛ تحلیل بدون آن ادامه یافت."}
    return {"ok": True, "read": parsed, "model": model}
