"""Candlestick chart as a PNG, drawn in pure Python (no matplotlib / PIL needed).

The picture is made for the local vision model *and* for the person: dark background,
green/red candles, EMA20 (blue) and EMA50 (purple) lines, support lines (green dashed),
resistance lines (red dashed), pivot (amber dotted) and the current price (white).
Price labels on the right edge are drawn with a tiny built-in digit font, so the model
can read the scale. The default look is dark; ``theme="light"`` gives a white background.
Optional annotations (arrows, markers, horizontal lines, short Latin/digit labels) are
drawn by the app from prices and bar offsets chosen by the model or the user. The colour
key is part of the prompt (see forex_vision).
"""
import math
import struct
import zlib

BG = (17, 20, 26)
GRID = (36, 41, 51)
UP = (38, 166, 126)
DOWN = (226, 78, 82)
EMA_FAST = (70, 150, 255)
EMA_SLOW = (176, 110, 255)
SUPPORT = (60, 200, 120)
RESISTANCE = (240, 90, 90)
PIVOT = (240, 180, 60)
PRICE = (235, 235, 235)
TEXT = (170, 176, 190)
FVG_BULL = (0, 205, 255)     # bullish fair value gap: cyan
FVG_BEAR = (255, 95, 205)    # bearish fair value gap: magenta

# 3x5 glyphs, one string of 15 bits per character (row by row, 1 = pixel)
_GLYPHS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
    " ": "000000000000000",
}

def _rows(*rows):
    return "".join(rows)


_GLYPHS.update({
    "A": _rows("010", "101", "111", "101", "101"), "B": _rows("110", "101", "110", "101", "110"),
    "C": _rows("011", "100", "100", "100", "011"), "D": _rows("110", "101", "101", "101", "110"),
    "E": _rows("111", "100", "110", "100", "111"), "F": _rows("111", "100", "110", "100", "100"),
    "G": _rows("011", "100", "101", "101", "011"), "H": _rows("101", "101", "111", "101", "101"),
    "I": _rows("111", "010", "010", "010", "111"), "J": _rows("001", "001", "001", "101", "010"),
    "K": _rows("101", "101", "110", "101", "101"), "L": _rows("100", "100", "100", "100", "111"),
    "M": _rows("101", "111", "111", "101", "101"), "N": _rows("110", "101", "101", "101", "101"),
    "O": _rows("010", "101", "101", "101", "010"), "P": _rows("110", "101", "110", "100", "100"),
    "Q": _rows("010", "101", "101", "111", "011"), "R": _rows("110", "101", "110", "101", "101"),
    "S": _rows("011", "100", "010", "001", "110"), "T": _rows("111", "010", "010", "010", "010"),
    "U": _rows("101", "101", "101", "101", "111"), "V": _rows("101", "101", "101", "101", "010"),
    "W": _rows("101", "101", "111", "111", "101"), "X": _rows("101", "101", "010", "101", "101"),
    "Y": _rows("101", "101", "010", "010", "010"), "Z": _rows("111", "001", "010", "100", "111"),
    "+": _rows("000", "010", "111", "010", "000"), ":": _rows("000", "010", "000", "010", "000"),
    "/": _rows("001", "001", "010", "100", "100"), "%": _rows("101", "001", "010", "100", "101"),
    "=": _rows("000", "111", "000", "111", "000"),
})
LABEL_CHARS = frozenset(_GLYPHS)

# colour sets: the default (dark) picture is what the vision prompt describes
PALETTES = {
    "dark": {"bg": BG, "grid": GRID, "text": TEXT, "price": PRICE, "tag": (50, 56, 70),
             "up": UP, "down": DOWN, "ema_fast": EMA_FAST, "ema_slow": EMA_SLOW,
             "support": SUPPORT, "resistance": RESISTANCE, "pivot": PIVOT,
             "fvg_bull": FVG_BULL, "fvg_bear": FVG_BEAR, "label_bg": (30, 35, 45),
             "green": (60, 200, 120), "red": (240, 90, 90), "amber": (240, 180, 60),
             "blue": (70, 150, 255), "neutral": (235, 235, 235)},
    "light": {"bg": (255, 255, 255), "grid": (228, 231, 236), "text": (70, 76, 90), "price": (25, 28, 36),
              "tag": (225, 229, 236), "up": (22, 140, 104), "down": (205, 50, 56),
              "ema_fast": (30, 110, 230), "ema_slow": (140, 70, 220), "support": (30, 150, 85),
              "resistance": (210, 60, 60), "pivot": (205, 140, 20), "fvg_bull": (0, 160, 205),
              "fvg_bear": (215, 50, 165), "label_bg": (244, 246, 249),
              "green": (22, 150, 90), "red": (210, 50, 55), "amber": (215, 140, 10),
              "blue": (30, 110, 230), "neutral": (25, 28, 36)},
}
ANNOTATION_COLORS = ("green", "red", "amber", "blue", "neutral")
MAX_ANNOTATIONS = 12


class Canvas:
    def __init__(self, width, height, color=BG):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(color) * (width * height))

    def put(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset:offset + 3] = bytes(color)

    def rect(self, x0, y0, x1, y1, color):
        x0, x1 = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        if x1 < x0 or y1 < y0:
            return
        row = bytes(color) * (x1 - x0 + 1)
        for y in range(y0, y1 + 1):
            start = (y * self.width + x0) * 3
            self.pixels[start:start + len(row)] = row

    def blend_rect(self, x0, y0, x1, y1, color, alpha=0.22):
        """Semi-transparent filled rectangle."""
        x0, x1 = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        if x1 < x0 or y1 < y0:
            return
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                offset = (y * self.width + x) * 3
                for k in range(3):
                    old = self.pixels[offset + k]
                    self.pixels[offset + k] = int(old + (color[k] - old) * alpha)

    def hline(self, y, x0, x1, color, dash=0):
        if not 0 <= y < self.height:
            return
        for x in range(max(0, x0), min(self.width - 1, x1) + 1):
            if dash and (x // dash) % 2:
                continue
            self.put(x, y, color)

    def line(self, x0, y0, x1, y1, color):
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.put(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def thick_line(self, x0, y0, x1, y1, color, width=3):
        for ox in range(-(width // 2), width - width // 2):
            for oy in range(-(width // 2), width - width // 2):
                self.line(x0 + ox, y0 + oy, x1 + ox, y1 + oy, color)

    def fill_triangle(self, p0, p1, p2, color):
        xs, ys = (p0[0], p1[0], p2[0]), (p0[1], p1[1], p2[1])
        denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if denominator == 0:
            return
        for y in range(max(0, min(ys)), min(self.height - 1, max(ys)) + 1):
            for x in range(max(0, min(xs)), min(self.width - 1, max(xs)) + 1):
                a = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denominator
                b = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denominator
                if a >= -0.02 and b >= -0.02 and (1 - a - b) >= -0.02:
                    self.put(x, y, color)

    def disc(self, cx, cy, radius, color):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                    self.put(x, y, color)

    def arrow(self, x0, y0, x1, y1, color, width=3, head=16):
        """Line from (x0, y0) to (x1, y1) with a filled head at the tip (x1, y1)."""
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 3:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        head = min(head, length * 0.6)
        base = (x1 - ux * head * 0.8, y1 - uy * head * 0.8)
        self.thick_line(x0, y0, int(base[0]), int(base[1]), color, width)
        wing = head * 0.55
        left = (int(base[0] - uy * wing), int(base[1] + ux * wing))
        right = (int(base[0] + uy * wing), int(base[1] - ux * wing))
        self.fill_triangle((x1, y1), left, right, color)

    def label(self, x, y, value, fg, bg, edge=None, scale=2):
        """Text on a small filled box; (x, y) is the box's top-left corner."""
        width, height = len(value) * 4 * scale + 6, 5 * scale + 6
        x = max(2, min(x, self.width - width - 2))
        y = max(2, min(y, self.height - height - 2))
        self.rect(x, y, x + width - 1, y + height - 1, bg)
        if edge:
            for xx in (x, x + width - 1):
                self.rect(xx, y, xx, y + height - 1, edge)
            for yy in (y, y + height - 1):
                self.rect(x, yy, x + width - 1, yy, edge)
        self.text(x + 3, y + 3, value, fg, scale=scale)
        return width, height

    def text(self, x, y, value, color, scale=2):
        for char in value.upper():
            glyph = _GLYPHS.get(char, _GLYPHS[" "])
            for index, bit in enumerate(glyph):
                if bit == "1":
                    gx, gy = index % 3, index // 3
                    self.rect(x + gx * scale, y + gy * scale,
                              x + gx * scale + scale - 1, y + gy * scale + scale - 1, color)
            x += 4 * scale

    def png(self):
        stride = self.width * 3
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride:(y + 1) * stride])

        def chunk(kind, data):
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def find_fvgs(rates, atr_value=None, limit=3, min_atr=0.05, lookback=None):
    """Fair Value Gaps (three-candle imbalances) in ``rates`` (oldest first, closed candles).

    Bullish FVG: high[i-1] < low[i+1]  -> zone [high[i-1], low[i+1]] (price left a gap upward).
    Bearish FVG: low[i-1]  > high[i+1] -> zone [high[i+1], low[i-1]].
    A gap is *filled* once a later candle trades through the whole zone; it is *touched*
    when price re-entered it only partly. The most important gaps are the unfilled ones
    that are large relative to ATR, recent, and close to the last price.

    Returns a list of dicts sorted by importance (best first):
    {"kind": "bull"|"bear", "index": middle-candle index, "start": first-candle index,
     "top", "bottom", "size", "filled", "touched", "end": index where it was filled or None,
     "score"}.
    """
    high, low, close = rates["high"], rates["low"], rates["close"]
    n = len(close)
    first = 1 if lookback is None else max(1, n - lookback)
    last_price = close[-1]
    gaps = []
    for i in range(first, n - 1):
        prev_high, prev_low = high[i - 1], low[i - 1]
        next_high, next_low = high[i + 1], low[i + 1]
        if prev_high < next_low:
            kind, bottom, top = "bull", prev_high, next_low
        elif prev_low > next_high:
            kind, bottom, top = "bear", next_high, prev_low
        else:
            continue
        size = top - bottom
        if atr_value and size < atr_value * min_atr:
            continue
        filled, touched, end = False, False, None
        for j in range(i + 2, n):
            if kind == "bull":
                if low[j] <= bottom:
                    filled, end = True, j
                    break
                touched = touched or low[j] < top
            else:
                if high[j] >= top:
                    filled, end = True, j
                    break
                touched = touched or high[j] > bottom
        age = n - 1 - i
        relative = size / atr_value if atr_value else 1.0
        mid = (top + bottom) / 2
        distance = abs(last_price - mid) / atr_value if atr_value else 0.0
        score = relative * (0.15 if filled else (0.7 if touched else 1.0))
        score *= 1.0 / (1.0 + age / 60.0) * 1.0 / (1.0 + distance / 6.0)
        gaps.append({"kind": kind, "index": i, "start": i - 1, "top": top, "bottom": bottom,
                     "size": size, "filled": filled, "touched": touched, "end": end,
                     "score": score})
    gaps.sort(key=lambda g: g["score"], reverse=True)
    return gaps[:max(0, limit)]


def find_swings(rates, atr_value=None, strength=3, limit=3, lookback=110, min_atr=0.5):
    """Main swing highs («سقف») and swing lows («کف») of ``rates`` (oldest first, closed candles).

    A swing high is a candle whose high is above the ``strength`` candles on its left and not
    below the ``strength`` candles on its right (mirror image for lows), so the newest
    ``strength`` candles can never be a confirmed swing. Swings are ranked by prominence (how far
    price moved away from the swing on both sides, in ATR units) with a mild preference for recent
    ones; swings that stand out less than ``min_atr`` ATR are ignored as noise.

    Returns ``{"highs": [...], "lows": [...]}``; each item is ``{"index", "price", "prominence"}``,
    sorted by index (oldest first) and capped at ``limit`` per side. Everything comes from the
    candles, never from the language model.
    """
    high, low = rates["high"], rates["low"]
    n = len(high)
    result = {"highs": [], "lows": []}
    if n < 2 * strength + 3:
        return result
    first = max(strength, n - lookback)
    reach = strength * 4
    scale = atr_value if atr_value and atr_value > 0 else (max(high[first:]) - min(low[first:])) / 40 or 1e-9

    def ranked(is_high):
        series = high if is_high else low
        found = []
        for i in range(first, n - strength):
            left, right = series[i - strength:i], series[i + 1:i + strength + 1]
            if is_high:
                if not (series[i] > max(left) and series[i] >= max(right)):
                    continue
                far_left = min(low[max(0, i - reach):i + 1])
                far_right = min(low[i:min(n, i + reach + 1)])
                prominence = series[i] - max(far_left, far_right)
            else:
                if not (series[i] < min(left) and series[i] <= min(right)):
                    continue
                far_left = max(high[max(0, i - reach):i + 1])
                far_right = max(high[i:min(n, i + reach + 1)])
                prominence = min(far_left, far_right) - series[i]
            relative = prominence / scale
            if relative < min_atr:
                continue
            age = n - 1 - i
            found.append({"index": i, "price": series[i], "prominence": prominence,
                          "score": relative / (1.0 + age / 80.0)})
        found.sort(key=lambda item: item["score"], reverse=True)
        best = sorted(found[:max(0, limit)], key=lambda item: item["index"])
        return [{"index": item["index"], "price": item["price"], "prominence": item["prominence"]} for item in best]

    result["highs"], result["lows"] = ranked(True), ranked(False)
    return result


def sanitize_annotations(raw, limit=MAX_ANNOTATIONS):
    """Clean the annotation list chosen by the model (or the user) before drawing.

    Each item: ``type`` arrow | marker | hline | text; ``price`` (and ``to_price`` for arrows);
    ``bar`` / ``to_bar`` = offset from the LAST candle (0 = last, -5 = five candles ago,
    +10 = ten candles into the future); optional ``label`` (Latin letters, digits and a few
    symbols only), ``color`` (green/red/amber/blue/neutral) and, for markers, ``shape``
    (up/down/dot). Bad items are dropped, numbers are clamped, labels are filtered.
    """
    cleaned = []
    if not isinstance(raw, (list, tuple)):
        return cleaned

    def number(item, key):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            value = float(value)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    for item in raw[:limit * 2]:
        if len(cleaned) >= limit:
            break
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).strip().lower()
        if kind not in ("arrow", "marker", "hline", "text"):
            continue
        price = number(item, "price")
        if price is None:
            continue
        bar = number(item, "bar")
        entry = {"type": kind, "price": price, "bar": max(-300, min(60, int(bar or 0)))}
        if kind == "arrow":
            to_price = number(item, "to_price")
            if to_price is None:
                continue
            to_bar = number(item, "to_bar")
            entry["to_price"] = to_price
            entry["to_bar"] = max(-300, min(60, int(to_bar if to_bar is not None else entry["bar"] + 8)))
        color = str(item.get("color", "")).strip().lower()
        if color not in ANNOTATION_COLORS:
            if kind == "arrow":
                color = "green" if entry["to_price"] >= price else "red"
            elif kind == "marker":
                color = "red" if str(item.get("shape", "")).lower() == "down" else "green"
            else:
                color = "amber"
        entry["color"] = color
        label = "".join(ch for ch in str(item.get("label", "")).upper() if ch in LABEL_CHARS)
        entry["label"] = " ".join(label.split())[:16]
        shape = str(item.get("shape", "")).strip().lower()
        entry["shape"] = shape if shape in ("up", "down", "dot") else "dot"
        if kind == "text" and not entry["label"]:
            continue
        cleaned.append(entry)
    return cleaned


def render(rates, ema20=None, ema50=None, supports=(), resistances=(), pivot=None,
           price=None, digits=5, bars=110, width=960, height=540, fvgs=(),
           theme="dark", annotations=()):
    """PNG bytes of the last ``bars`` candles.

    ``rates``: dict of lists (time, open, high, low, close), oldest first, closed candles.
    ``ema20`` / ``ema50``: lists as long as ``rates`` (None where undefined).
    ``supports`` / ``resistances``: prices; ``pivot``: price or None.
    ``fvgs``: gaps from :func:`find_fvgs` computed on the SAME ``rates`` (absolute indexes);
    each is drawn as a translucent box with dashed top/bottom borders, cyan = bullish,
    magenta = bearish, extended to the right edge while the gap is unfilled.
    ``theme``: ``"dark"`` (default) or ``"light"`` (white background).
    ``annotations``: output of :func:`sanitize_annotations`; arrows / markers / lines / short
    labels placed by price and by candle offset from the last candle (positive offsets use
    empty space reserved on the right of the last candle).
    """
    pal = PALETTES.get(theme, PALETTES["dark"])
    notes = sanitize_annotations(list(annotations or ()))
    n = len(rates["close"])
    first = max(0, n - bars)
    o, h, l, c = (rates[key][first:] for key in ("open", "high", "low", "close"))
    count = len(c)
    if count < 5:
        raise ValueError("not enough candles to draw")
    fast = (ema20 or [None] * n)[first:]
    slow = (ema50 or [None] * n)[first:]
    visible_fvgs = [g for g in fvgs if g["index"] + 1 >= first]
    extras = [p for p in list(supports) + list(resistances) + ([pivot] if pivot else []) if p]
    extras += [g["top"] for g in visible_fvgs] + [g["bottom"] for g in visible_fvgs]
    lo, hi = min(l), max(h)
    # keep levels that are near the visible range, ignore far-away ones
    span = (hi - lo) or 1e-9
    near = [p for p in extras if lo - 0.35 * span <= p <= hi + 0.35 * span]
    note_prices = []
    for note in notes:
        for key in ("price", "to_price"):
            if key in note and lo - 0.8 * span <= note[key] <= hi + 0.8 * span:
                note_prices.append(note[key])
    lo, hi = min([lo] + near + note_prices), max([hi] + near + note_prices)
    pad = (hi - lo) * 0.06 or 1e-9
    lo, hi = lo - pad, hi + pad

    left, right, top, bottom = 10, 96, 12, 14
    plot_w, plot_h = width - left - right, height - top - bottom
    canvas = Canvas(width, height, pal["bg"])

    def y_of(price_value):
        return int(top + (hi - price_value) / (hi - lo) * (plot_h - 1))

    # empty room on the right of the last candle for arrows that point into the future
    future = 0
    for note in notes:
        future = max(future, note["bar"], note.get("to_bar", 0))
    future = min(40, future + 2) if future > 0 else 0
    slots = count + future
    step = plot_w / slots

    def x_of(bar_offset):
        return int(left + (count - 1 + bar_offset + 0.5) * step)

    # grid and price scale
    for i in range(6):
        value = lo + (hi - lo) * i / 5
        y = y_of(value)
        canvas.hline(y, left, left + plot_w, pal["grid"])
        canvas.text(left + plot_w + 8, y - 5, f"{value:.{digits}f}", pal["text"], scale=2)
    # levels first, candles over them
    for p in supports:
        if p in near:
            for dy in (0, 1):
                canvas.hline(y_of(p) + dy, left, left + plot_w, pal["support"], dash=8)
    for p in resistances:
        if p in near:
            for dy in (0, 1):
                canvas.hline(y_of(p) + dy, left, left + plot_w, pal["resistance"], dash=8)
    if pivot and pivot in near:
        for dy in (0, 1):
            canvas.hline(y_of(pivot) + dy, left, left + plot_w, pal["pivot"], dash=3)
    for g in visible_fvgs:
        color = pal["fvg_bull"] if g["kind"] == "bull" else pal["fvg_bear"]
        x_start = int(left + (max(g["start"], first) - first + 0.5) * step)
        end_index = g["end"] if g["end"] is not None else None
        x_end = (int(left + (end_index - first + 0.5) * step) if end_index is not None
                 else left + plot_w)
        y_top, y_bottom = y_of(g["top"]), y_of(g["bottom"])
        if y_bottom - y_top < 2:
            y_bottom = y_top + 2
        canvas.blend_rect(x_start, y_top, x_end, y_bottom, color, alpha=0.20 if theme != "light" else 0.16)
        for y in (y_top, y_bottom):
            canvas.hline(y, x_start, x_end, color, dash=6)
            canvas.hline(y + 1, x_start, x_end, color, dash=6)
        # small solid marker at the left edge so the start candle is obvious
        canvas.rect(x_start, y_top, x_start + 2, y_bottom, color)
    body_w = max(1, int(step * 0.62))
    for i in range(count):
        x = int(left + (i + 0.5) * step)
        color = pal["up"] if c[i] >= o[i] else pal["down"]
        canvas.rect(x, y_of(h[i]), x, y_of(l[i]), color)
        top_body, bottom_body = y_of(max(o[i], c[i])), y_of(min(o[i], c[i]))
        if bottom_body - top_body < 1:
            bottom_body = top_body + 1
        canvas.rect(x - body_w // 2, top_body, x - body_w // 2 + body_w, bottom_body, color)
    for series, color in ((slow, pal["ema_slow"]), (fast, pal["ema_fast"])):
        previous = None
        for i, value in enumerate(series):
            if value is None or not lo <= value <= hi:
                previous = None
                continue
            point = (int(left + (i + 0.5) * step), y_of(value))
            if previous:
                canvas.line(previous[0], previous[1], point[0], point[1], color)
            previous = point
    last_price = price if price is not None else c[-1]
    if lo <= last_price <= hi:
        y = y_of(last_price)
        canvas.hline(y, left, left + plot_w, pal["price"], dash=3)
        canvas.rect(left + plot_w + 4, y - 8, width - 2, y + 8, pal["tag"])
        canvas.text(left + plot_w + 8, y - 5, f"{last_price:.{digits}f}", pal["price"], scale=2)
    # annotations last, so they sit on top of candles and levels
    for note in notes:
        color = pal[note["color"]]
        x, y = x_of(note["bar"]), y_of(note["price"])
        label = note["label"]
        if note["type"] == "hline":
            canvas.hline(y, left, left + plot_w, color, dash=10)
            canvas.hline(y + 1, left, left + plot_w, color, dash=10)
            if label:
                canvas.label(left + 4, y - 18, label, color, pal["label_bg"], edge=color)
        elif note["type"] == "arrow":
            x2, y2 = x_of(note["to_bar"]), y_of(note["to_price"])
            canvas.arrow(x, y, x2, y2, color, width=3)
            if label:
                above = y2 <= y
                canvas.label(x2 - len(label) * 4 - 3, y2 - 24 if above else y2 + 8, label, color,
                             pal["label_bg"], edge=color)
        elif note["type"] == "marker":
            size = 8
            if note["shape"] == "up":
                canvas.fill_triangle((x, y - size), (x - size, y + size), (x + size, y + size), color)
            elif note["shape"] == "down":
                canvas.fill_triangle((x, y + size), (x - size, y - size), (x + size, y - size), color)
            else:
                canvas.disc(x, y, 6, color)
            if label:
                below = note["shape"] == "up"
                canvas.label(x - len(label) * 4 - 3, y + size + 4 if below else y - size - 26, label, color,
                             pal["label_bg"], edge=color)
        elif label:
            canvas.label(x, y, label, color, pal["label_bg"], edge=color)
    return canvas.png()
