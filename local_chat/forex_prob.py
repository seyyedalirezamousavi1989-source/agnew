"""Empirical probability model for LocalChat's forex desk — pure Python.

The language model never invents a percentage. Every probability in the report comes
from here and has a meaning that can be checked:

    "Among past bars of THIS pair on THIS timeframe whose market state looked like the
     current one, price reached +A before -A within H bars in x % of the cases."

How it works
* State of every closed bar = trend bucket of the trading timeframe, bias of the two
  next higher timeframes (aligned in time, no look-ahead), position relative to EMA50
  (stretched or not), ADX regime and RSI zone.
* Outcome of every bar = which side of a symmetric barrier (±A, A scaled by the ATR of
  that bar) is touched first inside H bars: UP, DOWN, or RANGE (neither).
* Probabilities use hierarchical shrinkage: a coarse state is estimated first, finer
  states refine it only as far as they have enough (overlap-corrected) observations.
* A walk-forward test replays history chronologically — each bar is predicted using only
  bars whose outcome was already known — and reports the Brier skill against the plain
  base rate. If the model has no demonstrable edge, the report says so.

No random numbers, no external libraries: the same input gives the same output.
"""
import math

if __package__:
    from . import forex_analysis as fa
else:  # pragma: no cover - script usage
    import forex_analysis as fa

# bars to look ahead per trading timeframe (≈ the lifetime of a typical trade on it)
HORIZON = {"M1": 30, "M5": 24, "M15": 16, "M30": 16, "H1": 16, "H4": 12, "D1": 8, "W1": 6}
WARMUP = 60                 # bars an indicator needs before a state is meaningful
SHRINK_STRENGTH = 12.0      # prior weight (in independent-observation units)
MIN_BARS_FOR_MODEL = 400    # fewer closed bars than this: no probabilities are given
Z80 = 1.2816                # 80 % interval
MIN_LEVEL_SAMPLES = 25      # matches (overlap-corrected) needed to call a level "supported"

# how much of the (state − base rate) difference is kept, by walk-forward verdict
DAMPING = {"قابل اتکا": 1.0, "ضعیف": 0.6, "ناکافی": 0.4, "بدون لبه": 0.25}

TREND_LABELS = {2: "صعودی قوی", 1: "صعودی", 0: "خنثی", -1: "نزولی", -2: "نزولی قوی"}
HTF_LABELS = {1: "صعودی", 0: "خنثی/متناقض", -1: "نزولی"}
LOC_LABELS = {1: "کشیده بالای EMA50", 0: "نزدیک EMA50", -1: "کشیده زیر EMA50"}


# --------------------------------------------------------------------------------------
# per-bar features (no look-ahead: bar i only uses bars <= i)
# --------------------------------------------------------------------------------------

def trend_ratio_series(closes, highs=None, lows=None):
    """Vote ratio in [-1, 1] per bar from EMA20/50 order, close vs EMA50, MACD, slope, RSI."""
    n = len(closes)
    ema20, ema50 = fa.ema(closes, 20), fa.ema(closes, 50)
    _line, _signal, hist = fa.macd(closes)
    rsi14 = fa.rsi(closes, 14)
    out = [None] * n
    for i in range(n):
        if ema50[i] is None or hist[i] is None or rsi14[i] is None or i < 5 or ema50[i - 5] is None:
            continue
        votes = [
            closes[i] > ema50[i] if closes[i] != ema50[i] else None,
            ema20[i] > ema50[i] if ema20[i] != ema50[i] else None,
            hist[i] > 0 if hist[i] != 0 else None,
            ema50[i] > ema50[i - 5] if ema50[i] != ema50[i - 5] else None,
            True if rsi14[i] > 55 else (False if rsi14[i] < 45 else None),
        ]
        out[i] = sum(0 if v is None else (1 if v else -1) for v in votes) / len(votes)
    return out


def trend_bucket(ratio):
    if ratio is None:
        return 0
    if ratio >= 0.7:
        return 2
    if ratio >= 0.3:
        return 1
    if ratio <= -0.7:
        return -2
    if ratio <= -0.3:
        return -1
    return 0


def htf_bias_series(rates_closed):
    """Bias (-1/0/+1) of every closed candle of a higher timeframe."""
    ratios = trend_ratio_series(rates_closed["close"])
    out = []
    for ratio in ratios:
        if ratio is None:
            out.append(0)
        else:
            out.append(1 if ratio >= 0.3 else (-1 if ratio <= -0.3 else 0))
    return out


def align_htf(trade_times, trade_seconds, htf_times, htf_seconds, htf_bias):
    """HTF bias known at the close of each trading bar (a candle counts once it has closed)."""
    out = []
    j = -1
    for t in trade_times:
        close_time = t + trade_seconds
        while j + 1 < len(htf_times) and htf_times[j + 1] + htf_seconds <= close_time:
            j += 1
        out.append(htf_bias[j] if j >= 0 else 0)
    return out


SESSION_LABELS = {0: "آسیا/خارج از سشن", 1: "لندن", 2: "همپوشانی لندن-نیویورک", 3: "نیویورک (انتهای روز)"}
SESSION_MAX_FRAME = 4 * 3600     # from H4 up a candle spans several sessions: no session feature


def session_of(utc_hour):
    """Bucket of the UTC hour: 0 Asia/off-hours, 1 London, 2 London–New York overlap, 3 late New York."""
    if 7 <= utc_hour < 12:
        return 1
    if 12 <= utc_hour < 16:
        return 2
    if 16 <= utc_hour < 21:
        return 3
    return 0


def session_series(times, trade_seconds, utc_offset):
    """Session bucket at the close of every bar (all 0 when it is meaningless or unknown)."""
    if utc_offset is None or trade_seconds >= SESSION_MAX_FRAME:
        return [0] * len(times)
    return [session_of(((t + trade_seconds - utc_offset) // 3600) % 24) for t in times]


def build_features(rates_closed, trading_tf, htf_rates, utc_offset=None):
    """Per-bar feature lists for the trading timeframe.

    ``htf_rates`` maps timeframe name -> closed rates for the frames that vote on the bias.
    ``utc_offset`` is the broker server's UTC offset in seconds (None: session unknown).
    """
    o, h, l, c, t = (rates_closed[k] for k in ("open", "high", "low", "close", "time"))
    n = len(c)
    ratios = trend_ratio_series(c)
    atr14 = fa.atr(h, l, c, 14)
    ema50 = fa.ema(c, 50)
    adx14, _p, _m = fa.adx(h, l, c, 14)
    rsi14 = fa.rsi(c, 14)
    biases = []
    start = WARMUP
    trade_seconds = fa.TIMEFRAME_SECONDS[trading_tf]
    for name, rates in htf_rates.items():
        if not rates or len(rates["close"]) < 60:
            continue
        series = htf_bias_series(rates)
        biases.append(align_htf(t, trade_seconds, rates["time"], fa.TIMEFRAME_SECONDS[name], series))
        # the higher timeframe is only meaningful once its own indicators have warmed up
        ready_at = rates["time"][WARMUP] + fa.TIMEFRAME_SECONDS[name]
        first_ready = next((i for i in range(n) if t[i] + trade_seconds >= ready_at), n)
        start = max(start, first_ready)
    trend5, htf, loc, adx2, rsi3 = [], [], [], [], []
    for i in range(n):
        trend5.append(trend_bucket(ratios[i]))
        total = sum(b[i] for b in biases) if biases else 0
        htf.append(1 if total > 0 else (-1 if total < 0 else 0))
        distance = None
        if ema50[i] is not None and atr14[i]:
            distance = (c[i] - ema50[i]) / atr14[i]
        loc.append(0 if distance is None else (1 if distance > 1.5 else (-1 if distance < -1.5 else 0)))
        adx2.append(1 if (adx14[i] is not None and adx14[i] >= 22) else 0)
        value = rsi14[i]
        rsi3.append(0 if value is None else (1 if value > 60 else (-1 if value < 40 else 0)))
    sessions = session_series(t, trade_seconds, utc_offset)
    return {"trend5": trend5, "htf": htf, "loc": loc, "adx2": adx2, "rsi3": rsi3, "atr": atr14,
            "session": sessions, "has_session": any(sessions), "has_htf": bool(biases), "start": start}


def level_keys(features, i):
    """State keys from coarse to fine for bar ``i``."""
    t5 = features["trend5"][i]
    t3 = (t5 > 0) - (t5 < 0)
    htf = features["htf"][i]
    loc = features["loc"][i]
    return (
        (t3,),
        (t5, htf),
        (t5, htf, loc),
        (t5, htf, loc, features["session"][i]),
    )


# --------------------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------------------

def first_passage(opens, highs, lows, start, entry, up, down, horizon):
    """+1 when ``up`` is touched first, -1 when ``down`` is, 0 when neither inside ``horizon`` bars.

    Both touched inside one candle: the barrier closer to that candle's open is assumed to
    have been hit first (the intrabar order is unknown).
    """
    n = len(highs)
    for j in range(1, horizon + 1):
        k = start + j
        if k >= n:
            return None
        hit_up = highs[k] >= up
        hit_down = lows[k] <= down
        if hit_up and hit_down:
            return 1 if abs(opens[k] - up) <= abs(opens[k] - down) else -1
        if hit_up:
            return 1
        if hit_down:
            return -1
    return 0


def pick_barrier_ratio(closes, atr, horizon, start):
    """Barrier A in ATR units: 0.9 × the typical H-bar move, so that RANGE is a real outcome."""
    moves = []
    for i in range(start, len(closes) - horizon):
        if atr[i]:
            moves.append((closes[i + horizon] - closes[i]) / atr[i])
    if len(moves) < 50:
        return 2.0
    mean = sum(moves) / len(moves)
    sigma = math.sqrt(sum((m - mean) ** 2 for m in moves) / len(moves))
    return min(4.0, max(1.0, 0.9 * sigma))


def label_outcomes(rates_closed, atr, horizon, ratio, start):
    """Outcome per bar: 1 / -1 / 0, or None where the future is not complete yet."""
    o, h, l, c = (rates_closed[k] for k in ("open", "high", "low", "close"))
    n = len(c)
    labels = [None] * n
    for i in range(start, n - horizon):
        if not atr[i]:
            continue
        a = ratio * atr[i]
        labels[i] = first_passage(o, h, l, i, c[i], c[i] + a, c[i] - a, horizon)
    return labels


# --------------------------------------------------------------------------------------
# hierarchical shrinkage
# --------------------------------------------------------------------------------------

def _class_index(label):
    return 0 if label == 1 else (1 if label == -1 else 2)   # up, down, range


def _shrink(counts_by_level, effective, base):
    """Probabilities after folding in each finer level with a weight of its sample size.

    ``counts_by_level`` is coarse → fine, every item ``(counts[3], n)`` or None;
    ``effective`` converts the raw count to independent observations.
    Returns (probabilities, weight, per-level info).
    """
    p = list(base)
    weight = 0.0
    info = []
    for entry in counts_by_level:
        if not entry or entry[1] == 0:
            info.append((0, 0.0))
            continue
        counts, n = entry
        w = effective(n)
        scale = w / n
        total = w + SHRINK_STRENGTH
        p = [(counts[k] * scale + SHRINK_STRENGTH * p[k]) / total for k in range(3)]
        weight = w
        info.append((n, w))
    return p, weight, info


def _interval(p, weight):
    spread = Z80 * math.sqrt(max(p * (1 - p), 1e-9) / (weight + SHRINK_STRENGTH + 1))
    return max(0.0, p - spread), min(1.0, p + spread)


def _overlap_factor(horizon):
    return 1.0 / math.sqrt(max(1, horizon))


# --------------------------------------------------------------------------------------
# walk-forward validation
# --------------------------------------------------------------------------------------

def walk_forward(features, labels, horizon, start, test_fraction=0.4, max_test=2500):
    """Replay history in time order, predicting each bar from already-known outcomes only.

    Returns Brier scores (model vs plain base rate), the reliability of the favourite
    class, and the number of test bars. ``None`` when there is too little history.
    """
    n = len(labels)
    usable = [i for i in range(start, n) if labels[i] is not None]
    if len(usable) < 300:
        return None
    first_test = usable[max(int(len(usable) * (1 - test_fraction)), len(usable) - max_test)]
    factor = _overlap_factor(horizon)
    effective = lambda count: count * factor  # noqa: E731
    counts = [dict() for _ in range(4)]
    totals = [0, 0, 0]
    seen = 0
    brier_model = brier_base = 0.0
    tested = hits = 0
    bins = {}
    for t in usable:
        learn = t - horizon - 1            # outcome of this bar is fully known at bar t
        if learn >= start and labels[learn] is not None:
            k = _class_index(labels[learn])
            totals[k] += 1
            seen += 1
            for level, key in enumerate(level_keys(features, learn)):
                entry = counts[level].setdefault(key, [0, 0, 0])
                entry[k] += 1
        if t < first_test or seen < 60:
            continue
        base = [(totals[k] + 1) / (seen + 3) for k in range(3)]
        chain = []
        for level, key in enumerate(level_keys(features, t)):
            entry = counts[level].get(key)
            chain.append((entry, sum(entry)) if entry else None)
        probs, _w, _info = _shrink(chain, effective, base)
        actual = [0.0, 0.0, 0.0]
        actual[_class_index(labels[t])] = 1.0
        brier_model += sum((probs[k] - actual[k]) ** 2 for k in range(3))
        brier_base += sum((base[k] - actual[k]) ** 2 for k in range(3))
        tested += 1
        favourite = max(range(3), key=lambda k: probs[k])
        hits += 1 if actual[favourite] == 1.0 else 0
        bucket = min(int(probs[favourite] * 10), 9)
        stat = bins.setdefault(bucket, [0, 0.0, 0])
        stat[0] += 1
        stat[1] += probs[favourite]
        stat[2] += 1 if actual[favourite] == 1.0 else 0
    if tested < 100:
        return None
    reliability = [
        {"predicted": stat[1] / stat[0], "actual": stat[2] / stat[0], "count": stat[0]}
        for _bucket, stat in sorted(bins.items()) if stat[0] >= 30
    ]
    covered = sum(item["count"] for item in reliability)
    ece = (sum(abs(item["predicted"] - item["actual"]) * item["count"] for item in reliability) / covered
           if covered else None)
    return {
        "tested": tested,
        "ece": ece,
        "brier": brier_model / tested,
        "brier_base": brier_base / tested,
        "skill": 1.0 - (brier_model / brier_base) if brier_base else 0.0,
        "favourite_hit_rate": hits / tested,
        "reliability": reliability,
    }


def grade_validation(validation, weight):
    """Persian verdict on how far the percentages can be trusted."""
    if validation is None:
        return "ناکافی", "تاریخچهٔ کافی برای آزمون پیشرو نبود؛ درصدها را فقط تقریبی بدان."
    skill = validation["skill"]
    if skill >= 0.02 and weight >= MIN_LEVEL_SAMPLES:
        return "قابل اتکا", "مدل در آزمون پیشرو از نرخ پایه بهتر بود."
    if skill > 0.003:
        return "ضعیف", "مدل فقط کمی از نرخ پایه بهتر بود؛ جهت‌گیری ضعیف است."
    return "بدون لبه", "در آزمون پیشرو هیچ برتری نسبت به نرخ پایه نشان نداد؛ درصدها تقریباً همان نرخ تاریخی‌اند."


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------

def analyze(rates_closed, trading_tf, htf_rates, utc_offset=None, extra_damping=1.0):
    """Probabilities of the three outcomes for the next ``horizon`` bars.

    ``rates_closed``: closed candles of the trading timeframe (dict of lists, oldest first).
    ``htf_rates``: {name: closed candles} of the frames voting on the higher-timeframe bias.
    Returns None when there is too little history.
    """
    closes = rates_closed["close"]
    n = len(closes)
    horizon = HORIZON.get(trading_tf, 16)
    if n < MIN_BARS_FOR_MODEL:
        return None
    features = build_features(rates_closed, trading_tf, htf_rates, utc_offset)
    start = features["start"]
    ratio = pick_barrier_ratio(closes, features["atr"], horizon, start)
    labels = label_outcomes(rates_closed, features["atr"], horizon, ratio, start)
    last_index = n - 1
    factor = _overlap_factor(horizon)
    effective = lambda count: count * factor  # noqa: E731

    known = [i for i in range(start, n) if labels[i] is not None]
    totals = [0, 0, 0]
    for i in known:
        totals[_class_index(labels[i])] += 1
    if len(known) < 200:
        return None
    base = [(totals[k] + 1) / (len(known) + 3) for k in range(3)]

    keys_now = level_keys(features, last_index)
    chain_counts = [[0, 0, 0] for _ in range(4)]
    chain_matches = [[] for _ in range(4)]
    for i in known:
        keys = level_keys(features, i)
        for level in range(4):
            if keys[level] == keys_now[level]:
                chain_counts[level][_class_index(labels[i])] += 1
                chain_matches[level].append(i)
    chain = [(chain_counts[lv], len(chain_matches[lv])) if chain_matches[lv] else None for lv in range(4)]
    probs, weight, info = _shrink(chain, effective, base)

    validation = walk_forward(features, labels, horizon, start)
    grade, grade_text = grade_validation(validation, weight)
    # A state that did not beat the base rate in the walk-forward test must not look
    # convincing: pull the percentages back toward the base rate in proportion.
    damping = DAMPING[grade] * extra_damping
    raw_probs = list(probs)
    probs = [base[k] + damping * (probs[k] - base[k]) for k in range(3)]
    atr_now = features["atr"][last_index]
    names = ("up", "down", "range")
    return {
        "horizon_bars": horizon,
        "barrier_atr": ratio,
        "barrier_price": ratio * atr_now if atr_now else None,
        "atr": atr_now,
        "history_bars": n,
        "labeled_bars": len(known),
        "start_index": start,
        "probs": dict(zip(names, probs)),
        "probs_raw": dict(zip(names, raw_probs)),
        "damping": damping,
        "news_damping": extra_damping,
        "ci80": {names[k]: _interval(probs[k], weight) for k in range(3)},
        "base": dict(zip(names, base)),
        "matches": [{"level": lv, "n": info[lv][0], "effective": round(info[lv][1], 1)} for lv in range(4)],
        "effective_samples": weight,
        "state": {
            "trend": TREND_LABELS[features["trend5"][last_index]],
            "htf": HTF_LABELS[features["htf"][last_index]] if features["has_htf"] else "نامشخص",
            "location": LOC_LABELS[features["loc"][last_index]],
            "regime": "روندی (ADX≥22)" if features["adx2"][last_index] else "رنج/ضعیف (ADX<22)",
            "session": SESSION_LABELS[features["session"][last_index]] if features["has_session"] else "نامشخص",
        },
        "validation": validation,
        "grade": grade,
        "grade_text": grade_text,
        "_features": features,
        "_labels": labels,
        "_matches": chain_matches,
    }


# --------------------------------------------------------------------------------------
# odds of a concrete trade idea (entry / target / stop from the chart levels)
# --------------------------------------------------------------------------------------

def trade_odds(rates_closed, model, direction, target_atr, stop_atr):
    """Empirical chance that ``target`` is touched before ``stop`` for similar past states.

    ``direction`` is 1 (long) or -1 (short); distances are in units of the ATR of the bar.
    Uses the finest state level that still has enough matches and shrinks toward the
    all-bars rate of the same barrier pair.
    """
    o, h, l, c = (rates_closed[k] for k in ("open", "high", "low", "close"))
    atr = model["_features"]["atr"]
    horizon = model["horizon_bars"]
    labels = model["_labels"]
    factor = _overlap_factor(horizon)

    def outcome(i):
        if not atr[i]:
            return None
        up = c[i] + (target_atr if direction == 1 else stop_atr) * atr[i]
        down = c[i] - (stop_atr if direction == 1 else target_atr) * atr[i]
        result = first_passage(o, h, l, i, c[i], up, down, horizon)
        if result is None:
            return None
        if result == 0:
            return "timeout"
        return "target" if result == direction else "stop"

    def tally(indexes):
        counts = {"target": 0, "stop": 0, "timeout": 0}
        for i in indexes:
            if labels[i] is None:
                continue
            value = outcome(i)
            if value:
                counts[value] += 1
        return counts

    known = [i for i in range(model["start_index"], len(c)) if labels[i] is not None]
    sample = known[::2] if len(known) > 3000 else known
    base_counts = tally(sample)
    base_n = sum(base_counts.values())
    if base_n == 0:
        return None
    base = [(base_counts[k] + 1) / (base_n + 3) for k in ("target", "stop", "timeout")]
    probs, used_level, used_n = base, -1, base_n
    for level in (0, 1, 2, 3):
        matches = model["_matches"][level]
        if len(matches) * factor < MIN_LEVEL_SAMPLES:
            continue
        counts = tally(matches)
        n = sum(counts.values())
        if n == 0:
            continue
        w = n * factor
        scale = w / n
        total = w + SHRINK_STRENGTH
        vector = [counts[k] * scale for k in ("target", "stop", "timeout")]
        probs = [(vector[j] + SHRINK_STRENGTH * base[j]) / total for j in range(3)]
        used_level, used_n = level, n
    damping = model.get("damping", 1.0)
    probs = [base[j] + damping * (probs[j] - base[j]) for j in range(3)]
    weight = used_n * factor if used_level >= 0 else base_n * factor
    return {
        "target": probs[0], "stop": probs[1], "timeout": probs[2],
        "target_ci80": _interval(probs[0], weight),
        "samples": used_n, "effective": weight, "level": used_level,
        "expectancy_r": probs[0] * (target_atr / stop_atr) - probs[1],
    }


# --------------------------------------------------------------------------------------
# scenarios from levels
# --------------------------------------------------------------------------------------

def _nearest(levels, price, atr, side, minimum, maximum):
    """First level on ``side`` whose distance lies between ``minimum`` and ``maximum`` ATR."""
    for level in levels:
        distance = (level - price) / atr if side == 1 else (price - level) / atr
        if minimum <= distance <= maximum:
            return level
    return None


def build_scenarios(rates_closed, model, snapshot_tf, spread):
    """Two concrete scenarios (long / short) with levels and empirical odds.

    ``snapshot_tf`` is the analysis dict of the trading timeframe (levels, structure).
    ``spread`` is the current ask-bid in price units (0 when unknown).
    """
    price = snapshot_tf["close"]
    atr = model["atr"] or snapshot_tf["atr"]
    if not atr:
        return []
    supports = [lv["price"] for lv in snapshot_tf["supports"]]
    resistances = [lv["price"] for lv in snapshot_tf["resistances"]]
    scenarios = []
    for direction in (1, -1):
        stops_side = supports if direction == 1 else resistances
        goals_side = resistances if direction == 1 else supports
        structural = _nearest(stops_side, price, atr, -direction, 0.6, 3.0)
        if structural is not None:
            stop = structural - direction * 0.3 * atr
            basis = "پشت سطح ساختاری"
        else:
            stop = price - direction * 1.5 * atr
            basis = "۱٫۵ ATR"
        risk = abs(price - stop)
        target1 = _nearest(goals_side, price, atr, direction, 0.8, 5.0)
        if target1 is None or abs(target1 - price) < 1.2 * risk:
            target1_value = price + direction * 1.5 * risk
            target_basis = "۱٫۵ برابر ریسک (سطح مناسب پیدا نشد)"
        else:
            target1_value = target1
            target_basis = "سطح مقابل"
        remaining = [lv for lv in goals_side
                     if (lv - target1_value) * direction > 0.3 * atr and abs(lv - price) <= 5.0 * atr]
        target2 = min(remaining, key=lambda lv: abs(lv - price)) if remaining else price + direction * 3.0 * risk
        odds = trade_odds(rates_closed, model, direction,
                          abs(target1_value - price) / atr, risk / atr)
        if odds is None:
            continue
        cost_r = (spread / risk) if risk else 0.0
        trigger = snapshot_tf["swing_high"] if direction == 1 else snapshot_tf["swing_low"]
        if trigger is not None and abs(trigger - price) > 1.5 * atr:
            trigger = None
        scenarios.append({
            "direction": "long" if direction == 1 else "short",
            "entry": price,
            "stop": stop, "stop_basis": basis,
            "target1": target1_value, "target1_basis": target_basis,
            "target2": target2,
            "risk": risk,
            "rr1": abs(target1_value - price) / risk if risk else None,
            "rr2": abs(target2 - price) / risk if risk else None,
            "trigger": trigger,
            "cost_r": cost_r,
            "odds": odds,
            "expectancy_net_r": odds["expectancy_r"] - cost_r,
        })
    return scenarios


def favourite(model, margin=0.05):
    """Name of the most probable directional scenario, or 'range' / 'balanced'."""
    probs = model["probs"]
    top = max(probs, key=probs.get)
    others = sorted((v for k, v in probs.items() if k != top), reverse=True)
    if top == "range":
        return "range" if probs["range"] - others[0] >= margin else "balanced"
    return top if probs[top] - others[0] >= margin else "balanced"
