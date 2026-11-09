"""Backtest of the forex desk + watcher on recent history.

    python -m local_chat.forex_backtest EURUSD --tf M15 --days 30
    python -m local_chat.forex_backtest EURUSD GBPUSD --tf M15 --days 30 --gates relaxed --out backtest_out

What it does (the same code and rules that run live, replayed candle by candle)
1. Fetch raw candles + spreads from MetaTrader 5 (the trading timeframe and every higher timeframe
   the analysis votes with, plus a warm-up history for the empirical model).
2. Audit and clean the data: duplicates, order, impossible candles, repaired OHLC, gaps, spikes,
   coverage, spread quality. A report says exactly what was dropped or repaired.
3. For every closed candle of the test window: slice every timeframe *as it was at that moment*
   (no candle that was still forming, nothing from the future), run the full analysis and the
   rule-based decision, and feed ``forex_watch.PlanTracker`` (arm → wait → confirm / cancel / expire).
4. A confirmed scenario becomes an order at the *next* candle's open with real costs: spread from the
   data (ask for buys, bid for sells), entry/stop slippage, optional commission. Stop and target are
   checked with bid/ask-correct prices; if both are inside one candle the stop is assumed first;
   a candle that gaps over a level fills at the open. Risk per trade, one position at a time and the
   daily caps are the watcher's.
5. Report: trades, expectancy in R with a bootstrap confidence interval, win rate (Wilson interval),
   profit factor, drawdown, breakdowns, calibration of the predicted target probability, and two
   benchmarks (buy & hold, and random entries with the same stop/target distances and costs).

Known limits are printed with the result (no historical news calendar, bar-level spread proxy,
one month is one market regime, small samples).
"""
import argparse
import bisect
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

if __package__:
    from . import forex_analysis as fa
    from . import forex_agent, forex_chart, forex_decision, forex_desk, forex_intent, forex_prob, forex_watch, mt5_data
    from . import settings as app_settings
else:  # pragma: no cover
    import forex_analysis as fa
    import forex_agent
    import forex_chart
    import forex_decision
    import forex_desk
    import forex_intent
    import forex_prob
    import forex_watch
    import mt5_data
    import settings as app_settings

GATES = {"strict": ("قابل اتکا",), "relaxed": ("قابل اتکا", "ضعیف")}
DEFAULTS = {
    "days": 30, "gates": "strict", "risk_pct": 0.5, "max_trades_per_day": 3, "max_daily_loss_pct": 2.0,
    "fixed_spread_pips": None, "max_decision_bars": 6000, "slippage_spread_frac": 0.2, "stop_slippage_spread_frac": 0.5,
    "commission_pips": 0.0, "random_runs": 300, "seed": 7, "ai_scenario": False, "ai_every": 4, "server_offset_hours": None,
}
FALLBACK_SPREAD_PIPS = 1.0
MIN_COVERAGE = 0.6
FRAME_MARGIN = 30
WEEKEND_GAP = 40 * 3600
DAILY_BREAK_MAX = 2.5 * 3600
BOOTSTRAP_RUNS = 2000
MIN_TRADES_FOR_VERDICT = 30

LIMITATIONS = (
    "تقویم خبری تاریخی در دسترس نیست؛ شوک‌های خبری در این آزمایش فیلتر نشده‌اند و نتیجه نسبت به حالت زنده خوش‌بینانه‌تر است.",
    "اسپرد هر کندل از دادهٔ بروکر خوانده می‌شود (نمونهٔ ابتدای/میانگین کندل)، نه تیک‌به‌تیک؛ لغزش واقعی در خبر و ساعات کم‌نقدینگی بیشتر است.",
    "یک بازهٔ کوتاه فقط یک رژیم بازار را نشان می‌دهد؛ نتیجه برای بازه‌های دیگر تضمین نیست.",
    "اگر تعداد معامله‌ها کم باشد، هر نتیجه (مثبت یا منفی) می‌تواند اتفاقی باشد؛ فاصلهٔ اطمینان زیر همین را نشان می‌دهد.",
)


class BacktestError(Exception):
    """Persian, user-safe message."""

    def __init__(self, message, code="backtest_error"):
        super().__init__(message)
        self.message, self.code = message, code


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------------------
# data audit & cleaning
# --------------------------------------------------------------------------------------

def clean_frame(rates, tf, fixed_spread=0.0, point=0.0):
    """Clean one timeframe. Returns ``(clean, report)``; ``clean`` has time/open/high/low/close/spread.

    ``spread`` is in price units (broker points × point, or ``fixed_spread`` when the broker gives none).
    """
    seconds = fa.TIMEFRAME_SECONDS[tf]
    report = {"raw_bars": len(rates["time"]), "bad_values": 0, "duplicates": 0, "out_of_order": 0,
              "ohlc_repaired": 0, "gaps": 0, "missing_bars": 0, "weekend_gaps": 0, "daily_breaks": 0, "spikes": 0, "flat_bars": 0,
              "spread_source": "fixed", "spread_median": None, "spread_max": None}
    spreads_raw = rates.get("spread")
    have_spread = bool(spreads_raw) and len(spreads_raw) == len(rates["time"]) and point > 0
    rows = []
    for index in range(len(rates["time"])):
        try:
            t = int(rates["time"][index])
            o, h, l, c = (float(rates[key][index]) for key in ("open", "high", "low", "close"))
        except (TypeError, ValueError):
            report["bad_values"] += 1
            continue
        if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) or h < l:
            report["bad_values"] += 1
            continue
        sp = float(spreads_raw[index]) * point if have_spread else 0.0
        rows.append((t, o, h, l, c, sp if math.isfinite(sp) and sp > 0 else 0.0))
    if any(rows[i][0] < rows[i - 1][0] for i in range(1, len(rows))):
        report["out_of_order"] = sum(1 for i in range(1, len(rows)) if rows[i][0] < rows[i - 1][0])
        rows.sort(key=lambda r: r[0])
    unique = []
    for row in rows:
        if unique and row[0] == unique[-1][0]:
            report["duplicates"] += 1
            continue
        unique.append(row)
    clean = {key: [] for key in ("time", "open", "high", "low", "close", "spread")}
    ranges = []
    for t, o, h, l, c, sp in unique:
        top, bottom = max(h, o, c), min(l, o, c)
        if top != h or bottom != l:
            report["ohlc_repaired"] += 1
        if top == bottom:
            report["flat_bars"] += 1
        for key, value in zip(clean, (t, o, top, bottom, c, sp)):
            clean[key].append(value)
        ranges.append(top - bottom)
    n = len(clean["time"])
    times = clean["time"]
    for i in range(1, n):
        delta = times[i] - times[i - 1]
        if delta > seconds * 1.5:
            if tf != "D1" and tf != "W1" and delta >= WEEKEND_GAP:
                report["weekend_gaps"] += 1
            elif tf in ("D1", "W1") and delta <= 4 * 86400:
                continue
            elif tf not in ("H4", "D1", "W1") and delta <= DAILY_BREAK_MAX:
                report["daily_breaks"] += 1          # scheduled maintenance break (gold, indices…), not a data hole
            elif delta / seconds - 1 >= 3:
                report["gaps"] += 1
                report["missing_bars"] += int(delta / seconds) - 1
    if n:
        ordered = sorted(r for r in ranges if r > 0)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        if median > 0:
            report["spikes"] = sum(1 for r in ranges if r > 15 * median)
    positive = sorted(s for s in clean["spread"] if s > 0)
    report["spread_nonzero_share"] = len(positive) / n if n else 0.0
    if have_spread and len(positive) >= max(10, 0.5 * n):
        report["spread_source"] = "broker"
        report["spread_median"] = positive[len(positive) // 2]
        report["spread_max"] = positive[-1]
        median = report["spread_median"]
        clean["spread"] = [s if s > 0 else median for s in clean["spread"]]
    else:
        clean["spread"] = [fixed_spread] * n
        report["spread_median"] = report["spread_max"] = fixed_spread
    report["bars"] = n
    return clean, report


def _us_dst(utc_dt):
    """True during US daylight saving time (second Sunday of March → first Sunday of November)."""
    year = utc_dt.year
    march = datetime(year, 3, 8, tzinfo=timezone.utc)
    start = march + timedelta(days=(6 - march.weekday()) % 7, hours=7)
    november = datetime(year, 11, 1, tzinfo=timezone.utc)
    end = november + timedelta(days=(6 - november.weekday()) % 7, hours=6)
    return start <= utc_dt < end


def infer_server_clock(times, live_offset=None, forced_hours=None):
    """How broker server time maps to UTC: ``(function(server_epoch) -> offset_seconds, description)``.

    Forced hours win; then the weekly re-open pattern (a Monday 00:00 open means a New-York-close
    server: UTC+2 in winter, UTC+3 in US summer); then the live tick estimate; else UTC.
    """
    if forced_hours is not None:
        return (lambda _t: int(forced_hours * 3600)), f"ساعت سرور دستی: UTC{forced_hours:+g}"
    reopens = [times[i] for i in range(1, len(times)) if times[i] - times[i - 1] >= WEEKEND_GAP]
    pattern = {}
    for t0 in reopens:
        moment = datetime.fromtimestamp(t0, timezone.utc)
        pattern[(moment.weekday(), moment.hour)] = pattern.get((moment.weekday(), moment.hour), 0) + 1
    if pattern:
        (weekday, hour), _ = max(pattern.items(), key=lambda item: item[1])
        if weekday == 0 and hour in (0, 1):       # FX re-opens Monday 00:00, gold/indices 01:00 (NY-close server)
            def offset(t):
                utc = datetime.fromtimestamp(t - 2 * 3600, timezone.utc)
                return (3 if _us_dst(utc) else 2) * 3600
            return offset, "ساعت سرور از الگوی بازگشایی هفتگی: بروکر نوع «بسته‌شدن نیویورک» (UTC+2/+3 با ساعت تابستانی آمریکا)"
        if weekday == 6 and 21 <= hour <= 23:
            fixed = (hour - 22) * 3600
            return (lambda _t: fixed), f"ساعت سرور از الگوی بازگشایی هفتگی: UTC{(hour - 22):+d}"
    if live_offset is not None:
        return (lambda _t: live_offset), f"ساعت سرور از تیک زنده: UTC{live_offset / 3600:+g}"
    return (lambda _t: 0), "ساعت سرور نامشخص؛ UTC فرض شد (ساعت سشن‌ها ممکن است چند ساعت جابه‌جا باشد)"


# --------------------------------------------------------------------------------------
# trade simulation
# --------------------------------------------------------------------------------------

def simulate_trade(bars, entry_index, direction, stop, target, max_hold, costs, planned_risk=None):
    """Replay one order that is sent right before candle ``entry_index`` opens.

    ``bars``: cleaned trading-timeframe candles (bid prices) incl. ``spread``. ``costs``: slippage
    fractions of the spread and commission in price units. Returns a trade dict or None when the
    fill would be invalid (stop/target already crossed or the risk exploded versus the plan).
    """
    long = direction == "long"
    n = len(bars["time"])
    if entry_index >= n:
        return None
    o = bars["open"][entry_index]
    s0 = bars["spread"][entry_index]
    slip = costs["slippage_spread_frac"] * s0
    entry = o + s0 + slip if long else o - slip
    risk = entry - stop if long else stop - entry
    reward = target - entry if long else entry - target
    if risk <= 0 or reward <= 0:
        return None
    if planned_risk and risk > 1.6 * planned_risk:
        return None
    exit_price, reason, exit_index = None, None, None
    for j in range(entry_index, n):
        op, hi, lo, cl, sp = (bars[k][j] for k in ("open", "high", "low", "close", "spread"))
        stop_slip = costs["stop_slippage_spread_frac"] * sp
        if long:
            if op <= stop:
                exit_price, reason = op - stop_slip, "stop"
            elif op >= target:
                exit_price, reason = op, "target"
            elif lo <= stop:
                exit_price, reason = stop - stop_slip, "stop"
            elif hi >= target:
                exit_price, reason = target, "target"
        else:
            if op + sp >= stop:
                exit_price, reason = op + sp + stop_slip, "stop"
            elif op + sp <= target:
                exit_price, reason = op + sp, "target"
            elif hi + sp >= stop:
                exit_price, reason = stop + stop_slip, "stop"
            elif lo + sp <= target:
                exit_price, reason = target, "target"
        if reason:
            exit_index = j
            break
        if j - entry_index + 1 >= max_hold:
            exit_price, reason, exit_index = (cl if long else cl + sp), "timeout", j
            break
    if reason is None:
        j = n - 1
        exit_price, reason, exit_index = (bars["close"][j] if long else bars["close"][j] + bars["spread"][j]), "end_of_data", j
    pnl = (exit_price - entry) if long else (entry - exit_price)
    pnl -= costs["commission_price"]
    return {
        "direction": direction, "entry_index": entry_index, "exit_index": exit_index,
        "entry_time": bars["time"][entry_index], "exit_time": bars["time"][exit_index],
        "exit_close_time": bars["time"][exit_index] + costs.get("tf_seconds", 0),
        "entry": entry, "exit": exit_price, "stop": stop, "target": target, "risk": risk,
        "r": pnl / risk, "reason": reason, "bars_held": exit_index - entry_index + 1,
        "reward_r": reward / risk,
    }


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def wilson(wins, n, z=1.96):
    if n == 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_ci(values, seed, runs=BOOTSTRAP_RUNS):
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(runs))
    return means[int(0.025 * runs)], means[int(0.975 * runs) - 1]


def equity_curve(trades, risk_pct):
    equity, curve, peak, max_dd = 100.0, [(None, 100.0)], 100.0, 0.0
    for trade in trades:
        equity *= 1 + risk_pct * trade.get("risk_mult", 1.0) / 100.0 * trade["r"]
        curve.append((trade["exit_time"], equity))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return curve, max_dd


def _hour_bucket(epoch, offset):
    hour = datetime.fromtimestamp(epoch - offset, timezone.utc).hour
    if 0 <= hour < 7:
        return "آسیا"
    if 7 <= hour < 12:
        return "لندن"
    if 12 <= hour < 16:
        return "هم‌پوشانی لندن/نیویورک"
    if 16 <= hour < 21:
        return "نیویورک"
    return "خارج از سشن اصلی"


def _group(trades, key):
    groups = {}
    for trade in trades:
        groups.setdefault(key(trade), []).append(trade["r"])
    return {name: {"n": len(v), "avg_r": sum(v) / len(v), "win_rate": sum(1 for x in v if x > 0) / len(v)}
            for name, v in sorted(groups.items())}


def summarize(trades, params, offset_fn):
    n = len(trades)
    rs = [t["r"] for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    curve, max_dd = equity_curve(trades, params["risk_pct"])
    streak = longest = 0
    for r in rs:
        streak = streak + 1 if r <= 0 else 0
        longest = max(longest, streak)
    mean = sum(rs) / n if n else None
    sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1)) if n >= 2 else None
    gross_loss = -sum(losses)
    out = {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / n if n else None, "win_rate_ci": wilson(len(wins), n),
        "expectancy_r": mean, "expectancy_ci": bootstrap_ci(rs, params["seed"]),
        "t_stat": (mean / (sd / math.sqrt(n))) if sd else None,
        "avg_win_r": sum(wins) / len(wins) if wins else None,
        "avg_loss_r": sum(losses) / len(losses) if losses else None,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
        "total_r": sum(rs), "return_pct": curve[-1][1] - 100.0, "max_drawdown_pct": max_dd,
        "longest_losing_streak": longest,
        "avg_bars_held": sum(t["bars_held"] for t in trades) / n if n else None,
        "by_direction": _group(trades, lambda t: "خرید" if t["direction"] == "long" else "فروش"),
        "by_session": _group(trades, lambda t: _hour_bucket(t["entry_time"], offset_fn(t["entry_time"]))),
        "by_exit": _group(trades, lambda t: {"target": "هدف", "stop": "حد ضرر", "timeout": "پایان مهلت",
                                              "end_of_data": "پایان داده"}[t["reason"]]),
        "curve": curve,
    }
    return out


def random_baseline(bars, decision_indices, trades, params, costs, max_hold):
    """Same number of trades at random candles/directions with each real trade's stop & target distances."""
    if not trades or len(decision_indices) < 5:
        return None
    rng = random.Random(params["seed"] + 1)
    usable = [i for i in decision_indices if i + 1 < len(bars["time"])]
    means = []
    for _ in range(params["random_runs"]):
        total, count = 0.0, 0
        for trade in trades:
            i = rng.choice(usable)
            direction = rng.choice(("long", "short"))
            ref = bars["close"][i]
            risk = trade["risk"]
            reward = trade["reward_r"] * risk
            stop = ref - risk if direction == "long" else ref + risk
            target = ref + reward if direction == "long" else ref - reward
            result = simulate_trade(bars, i + 1, direction, stop, target, max_hold, costs)
            if result:
                total += result["r"]
                count += 1
        if count:
            means.append(total / count)
    if not means:
        return None
    actual = sum(t["r"] for t in trades) / len(trades)
    means.sort()
    below = sum(1 for m in means if m < actual) / len(means)
    return {"runs": len(means), "mean_r": sum(means) / len(means),
            "p5": means[int(0.05 * len(means))], "p95": means[min(len(means) - 1, int(0.95 * len(means)))],
            "strategy_percentile": below, "p_value": 1 - below}


def verdict(stats, baseline, params=None):
    n = stats["trades"]
    if n == 0:
        relaxed = bool(params) and params.get("gates") == "relaxed"
        hint = ("بازهٔ بلندتر (مثلاً ۳۰ تا ۶۰ روز) یا نماد/تایم‌فریم دیگری را امتحان کن."
                if relaxed else "بازهٔ بلندتر را امتحان کن یا حالت relaxed را برای نمونهٔ بیشتر (با اعتبار کمتر) اجرا کن.")
        return ("neutral", "هیچ معامله‌ای رخ نداد؛ یعنی " + ("حتی با فیلتر ملایم، " if relaxed else "")
                + "مدل لبهٔ آماری تشخیص نداد یا شرط‌های ورود کامل نشد. این نتیجه نه مثبت است نه منفی؛ " + hint)
    ci = stats["expectancy_ci"]
    if n < MIN_TRADES_FOR_VERDICT:
        sign = "مثبت" if stats["expectancy_r"] > 0 else "منفی"
        return ("small", f"نمونه کوچک است ({n} معامله؛ برای نتیجه‌گیری حداقل حدود {MIN_TRADES_FOR_VERDICT} لازم است). "
                f"میانگین {sign} دیده شد ولی می‌تواند کاملاً اتفاقی باشد؛ آن را سند سودآوری یا ضرردهی نگیر.")
    if ci and ci[0] > 0 and (baseline is None or baseline["p_value"] <= 0.05) and (baseline or params and params.get("pooled")):
        return ("positive", "امید ریاضی مثبت است، بازهٔ ۹۵٪ آن بالای صفر است و از ورود تصادفی بهتر بوده؛ "
                "نشانهٔ لبه است ولی برای ماه‌ها و نمادهای دیگر هم باید تکرار شود.")
    if ci and ci[1] < 0:
        return ("negative", "امید ریاضی منفی است و بازهٔ ۹۵٪ آن زیر صفر است؛ این تنظیم در این بازه ضرردهی بوده و نباید خودکار شود.")
    return ("unclear", "نتیجه از نظر آماری نامشخص است: بازهٔ اطمینان صفر را می‌پوشاند یا از ورود تصادفی جدا نمی‌شود.")


# --------------------------------------------------------------------------------------
# the replay
# --------------------------------------------------------------------------------------

def plan_fetch(trading_tf, days, deep=True):
    frames = forex_intent.plan_timeframes(trading_tf, ())
    warm = forex_desk.bars_plan(trading_tf, frames, deep)
    counts = {}
    for name in frames:
        seconds = fa.TIMEFRAME_SECONDS[name]
        counts[name] = int(warm[name] + math.ceil(days * 86400 / seconds) + FRAME_MARGIN)
    return frames, warm, counts


_BLOCKER_KINDS = (("اعتبار آماری", "اعتبار آماری ضعیف"), ("امید ریاضی", "امید ریاضی کم"), ("احتمال رسیدن", "احتمال هدف کم"),
                  ("بازار", "بازار بسته"), ("اسپرد", "اسپرد بالا"), ("خبر", "خبر"), ("خوانش", "تعارض با چارت"))


def _blocker_kind(text):
    for prefix, name in _BLOCKER_KINDS:
        if text.startswith(prefix):
            return name
    return "دیگر"


def run(symbol, trading_tf, config, params=None, history=None, progress=None, agent=None):
    """Full backtest. ``history`` (test hook) replaces the MT5 download. Returns the result dict.

    ``agent`` (``forex_agent.AgentDecider``) lets the local model take or skip every confirmed scenario.
    Skipped scenarios are still replayed (hypothetically) so the report can say whether the model's
    choices were better than taking everything the rules approved.
    """
    params = {**DEFAULTS, **(params or {})}
    if params["gates"] not in GATES:
        raise BacktestError("حالت gates باید strict یا relaxed باشد.")
    if trading_tf not in fa.TIMEFRAME_SECONDS or trading_tf == "W1":
        raise BacktestError("تایم‌فریم بک‌تست نامعتبر است.")
    days = int(params["days"])
    if not 3 <= days <= 365:
        raise BacktestError("تعداد روز باید بین ۳ و ۳۶۵ باشد.")
    frames, warm, counts = plan_fetch(trading_tf, days)
    if history is None:
        history = mt5_data.fetch_history(symbol, counts, config["symbol_suffix"], config["terminal_path"])
    info = history["info"]
    digits, point = info["digits"], info["point"]
    pip = fa.pip_size(symbol, digits, point)
    one_pip = pip or point * 10 or 0.0001
    live_spread = info.get("spread_points", 0) * point
    if params["fixed_spread_pips"] is not None:
        fixed_spread, spread_note = params["fixed_spread_pips"] * one_pip, f"اسپرد ثابت دستی {params['fixed_spread_pips']} pip"
    elif live_spread > 0:
        fixed_spread, spread_note = live_spread, f"اسپرد ثابت از اسپرد لحظه‌ای نماد ({live_spread / one_pip:.2f} pip)"
    else:
        fixed_spread, spread_note = FALLBACK_SPREAD_PIPS * one_pip, f"اسپرد ثابت پیش‌فرض {FALLBACK_SPREAD_PIPS} pip"

    clean, audits = {}, {}
    for name in frames:
        if name not in history["frames"]:
            raise BacktestError(f"تاریخچهٔ {name} از MT5 دریافت نشد؛ تاریخچهٔ بروکر کوتاه‌تر از نیاز است.", "no_history")
        clean[name], audits[name] = clean_frame(history["frames"][name], name, fixed_spread, point)
    bars = clean[trading_tf]
    # the newest candle may still be forming: never trade on it
    for frame in clean.values():
        for key in frame:
            frame[key] = frame[key][:-1]
    n = len(bars["time"])
    seconds = fa.TIMEFRAME_SECONDS[trading_tf]
    if n < forex_prob.MIN_BARS_FOR_MODEL:
        raise BacktestError("کندل کافی برای تحلیل نبود.", "no_history")
    end_time = bars["time"][-1] + seconds
    window_start = end_time - days * 86400
    first = bisect.bisect_left(bars["time"], window_start)
    indices = list(range(first, n - 1))
    if len(indices) > params["max_decision_bars"]:
        raise BacktestError(f"بازه برای {trading_tf} بیش از {params['max_decision_bars']} کندل می‌شود؛ تعداد روز را کمتر کن.", "too_long")
    expected = days * 86400 * (5 / 7) / seconds
    coverage = len(indices) / expected if expected else 0
    quality = {"coverage": coverage, "frames": audits, "spread_note": spread_note,
               "window_bars": len(indices), "expected_bars": round(expected)}
    if coverage < MIN_COVERAGE:
        raise BacktestError(
            f"فقط {coverage * 100:.0f}% از کندل‌های مورد انتظار {trading_tf} در تاریخچه‌ی بروکر هست؛ "
            "داده برای بک‌تست قابل‌اعتماد نیست (تاریخچه را در MT5 با باز کردن چارت و اسکرول کامل کن).", "poor_data")

    live_offset = None
    tick_time = history["tick"].get("time")
    if tick_time:
        live_offset = fa.estimate_server_offset(tick_time, history["fetched_at"])
    offset_fn, clock_note = infer_server_clock(bars["time"], live_offset, params["server_offset_hours"])

    costs = {"slippage_spread_frac": params["slippage_spread_frac"],
             "stop_slippage_spread_frac": params["stop_slippage_spread_frac"],
             "commission_price": params["commission_pips"] * (pip or point * 10 or 0.0001), "tf_seconds": seconds}
    tradable = GATES[params["gates"]]
    run_config = {**config, "news_enabled": False}
    frame_times = {name: clean[name]["time"] for name in frames}
    spread_key = ("time", "open", "high", "low", "close")

    tracker = forex_watch.PlanTracker(wide=bool(agent and agent.mode == "wide"))
    agent_skips = []
    ai_mode = bool(params["ai_scenario"] and agent)
    ai_every = max(1, int(params["ai_every"]))
    ai_proposals = {"asked": 0, "armed": 0, "idle": 0, "invalid": 0}
    bars_idle = 0
    trades, log = [], []
    grades, blockers, leans, rejections = {}, {}, {}, {}
    tally = {"analysed": 0, "skipped": 0, "no_trade": 0, "watch": 0, "plan": 0, "armed": 0, "confirmed": 0,
             "cancelled": 0, "expired": 0, "rejected": 0, "refused_position": 0, "refused_daily": 0,
             "refused_fill": 0, "agent_skip": 0}
    open_until = -1
    started = time.time()
    for done, i in enumerate(indices):
        T = bars["time"][i] + seconds                       # server time of this candle's close
        offset = offset_fn(T)
        now_utc = datetime.fromtimestamp(T - offset + 1, timezone.utc)
        raw_frames = {}
        for name in frames:
            cut = bisect.bisect_right(frame_times[name], T - fa.TIMEFRAME_SECONDS[name])
            start = max(0, cut - warm[name])
            raw_frames[name] = {key: clean[name][key][start:cut] for key in spread_key}
        bid = bars["close"][i]
        raw = {"symbol": symbol, "broker_symbol": symbol, "info": {"digits": digits, "point": point},
               "tick": {"time": T + 1, "bid": bid, "ask": bid + bars["spread"][i]},
               "rates": raw_frames, "failed_timeframes": []}
        try:
            report = forex_desk.analyze_raw(raw, symbol, trading_tf, run_config, now_utc, deep=True,
                                            tradable=tradable)
        except forex_desk.DeskError:
            tally["skipped"] += 1
            continue
        tally["analysed"] += 1
        decision = report["decision"]
        tally[decision["action"]] += 1
        grade = report["quant"]["grade"] if report.get("quant") else "بدون مدل"
        grades[grade] = grades.get(grade, 0) + 1
        leans[report["lean"]] = leans.get(report["lean"], 0) + 1
        if decision["action"] == "watch":
            for reason_text in decision["blockers"]:
                kind = _blocker_kind(reason_text)
                blockers[kind] = blockers.get(kind, 0) + 1
        if ai_mode:                 # no edge filters: only the model's own scenario is tracked
            decision = forex_agent.synthetic_decision(tracker.plan)
        outcome, detail = tracker.advance(decision, bars["close"][i])
        if ai_mode and tracker.plan is None and outcome != "confirmed":
            bars_idle += 1
            if (bars_idle - 1) % ai_every == 0 and open_until <= i and report["snapshot"]["market"] == "open":
                ai_proposals["asked"] += 1
                before = agent.stats["invalid"]
                proposal, note = agent.propose(report)
                if proposal is None:
                    ai_proposals["invalid" if agent.stats["invalid"] > before else "idle"] += 1
                else:
                    proposal["reference_price"] = bars["close"][i]
                    tracker.plan, tracker.age = proposal, 0
                    ai_proposals["armed"] += 1
                    tally["armed"] += 1
                    outcome = "armed"
        if outcome in tally:
            tally[outcome] += 1
        if outcome == "rejected":
            why = ("، ".join(sorted({_blocker_kind(b) for b in decision["blockers"]})) if decision["blockers"]
                   else {"no_trade": "مدل دیگر لبه/جهت نمی‌دید", "plan": "جهت سناریو عوض شد"}.get(decision["action"], "نامشخص"))
            rejections[why] = rejections.get(why, 0) + 1
        if outcome == "confirmed":
            reason = None
            if open_until > i:
                reason = "refused_position"
            else:
                day = datetime.fromtimestamp(T - offset, timezone.utc).date()
                today = [t for t in trades if t["exit_close_time"] <= T
                         and datetime.fromtimestamp(t["entry_time"] - offset_fn(t["entry_time"]), timezone.utc).date() == day]
                opened = [t for t in trades
                          if datetime.fromtimestamp(t["entry_time"] - offset_fn(t["entry_time"]), timezone.utc).date() == day]
                if len(opened) >= params["max_trades_per_day"] or \
                        sum(t["r"] * t.get("risk_mult", 1.0) for t in today) * params["risk_pct"] <= -params["max_daily_loss_pct"]:
                    reason = "refused_daily"
            if reason:
                tally[reason] += 1
                outcome = reason
            else:
                plan = detail
                trade = simulate_trade(
                    bars, i + 1, plan["direction"], plan["stop"], plan["target1"], (report["quant"] or {}).get("horizon_bars") or max(2, forex_prob.HORIZON.get(trading_tf, 16)),
                    costs, planned_risk=abs(plan["reference_price"] - plan["stop"]))
                if trade is None:
                    tally["refused_fill"] += 1
                    outcome = "refused_fill"
                else:
                    trade.update(plan_p_target=plan["p_target"], plan_expectancy_r=plan["expectancy_r"],
                                 decision_time=T, grade=(report["quant"] or {}).get("grade", "—") if not ai_mode else "سناریوی مدل",
                                 risk_mult=1.0)
                    if agent:
                        ruling = agent.decide(report, decision)
                        trade.update(agent_reason=ruling["reason"], agent_confidence=ruling["confidence"])
                        if ruling["action"] != "take":
                            tally["agent_skip"] += 1
                            agent_skips.append(trade)           # hypothetical: never occupies the position
                            outcome = "agent_skip"
                            trade = None
                        else:
                            trade["risk_mult"] = ruling["risk_mult"]
                    if trade is not None:
                        trades.append(trade)
                        open_until = trade["exit_index"]
        log.append((T, decision["action"], outcome))
        if progress and (done % 10 == 0 or done == len(indices) - 1):
            try:
                progress(done + 1, len(indices), len(trades))
            except Cancelled:
                raise

    stats = summarize(trades, params, offset_fn)
    baseline = random_baseline(bars, indices, trades, params, costs, max(2, forex_prob.HORIZON.get(trading_tf, 16)))
    kind, text = verdict(stats, baseline, params)
    first_close, last_close = bars["close"][indices[0]], bars["close"][indices[-1]]
    stats["calibration"] = _calibration(trades)
    agent_report = _agent_report(agent, trades, agent_skips) if agent else None
    if agent_report and ai_mode:
        agent_report["ai_scenario"] = dict(ai_proposals, every=ai_every)
    if agent_report:
        kind, text = _with_agent_verdict(kind, text, agent_report)
    return {
        "symbol": symbol, "timeframe": trading_tf, "params": params, "digits": digits, "pip": pip,
        "window": {"start": bars["time"][indices[0]], "end": bars["time"][indices[-1]] + seconds,
                   "bars": len(indices), "days": days},
        "clock_note": clock_note, "quality": quality, "tally": tally,
        "diagnostics": {"grades": grades, "blockers": blockers, "leans": leans, "rejections": rejections}, "stats": stats, "baseline": baseline,
        "buy_hold_pct": (last_close / first_close - 1) * 100.0, "verdict": {"kind": kind, "text": text},
        "agent": agent_report, "trades": trades, "log": log, "limitations": list(LIMITATIONS), "seconds": time.time() - started,
    }


def _avg(values):
    return sum(values) / len(values) if values else None


def _agent_report(agent, taken, skipped):
    """Did the model's choices beat taking everything the rules approved?"""
    everything = [t["r"] for t in taken] + [t["r"] for t in skipped]
    taken_r, skipped_r = [t["r"] for t in taken], [t["r"] for t in skipped]
    calls = max(1, agent.stats["calls"])
    return {
        "model": agent.model, "mode": agent.mode, **{k: agent.stats[k] for k in ("calls", "taken", "skipped", "invalid", "errors")},
        "avg_seconds": agent.stats["seconds"] / calls,
        "candidates": len(everything), "taken_avg_r": _avg(taken_r), "skipped_avg_r": _avg(skipped_r),
        "all_avg_r": _avg(everything),
        "value_added_r": (_avg(taken_r) - _avg(everything)) if taken_r and everything else None,
        "skipped_winners": sum(1 for r in skipped_r if r > 0), "skipped_losers": sum(1 for r in skipped_r if r <= 0),
        "taken_winners": sum(1 for r in taken_r if r > 0), "taken_losers": sum(1 for r in taken_r if r <= 0),
        "skips": [{"time": t["entry_time"], "direction": t["direction"], "r": t["r"], "reason": t.get("agent_reason", "")}
                  for t in skipped[:25]],
    }


def _with_agent_verdict(kind, text, report):
    n = report["candidates"]
    if n == 0:
        return kind, text + " (مدل هوش مصنوعی هیچ سناریوی تأییدشده‌ای برای تصمیم‌گیری نداشت.)"
    if n < MIN_TRADES_FOR_VERDICT:
        extra = (f" دربارهٔ مدل: فقط {n} سناریو برای تصمیم‌گیری بود؛ برای قضاوت دربارهٔ تصمیم‌های مدل کافی نیست.")
    elif report["value_added_r"] is None:
        extra = " دربارهٔ مدل: همهٔ سناریوها را رد کرد؛ نمی‌شود ارزش‌افزوده‌اش را سنجید."
    elif report["value_added_r"] > 0.05:
        extra = f" دربارهٔ مدل: انتخاب‌هایش به‌طور میانگین {report['value_added_r']:+.2f}R بهتر از گرفتن همهٔ سناریوها بود."
    elif report["value_added_r"] < -0.05:
        extra = f" دربارهٔ مدل: انتخاب‌هایش {report['value_added_r']:+.2f}R بدتر از گرفتن همهٔ سناریوها بود؛ به او اعتماد نکن."
    else:
        extra = " دربارهٔ مدل: تفاوتی با گرفتن همهٔ سناریوها نداشت."
    return kind, text + extra


def _calibration(trades):
    if not trades:
        return None
    decided = [t for t in trades if t["reason"] in ("target", "stop")]
    return {
        "predicted_target_prob": sum(t["plan_p_target"] for t in trades) / len(trades),
        "realized_target_rate": sum(1 for t in trades if t["reason"] == "target") / len(trades),
        "target_before_stop_rate": (sum(1 for t in decided if t["reason"] == "target") / len(decided)) if decided else None,
        "predicted_expectancy_r": sum(t["plan_expectancy_r"] for t in trades) / len(trades),
        "realized_expectancy_r": sum(t["r"] for t in trades) / len(trades),
    }


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------

def _t(epoch, offset=0):
    return datetime.fromtimestamp(epoch - offset, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _r(value):
    return "-" if value is None else f"{value:+.2f}R"


def _pct(value, digits=1):
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def trades_csv(result):
    digits = result["digits"]
    lines = ["entry_time_server,exit_time_server,direction,entry,stop,target,exit,r,reason,bars_held,p_target,grade,agent_confidence,agent_reason"]
    for t in result["trades"]:
        lines.append(",".join(str(x) for x in (
            _t(t["entry_time"]), _t(t["exit_time"]), t["direction"], f"{t['entry']:.{digits}f}", f"{t['stop']:.{digits}f}",
            f"{t['target']:.{digits}f}", f"{t['exit']:.{digits}f}", f"{t['r']:.3f}", t["reason"], t["bars_held"],
            f"{t['plan_p_target']:.3f}", t["grade"], t.get("agent_confidence", ""),
            '"' + t.get("agent_reason", "").replace('"', "'").replace("\n", " ") + '"')))
    return "\n".join(lines) + "\n"


def equity_png(result):
    curve = [value for _t_, value in result["stats"]["curve"]]
    canvas = forex_chart.Canvas(960, 320)
    left, right, top, bottom = 70, 930, 30, 280
    low, high = min(curve + [100.0]), max(curve + [100.0])
    pad = max((high - low) * 0.1, 0.05)
    low, high = low - pad, high + pad

    def y(v):
        return int(bottom - (v - low) / (high - low) * (bottom - top))

    def x(k):
        return int(left + (right - left) * (k / max(1, len(curve) - 1)))

    grid, base_color, line_color = (42, 51, 66), (139, 149, 167), (91, 157, 255)
    canvas.hline(y(100.0), left, right, base_color, dash=6)
    for tick in (low + pad, 100.0, high - pad):
        canvas.text(6, y(tick) - 5, f"{tick:.2f}", base_color, scale=2)
        canvas.hline(y(tick), left, right, grid)
    for k in range(1, len(curve)):
        for shift in (0, 1):
            canvas.line(x(k - 1), y(curve[k - 1]) + shift, x(k), y(curve[k]) + shift, line_color)
    for k, value in enumerate(curve[1:], 1):
        canvas.rect(x(k) - 2, y(value) - 2, x(k) + 2, y(value) + 2, (47, 191, 143) if value >= curve[k - 1] else (229, 87, 91))
    return canvas.png()


def render_text(result):
    """Persian plain-text report (console, file and the language-model digest source)."""
    s, q, p = result["stats"], result["quality"], result["params"]
    lines = [
        f"بک‌تست {result['symbol']} {result['timeframe']} | {result['window']['days']} روز "
        f"({_t(result['window']['start'])} تا {_t(result['window']['end'])} به ساعت سرور) | "
        f"{result['window']['bars']} کندل تحلیل‌شده | حالت {p['gates']}",
        f"ریسک هر معامله {p['risk_pct']}% | سقف {p['max_trades_per_day']} معامله و ضرر {p['max_daily_loss_pct']}% در روز | "
        f"لغزش ورود {p['slippage_spread_frac']}×اسپرد، لغزش حد ضرر {p['stop_slippage_spread_frac']}×اسپرد",
        "",
        f"نتیجه: {result['verdict']['text']}",
        "",
        f"معامله‌ها: {s['trades']} (برد {s['wins']}، باخت {s['losses']}) | نرخ برد {_pct(s['win_rate'])}"
        + (f" [بازهٔ ۹۵٪: {_pct(s['win_rate_ci'][0], 0)} تا {_pct(s['win_rate_ci'][1], 0)}]" if s["win_rate_ci"] else ""),
    ]
    if s["trades"]:
        ci = s["expectancy_ci"]
        profit_factor = "-" if s["profit_factor"] is None else "%.2f" % s["profit_factor"]
        lines += [
            f"امید ریاضی هر معامله: {s['expectancy_r']:+.2f}R"
            + (f" [بازهٔ ۹۵٪: {ci[0]:+.2f} تا {ci[1]:+.2f}R]" if ci else " (برای بازهٔ اطمینان حداقل ۵ معامله لازم است)"),
            f"مجموع {s['total_r']:+.2f}R | بازده با ریسک {p['risk_pct']}%: {s['return_pct']:+.2f}% | "
            f"حداکثر افت سرمایه {s['max_drawdown_pct']:.2f}% | بدترین سری باخت پیاپی {s['longest_losing_streak']}",
            f"میانگین برد {_r(s['avg_win_r'])} | میانگین باخت {_r(s['avg_loss_r'])} | "
            f"ضریب سود {profit_factor} | "
            f"میانگین نگهداری {s['avg_bars_held']:.1f} کندل",
        ]
        cal = s["calibration"]
        lines.append(
            f"کالیبراسیون: احتمال هدف پیش‌بینی‌شده {_pct(cal['predicted_target_prob'], 0)} در برابر هدف خورده در عمل "
            f"{_pct(cal['realized_target_rate'], 0)}؛ امید ریاضی پیش‌بینی‌شده {cal['predicted_expectancy_r']:+.2f}R در برابر واقعی "
            f"{cal['realized_expectancy_r']:+.2f}R")
        for title, groups in (("جهت", s["by_direction"]), ("سشن ورود", s["by_session"]), ("نوع خروج", s["by_exit"])):
            lines.append(f"{title}: " + " | ".join(f"{k}: {v['n']} معامله، میانگین {v['avg_r']:+.2f}R" for k, v in groups.items()))
    b = result["baseline"]
    lines.append("")
    lines.append(f"خرید و نگه‌داری همین بازه: {result['buy_hold_pct']:+.2f}%")
    if b:
        lines.append(f"ورود تصادفی با همان فاصله حد ضرر/هدف و هزینه‌ها ({b['runs']} بار): میانگین {b['mean_r']:+.2f}R "
                     f"(بازهٔ ۵–۹۵٪: {b['p5']:+.2f} تا {b['p95']:+.2f}R)؛ استراتژی از {b['strategy_percentile'] * 100:.0f}% "
                     f"اجراهای تصادفی بهتر بود (p≈{b['p_value']:.2f})")
    t = result["tally"]
    agent_block = result.get("agent")
    if agent_block:
        lines += ["", f"تصمیم‌گیری مدل هوش مصنوعی ({agent_block['model']}، حالت {agent_block['mode']}): {agent_block['candidates']} سناریوی تأییدشده "
                  f"← گرفت {agent_block['taken']}، رد کرد {agent_block['skipped']}، پاسخ نامعتبر {agent_block['invalid']}، خطا "
                  f"{agent_block['errors']} | میانگین پاسخ {agent_block['avg_seconds']:.1f} ثانیه",
                  f"میانگین R سناریوهای گرفته‌شده {_r(agent_block['taken_avg_r'])}؛ رد‌شده‌ها (فرضی) {_r(agent_block['skipped_avg_r'])}؛ "
                  f"همهٔ سناریوها {_r(agent_block['all_avg_r'])}؛ ارزش‌افزودهٔ مدل {_r(agent_block['value_added_r'])} "
                  f"(بردهایی که رد کرد {agent_block['skipped_winners']}، باخت‌هایی که رد کرد {agent_block['skipped_losers']})"]
        ai = agent_block.get("ai_scenario")
        if ai:
            lines.append(f"حالت سناریوی آزاد مدل (بدون فیلتر لبه): هر {ai['every']} کندل یک‌بار پرسیده شد؛ {ai['asked']} درخواست ← "
                         f"{ai['armed']} سناریو، {ai['idle']} بار «ستاپی نیست»، {ai['invalid']} پیشنهاد نامعتبر/رد‌شده")
    lines += [
        "",
        f"قیف تصمیم: {t['analysed']} تحلیل → «معامله نکن» {t['no_trade']}، «فقط پایش» {t['watch']}، «سناریوی آماده» {t['plan']}؛ "
        f"سناریو فعال {t['armed']}، تأیید {t['confirmed']}، لغو {t['cancelled']}، منقضی {t['expired']}، ردشده در تحلیل تازه "
        f"{t['rejected']}، ردشده به‌خاطر پوزیشن باز {t['refused_position']}، سقف روزانه {t['refused_daily']}، پر نشدن سفارش "
        f"{t['refused_fill']}، ردشده توسط مدل {t['agent_skip']}",
        "",
        f"کیفیت داده ({result['symbol']} {result['timeframe']}): پوشش {q['coverage'] * 100:.0f}% از کندل‌های مورد انتظار",
    ]
    for name, audit in q["frames"].items():
        issues = [f"{label} {audit[key]}" for key, label in (
            ("bad_values", "مقدار نامعتبر"), ("duplicates", "تکراری"), ("out_of_order", "نامرتب"),
            ("ohlc_repaired", "OHLC ترمیم‌شده"), ("gaps", "شکاف غیرعادی"), ("spikes", "جهش >۱۵×معمول"),
            ("flat_bars", "کندل تخت")) if audit[key]]
        lines.append(f"  {name}: {audit['bars']} کندل، اسپرد {'بروکر' if audit['spread_source'] == 'broker' else 'ثابت'}"
                     + (f"؛ مشکلات: {'، '.join(issues)}" if issues else "؛ بدون مشکل")
                     + (f" ({audit['daily_breaks']} وقفهٔ روزانهٔ عادی)" if audit.get("daily_breaks") else ""))
    diag = result.get("diagnostics")
    if diag and result["tally"]["analysed"]:
        total = result["tally"]["analysed"]
        lines += ["", "چرا معامله نشد/کم شد: اعتبار مدل در کندل‌ها: " + "، ".join(
            f"{name} {count * 100 / total:.0f}%" for name, count in sorted(diag["grades"].items(), key=lambda kv: -kv[1]))]
        if diag["blockers"]:
            lines.append("مانع‌های «فقط پایش»: " + "، ".join(f"{name} {count}" for name, count in
                                                         sorted(diag["blockers"].items(), key=lambda kv: -kv[1])))
        if diag.get("rejections"):
            lines.append("دلیل ردشدن سناریو بعد از تأیید کندل: " + "، ".join(
                f"{name} {count}" for name, count in sorted(diag["rejections"].items(), key=lambda kv: -kv[1])))
    lines += ["", "اسپرد: " + q.get("spread_note", "-"), "ساعت سرور: " + result["clock_note"], "", "محدودیت‌ها:"]
    lines += [f"- {item}" for item in result["limitations"]]
    return "\n".join(lines)


def digest(result, max_chars=2600):
    """Compact digest for the language model (numbers it may quote verbatim)."""
    text = render_text(result)
    head = text.split("\n\nمحدودیت‌ها:")[0]
    return head[:max_chars]


def public_result(result, include_curve=True):
    """JSON-safe subset for the page."""
    out = {k: result[k] for k in ("symbol", "timeframe", "params", "digits", "window", "clock_note", "quality",
                                  "tally", "baseline", "buy_hold_pct", "verdict", "limitations", "seconds")}
    stats = {k: v for k, v in result["stats"].items() if k != "curve"}
    stats["curve"] = [[a, b] for a, b in result["stats"]["curve"]] if include_curve else []
    out["stats"] = stats
    out["agent"] = result.get("agent")
    out["trades"] = [{k: t[k] for k in ("direction", "entry_time", "exit_time", "entry", "exit", "stop", "target", "r",
                                        "reason", "bars_held", "plan_p_target")}
                     | {"agent_reason": t.get("agent_reason", ""), "agent_confidence": t.get("agent_confidence")}
                     for t in result["trades"]]
    out["text"] = render_text(result)
    return out


def save(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{result['symbol']}_{result['timeframe']}"
    paths = {
        "json": os.path.join(out_dir, f"backtest_{stem}.json"),
        "csv": os.path.join(out_dir, f"trades_{stem}.csv"),
        "png": os.path.join(out_dir, f"equity_{stem}.png"),
        "txt": os.path.join(out_dir, f"report_{stem}.txt"),
    }
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(public_result(result), handle, ensure_ascii=False, indent=2)
    with open(paths["csv"], "w", encoding="utf-8") as handle:
        handle.write(trades_csv(result))
    with open(paths["png"], "wb") as handle:
        handle.write(equity_png(result))
    with open(paths["txt"], "w", encoding="utf-8") as handle:
        handle.write(render_text(result))
    return paths


def pool(results, params):
    """Combined statistics over several symbols (R-based; each symbol risks ``risk_pct`` independently)."""
    trades = sorted((t for r in results for t in r["trades"]), key=lambda t: t["exit_time"])
    stats = summarize(trades, params, lambda _t: 0)
    kind, text = verdict(stats, None, {**params, "pooled": True})
    lines = [f"=== مجموع {len(results)} نماد: {', '.join(r['symbol'] for r in results)} ===", f"نتیجه: {text}",
             f"معامله‌ها: {stats['trades']} | نرخ برد {_pct(stats['win_rate'])}"]
    if stats["trades"]:
        ci = stats["expectancy_ci"]
        lines.append(f"امید ریاضی هر معامله {stats['expectancy_r']:+.2f}R"
                     + (f" [بازهٔ ۹۵٪: {ci[0]:+.2f} تا {ci[1]:+.2f}R]" if ci else "")
                     + f" | مجموع {stats['total_r']:+.2f}R | ضریب سود "
                     + ("-" if stats["profit_factor"] is None else "%.2f" % stats["profit_factor"]))
    return {"kind": kind, "text": "\n".join(lines), "stats": stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--tf", default="M15")
    parser.add_argument("--days", type=int, default=DEFAULTS["days"])
    parser.add_argument("--gates", choices=sorted(GATES), default="strict")
    parser.add_argument("--risk", type=float, default=DEFAULTS["risk_pct"])
    parser.add_argument("--spread-pips", type=float, default=None,
                        help="used only when the broker gives no per-candle spread (default: the symbol's current spread)")
    parser.add_argument("--max-bars", type=int, default=20000, help="safety cap on analysed candles per symbol")
    parser.add_argument("--commission-pips", type=float, default=0.0, help="round-turn commission in pips")
    parser.add_argument("--server-offset", type=float, default=None, help="broker server UTC offset in hours (default: auto)")
    parser.add_argument("--agent", action="store_true", help="the local language model takes/skips each confirmed scenario")
    parser.add_argument("--model", default="", help="Ollama model for --agent (default: text_model from settings)")
    parser.add_argument("--agent-mode", choices=forex_agent.MODES, default="veto")
    parser.add_argument("--ai-scenario", action="store_true",
                        help="no edge filters: the model proposes the scenarios (implies --agent)")
    parser.add_argument("--ai-every", type=int, default=DEFAULTS["ai_every"], help="candles between proposals while idle")
    parser.add_argument("--out", default="backtest_out")
    args = parser.parse_args(argv)
    config = app_settings.forex_settings()
    params = {"days": args.days, "gates": args.gates, "risk_pct": args.risk, "fixed_spread_pips": args.spread_pips,
              "commission_pips": args.commission_pips, "server_offset_hours": args.server_offset,
              "max_decision_bars": args.max_bars, "ai_scenario": args.ai_scenario, "ai_every": args.ai_every}
    code = 0
    results = []
    for symbol in (s.upper() for s in args.symbols):
        agent = None
        if args.agent or args.ai_scenario:
            model = args.model or config.get("text_model", "")
            if not model:
                parser.error("--agent needs --model (or a text_model in the forex settings)")
            agent = forex_agent.AgentDecider(forex_agent.ollama_chat, model, args.agent_mode)
        print(f"\n=== {symbol} {args.tf.upper()} — دریافت و تحلیل؛ چند دقیقه طول می‌کشد ===")

        def progress(done, total, count):
            print(f"\r  {done}/{total} کندل | {count} معامله", end="", flush=True)

        try:
            result = run(symbol, args.tf.upper(), config, params, progress=progress, agent=agent)
        except (BacktestError, mt5_data.MT5Error) as exc:
            print(f"\n  خطا: {exc.message}")
            code = 1
            continue
        print("\n")
        print(render_text(result))
        paths = save(result, args.out)
        print("\nذخیره شد: " + ", ".join(paths.values()))
        results.append(result)
    if len(results) > 1:
        pooled = pool(results, {**DEFAULTS, **params})
        print("\n" + pooled["text"])
        with open(os.path.join(args.out, "pooled_report.txt"), "w", encoding="utf-8") as handle:
            handle.write(pooled["text"])
    return code


if __name__ == "__main__":
    sys.exit(main())
