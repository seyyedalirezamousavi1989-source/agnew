"""One deterministic decision per analysis (no language model involved).

The narrative written by a language model can differ between models and runs. This module
turns the numbers into a single, rule-based verdict that the page shows on top and that the
automatic watcher (``forex_watch``) executes. The narrative must agree with it.

Actions
* ``no_trade`` – no statistical edge / no usable scenario → stay out.
* ``watch``    – there is a scenario but a gate failed (weak validation, news, spread, …):
                 follow it, do not trade it automatically.
* ``plan``     – every gate passed: a conditional plan with trigger, invalidation and targets.
"""
MIN_EXPECTANCY_R = 0.10       # net of spread, in R, for the scenario in the lean direction
MIN_TARGET_PROB = 0.35        # P(target before stop) must be at least this
MAX_SPREAD_ATR = 0.25
CHASE_ATR = 0.25              # fallback trigger distance when the chart has no swing level nearby
TRADABLE_GRADES = ("قابل اتکا",)
DIRECTION_OF_LEAN = {"up": "long", "down": "short"}
WORD = {"long": "خرید", "short": "فروش"}
# blockers a model may override in "wide" mode (never: closed market, news, spread)
SOFT_BLOCKERS = ("اعتبار آماری", "امید ریاضی", "احتمال رسیدن", "خوانش")


def _fmt(value, digits):
    return "-" if value is None else f"{value:.{digits}f}"


def _blockers(report, scenario, tradable=TRADABLE_GRADES):
    """Reasons the plan must not be traded automatically (Persian, short)."""
    quant, snapshot = report["quant"], report["snapshot"]
    out = []
    if quant["grade"] not in tradable:
        out.append(f"اعتبار آماری مدل «{quant['grade']}» است (برای معامله باید «{'، '.join(tradable)}» باشد)")
    if scenario["expectancy_net_r"] < MIN_EXPECTANCY_R:
        out.append(f"امید ریاضی خالص {scenario['expectancy_net_r']:+.2f}R کمتر از حداقل {MIN_EXPECTANCY_R:.2f}R است")
    if scenario["odds"]["target"] < MIN_TARGET_PROB:
        out.append(f"احتمال رسیدن به هدف قبل از حد ضرر {scenario['odds']['target'] * 100:.0f}% است (کمتر از "
                   f"{MIN_TARGET_PROB * 100:.0f}%)")
    if snapshot["market"] != "open":
        out.append("بازار باز/زنده نیست")
    spread, atr = snapshot["spread"], quant["atr"]
    if spread and atr and spread / atr > MAX_SPREAD_ATR:
        out.append("اسپرد نسبت به نوسان بالاست")
    news = report.get("news")
    if news and news.get("ok") and news.get("imminent"):
        out.append("خبر مهم همین حالا یا خیلی نزدیک است")
    vision = report.get("vision")
    if vision and vision.get("agreement") == "conflict":
        out.append("خوانش تصویری چارت با مدل کمّی تعارض دارد")
    return out


def decide(report, tradable=TRADABLE_GRADES):
    """Decision dict for ``report`` (see module docstring). Pure function."""
    quant, snapshot = report.get("quant"), report["snapshot"]
    tf = report["timeframe"]
    digits = snapshot["digits"]
    base = {"action": "no_trade", "direction": None, "headline": "", "reasons": [], "blockers": [],
            "plan": None, "steps": [], "auto_ok": False, "soft_only": False}
    if quant is None:
        base["headline"] = "داده برای برآورد احتمال کافی نیست؛ معامله نکن."
        base["reasons"] = ["تاریخچهٔ کافی برای مدل تجربی نبود"]
        base["steps"] = ["نماد/تایم‌فریم دیگری را امتحان کن یا بعداً دوباره تحلیل بگیر."]
        return base
    lean = report["lean"]
    direction = DIRECTION_OF_LEAN.get(lean)
    if quant["grade"] in ("بدون لبه", "ناکافی"):
        base["headline"] = f"فعلاً معامله نکن؛ اعتبار آماری مدل «{quant['grade']}» است."
        base["reasons"] = [quant.get("grade_text") or "مدل روی تاریخچهٔ این نماد از نرخ پایه بهتر نبوده است"]
        base["steps"] = ["روی این نماد/تایم‌فریم وارد نشو.",
                         "نماد یا تایم‌فریم دیگری را تحلیل کن، یا بعد از چند کندل دوباره بگیر."]
        return base
    if direction is None:
        text = "رنج" if lean == "range" else "برتری مشخص بین سناریوها"
        base["headline"] = "فعلاً معامله نکن و صبر کن؛ " + ("بازار بیشتر رنج است." if lean == "range"
                                                       else "هیچ سناریویی برتری روشن ندارد.")
        base["reasons"] = [f"{text} وجود ندارد" if lean != "range" else "محتمل‌ترین سناریو رنج است"]
        base["steps"] = [f"منتظر بسته‌شدن کندل بعدی {tf} بمان و دوباره تحلیل بگیر."]
        return base
    scenario = next((s for s in report["scenarios"] if s["direction"] == direction), None)
    if scenario is None:
        base["headline"] = "سناریوی معاملاتی قابل‌سنجش پیدا نشد؛ معامله نکن."
        base["reasons"] = ["برای جهت غالب، سطح ورود/ابطال معتبر ساخته نشد"]
        return base
    blockers = _blockers(report, scenario, tradable)
    atr = quant["atr"]
    trigger = scenario["trigger"]
    if trigger is None:
        trigger = scenario["entry"] + (CHASE_ATR * atr if direction == "long" else -CHASE_ATR * atr)
    expires = max(2, quant["horizon_bars"] // 2)
    plan = {
        "direction": direction, "tf": tf, "trigger": trigger, "invalidation": scenario["stop"],
        "stop": scenario["stop"], "target1": scenario["target1"], "target2": scenario["target2"],
        "rr1": scenario["rr1"], "p_target": scenario["odds"]["target"], "expectancy_r": scenario["expectancy_net_r"],
        "expires_bars": expires, "reference_price": scenario["entry"],
    }
    side_word = "بالای" if direction == "long" else "پایین"
    against = "پایین" if direction == "long" else "بالای"
    steps = [
        "الان وارد نشو؛ سناریو هنوز تأیید نشده است.",
        f"شرط ورود {WORD[direction]}: یک کندل {tf} {side_word} {_fmt(trigger, digits)} «ببندد» "
        f"(لمس سایه کافی نیست).",
        f"بعد از تأیید: حد ضرر {_fmt(scenario['stop'], digits)}، هدف اول {_fmt(scenario['target1'], digits)} "
        f"(R:R حدود {scenario['rr1']:.1f}).",
        f"اگر کندلی {against} {_fmt(scenario['stop'], digits)} بست، یا تا {expires} کندل {tf} تأیید نشد، "
        "سناریو لغو است و باید تحلیل را دوباره گرفت.",
    ]
    base.update(direction=direction, plan=plan, steps=steps)
    if blockers:
        base["action"] = "watch"
        base["blockers"] = blockers
        base["soft_only"] = all(b.startswith(SOFT_BLOCKERS) for b in blockers)
        base["headline"] = (f"فقط پایش کن؛ سناریوی {WORD[direction]} محتمل‌تر است ولی برای معامله شرایط کامل نیست.")
        base["reasons"] = blockers
        base["steps"].insert(1, "چون شرایط کامل نیست، حتی با تأیید هم ورود خودکار انجام نمی‌شود؛ اگر معامله می‌کنی "
                                "ریسک را کوچک نگه دار.")
        return base
    base["action"] = "plan"
    base["auto_ok"] = True
    base["headline"] = (f"سناریوی {WORD[direction]} آماده است و همهٔ شرط‌ها برقرار است؛ فقط بعد از تأیید کندل وارد شو.")
    base["reasons"] = [
        f"احتمال رسیدن به هدف قبل از حد ضرر {scenario['odds']['target'] * 100:.0f}%",
        f"امید ریاضی خالص {scenario['expectancy_net_r']:+.2f}R",
        f"اعتبار آماری مدل: {quant['grade']}",
    ]
    return base


def describe(decision):
    """Compact Persian block for the language-model digest and the console."""
    lines = [f"تصمیم کد: {decision['headline']}"]
    if decision["reasons"]:
        lines.append("دلیل: " + "؛ ".join(decision["reasons"]))
    return "\n".join(lines)


def trigger_hit(plan, bar_close):
    """True when a closed bar confirms the plan (close beyond the trigger on the right side)."""
    if plan["direction"] == "long":
        return bar_close > plan["trigger"]
    return bar_close < plan["trigger"]


def invalidated(plan, bar_close):
    """True when a closed bar already violates the plan's invalidation level."""
    if plan["direction"] == "long":
        return bar_close < plan["invalidation"]
    return bar_close > plan["invalidation"]


def position_volume(equity, risk_pct, stop_distance, tick_size, tick_value, vol_min, vol_max, vol_step):
    """Lots risking ≈ ``risk_pct`` % of equity on ``stop_distance`` (price units); None if impossible/unsafe."""
    if min(equity, risk_pct, stop_distance, tick_size, tick_value, vol_step) <= 0:
        return None
    loss_per_lot = stop_distance / tick_size * tick_value
    raw = equity * risk_pct / 100.0 / loss_per_lot
    steps = int(raw / vol_step + 1e-9)       # epsilon: 24.999999 must count as 25
    volume = min(vol_max, steps * vol_step)
    if volume < vol_min:
        return None                     # even the smallest lot would risk more than allowed
    return round(volume, 8)
