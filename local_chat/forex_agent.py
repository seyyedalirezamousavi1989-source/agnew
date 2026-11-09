"""The local language model as the trader's decision-maker — inside hard guard rails.

The code builds the market picture and a concrete scenario (trigger, stop, targets, empirical
odds). When that scenario is *confirmed* by a closed candle, the model is asked ONE question:
take it or skip it (and at what risk). Its answer is JSON that is validated here.

What the model can do:   take / skip, normal or reduced risk, give a reason.
What it can never do:    change levels, raise the risk above the configured value, trade when the
                         code says "no trade", trade through a hard blocker (closed market, news,
                         spread), or act on an unreadable/low-confidence answer (that is a skip).

Modes: ``veto`` (default) — only scenarios the rules fully approved are shown to the model.
       ``wide`` — scenarios blocked only by soft reasons (weak validation, low expectancy,
       chart conflict) are shown too; if the model takes one the risk is forced to the reduced size.
The backtest measures whether the model's choices beat "take everything the rules approve".
"""
import json
import os
import re
import time
from urllib.request import Request, urlopen

if __package__:
    from . import forex_desk
else:  # pragma: no cover
    import forex_desk

MODES = ("veto", "wide")
MIN_CONFIDENCE = 55
REDUCED_RISK = 0.5
MAX_REASON = 240
DIGEST_CHARS = 3200
OLLAMA_URL = os.environ.get("LOCALCHAT_OLLAMA_URL", "http://127.0.0.1:11434")

AGENT_PROMPT = (
    "تو معامله‌گر ارشد بازارهای مالی هستی و فقط یک تصمیم می‌گیری: سناریوی پیشنهادیِ کد را «بگیر» (take) یا «رد کن» (skip). "
    "تحلیل کمّی و سناریو با کد ساخته شده و سطوح (ورود، حد ضرر، هدف) ثابت‌اند؛ تو نمی‌توانی آن‌ها را عوض کنی. "
    "قواعد: (۱) فقط از اعداد همان گزارش استفاده کن. (۲) به این‌ها وزن بده: هم‌جهتی تایم‌فریم‌های بالاتر، اعتبار آماری مدل، "
    "احتمال هدف و امید ریاضی، سشن، تعارض با چارت، خبر و اسپرد. (۳) اگر کمترین تردید جدی داری، skip؛ نگرفتن معامله هزینه ندارد، "
    "گرفتن معاملهٔ بد دارد. (۴) اگر معامله را می‌گیری و شرایط مرزی است، risk را reduced بگذار. "
    "خروجی فقط یک شیء JSON با همین کلیدها و بدون هیچ متن دیگر: "
    '{"action":"take|skip","risk":"normal|reduced","confidence":0-100,"reason":"حداکثر دو جملهٔ فارسی"}'
)


SCENARIO_PROMPT = (
    "تو معامله‌گر ارشد بازارهای مالی هستی. از روی تحلیل داده‌شده یک سناریوی معاملاتی شرطی بچین که اسکریپت رصد کند. "
    "سناریو یعنی: اگر یک کندل بسته‌شده «بالای/پایین trigger» بست، وارد شو؛ stop سطح ابطال و حد ضرر است؛ target هدف. "
    "همهٔ قیمت‌ها باید روی همان مقیاس قیمت گزارش باشند و از سطوح حمایت/مقاومت و ساختار همان گزارش بیایند. "
    "p_target برآورد صادقانهٔ خودت (درصد) از احتمال رسیدن به target قبل از stop است. "
    "اگر هیچ ستاپ معقولی نمی‌بینی bias را none بگذار. خروجی فقط یک شیء JSON بدون متن دیگر: "
    '{"bias":"long|short|none","trigger":قیمت,"stop":قیمت,"target":قیمت,"p_target":0-100,"expires_bars":2-12,"reason":"حداکثر دو جملهٔ فارسی"}'
)
# the model may choose levels, inside these bounds (ATR multiples) — everything else is rejected
MIN_RISK_ATR, MAX_RISK_ATR, MAX_REWARD_ATR, MIN_RR = 0.5, 3.5, 8.0, 1.0
TRIGGER_BEHIND_ATR, TRIGGER_AHEAD_ATR = 0.3, 2.0


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def validate_scenario(data, price, atr, tf):
    """Plan dict (same shape as ``forex_decision`` plans) from a model proposal, or ``(None, reason)``."""
    if not isinstance(data, dict) or atr is None or atr <= 0:
        return None, "پیشنهاد نامعتبر"
    bias = str(data.get("bias", "")).strip().lower()
    if bias == "none":
        return None, "مدل ستاپی ندید"
    if bias not in ("long", "short"):
        return None, "جهت نامعتبر"
    trigger, stop, target = (_number(data.get(k)) for k in ("trigger", "stop", "target"))
    if None in (trigger, stop, target):
        return None, "قیمت‌ها نامعتبر"
    sign = 1 if bias == "long" else -1
    entry = price + sign * max(0.0, sign * (trigger - price))          # the price a confirming close would be near
    if not (-TRIGGER_BEHIND_ATR * atr <= sign * (trigger - price) <= TRIGGER_AHEAD_ATR * atr):
        return None, "سطح تأیید خیلی دور از قیمت است"
    risk = sign * (entry - stop)
    reward = sign * (target - entry)
    if not MIN_RISK_ATR * atr <= risk <= MAX_RISK_ATR * atr:
        return None, "فاصلهٔ حد ضرر خارج از محدودهٔ مجاز (۰٫۵ تا ۳٫۵ ATR) است"
    if reward < MIN_RR * risk or reward > MAX_REWARD_ATR * atr:
        return None, "هدف خارج از محدودهٔ مجاز است (R:R حداقل ۱ و حداکثر ۸ ATR)"
    p = _number(data.get("p_target"))
    p = min(0.9, max(0.05, (p if p is not None else 40.0) / 100.0))
    expires = _number(data.get("expires_bars"))
    expires = int(min(12, max(2, expires))) if expires is not None else 6
    rr = reward / risk
    return {
        "direction": bias, "tf": tf, "trigger": trigger, "invalidation": stop, "stop": stop, "target1": target,
        "target2": target, "rr1": rr, "p_target": p, "expectancy_r": p * rr - (1 - p), "expires_bars": expires,
        "reference_price": price, "source": "model", "reason": str(data.get("reason", "")).strip()[:MAX_REASON],
    }, ""


def synthetic_decision(plan):
    """A decision wrapper so the model's own scenario runs through the same PlanTracker."""
    if plan is None:
        return {"action": "no_trade", "direction": None, "headline": "", "reasons": [], "blockers": [], "plan": None,
                "steps": [], "auto_ok": False, "soft_only": False}
    return {"action": "plan", "direction": plan["direction"], "headline": "سناریوی مدل", "reasons": [], "blockers": [],
            "plan": dict(plan), "steps": [], "auto_ok": True, "soft_only": False}


def ollama_chat(model, messages, options, timeout=300):
    """Standalone non-streaming Ollama chat (CLI use). The server passes its own function instead."""
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": "5m", "think": False,
            "options": {"num_ctx": 8192, **options}}
    for attempt in range(2):
        request = Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "")
        except Exception:  # noqa: BLE001
            if attempt == 0 and "think" in body:
                body.pop("think")            # models without thinking support reject the field
                continue
            raise
    return ""


def parse_reply(text):
    """Validated verdict from the model's text, or None when it is unusable."""
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in ("take", "skip"):
        return None
    risk = str(data.get("risk", "normal")).strip().lower()
    try:
        confidence = int(float(data.get("confidence", 0)))
    except (TypeError, ValueError):
        confidence = 0
    return {"action": action, "risk": "reduced" if risk == "reduced" else "normal",
            "confidence": max(0, min(100, confidence)), "reason": str(data.get("reason", "")).strip()[:MAX_REASON]}


def build_messages(report, decision):
    plan = decision["plan"]
    digits = report["snapshot"]["digits"]
    side = "خرید (long)" if plan["direction"] == "long" else "فروش (short)"
    lines = [
        forex_desk.format_digest(report, DIGEST_CHARS),
        "",
        f"سناریوی پیشنهادی کد (همین حالا کندل {plan['tf']} تأیید کرد): {side} | حد ضرر {plan['stop']:.{digits}f} | "
        f"هدف اول {plan['target1']:.{digits}f} | R:R {plan['rr1']:.1f} | احتمال هدف قبل از حد ضرر {plan['p_target'] * 100:.0f}% | "
        f"امید ریاضی خالص {plan['expectancy_r']:+.2f}R",
    ]
    if decision["blockers"]:
        lines.append("موانع نرم (قواعد کد کامل راضی نیستند): " + "؛ ".join(decision["blockers"]))
    lines.append("تصمیمت را فقط به‌صورت JSON بده.")
    return [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


class AgentDecider:
    """Asks the model about confirmed scenarios and applies the guard rails (see module docstring)."""

    def __init__(self, chat, model, mode="veto", min_confidence=MIN_CONFIDENCE):
        if mode not in MODES:
            raise ValueError("mode must be veto or wide")
        self.chat, self.model, self.mode, self.min_confidence = chat, model, mode, min_confidence
        self.stats = {"calls": 0, "taken": 0, "skipped": 0, "invalid": 0, "errors": 0, "seconds": 0.0}

    def propose(self, report):
        """The model's own conditional scenario: ``(plan or None, note)``. Levels are validated against ATR bounds."""
        snapshot = report["snapshot"]
        tf = report["timeframe"]
        frame = snapshot["timeframes"][tf]
        price, atr, digits = frame.get("close"), frame.get("atr"), snapshot["digits"]
        if price is None or not atr or atr <= 0:
            return None, "ATR یا قیمت در دسترس نیست"
        user = (forex_desk.format_digest(report, DIGEST_CHARS)
                + f"\n\nقیمت فعلی {price:.{digits}f} | ATR {atr:.{digits}f} (فاصلهٔ stop باید بین ۰٫۵ تا ۳٫۵ ATR باشد). "
                "سناریوی شرطی‌ات را فقط به‌صورت JSON بده.")
        self.stats["calls"] += 1
        started = time.time()
        try:
            text = self.chat(self.model, [{"role": "system", "content": SCENARIO_PROMPT}, {"role": "user", "content": user}],
                             {"temperature": 0, "num_predict": 400})
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self.stats["seconds"] += time.time() - started
            return None, f"مدل پاسخ نداد ({type(exc).__name__})"
        self.stats["seconds"] += time.time() - started
        match_start, match_end = (text or "").find("{"), (text or "").rfind("}")
        data = None
        if match_start >= 0 and match_end > match_start:
            try:
                data = json.loads(re.sub(r"<think>.*?</think>", "", text, flags=re.S)[
                    (re.sub(r"<think>.*?</think>", "", text, flags=re.S)).find("{"):
                    (re.sub(r"<think>.*?</think>", "", text, flags=re.S)).rfind("}") + 1])
            except ValueError:
                data = None
        if data is None:
            self.stats["invalid"] += 1
            return None, "پاسخ مدل JSON معتبر نبود"
        plan, note = validate_scenario(data, price, atr, tf)
        if plan is None and note not in ("مدل ستاپی ندید",):
            self.stats["invalid"] += 1
        return plan, note

    def accepts(self, decision):
        """Which code decisions may be shown to the model."""
        if decision["action"] == "plan":
            return True
        return self.mode == "wide" and decision["action"] == "watch" and decision.get("soft_only", False)

    def decide(self, report, decision):
        """``{"action": take|skip, "risk_mult", "confidence", "reason", "called"}`` — always safe to act on."""
        if decision["plan"] is None or not self.accepts(decision):
            return {"action": "skip", "risk_mult": 0.0, "confidence": 0, "called": False,
                    "reason": "قواعد کد این سناریو را برای تصمیم مدل قابل‌قبول نمی‌دانند."}
        started = time.time()
        self.stats["calls"] += 1
        try:
            text = self.chat(self.model, build_messages(report, decision), {"temperature": 0, "num_predict": 300})
        except Exception as exc:  # noqa: BLE001 - a dead model must mean "no trade", never a crash
            self.stats["errors"] += 1
            self.stats["seconds"] += time.time() - started
            return {"action": "skip", "risk_mult": 0.0, "confidence": 0, "called": True,
                    "reason": f"مدل پاسخ نداد ({type(exc).__name__}); برای ایمنی رد شد."}
        self.stats["seconds"] += time.time() - started
        verdict = parse_reply(text)
        if verdict is None:
            self.stats["invalid"] += 1
            return {"action": "skip", "risk_mult": 0.0, "confidence": 0, "called": True,
                    "reason": "پاسخ مدل JSON معتبر نبود؛ برای ایمنی رد شد."}
        if verdict["action"] == "take" and verdict["confidence"] < self.min_confidence:
            self.stats["skipped"] += 1
            return {"action": "skip", "risk_mult": 0.0, "confidence": verdict["confidence"], "called": True,
                    "reason": f"اطمینان مدل ({verdict['confidence']}) کمتر از حداقل {self.min_confidence} بود. " + verdict["reason"]}
        if verdict["action"] == "skip":
            self.stats["skipped"] += 1
            return {"action": "skip", "risk_mult": 0.0, "confidence": verdict["confidence"], "called": True,
                    "reason": verdict["reason"]}
        reduced = verdict["risk"] == "reduced" or decision["action"] != "plan"
        self.stats["taken"] += 1
        return {"action": "take", "risk_mult": REDUCED_RISK if reduced else 1.0, "confidence": verdict["confidence"],
                "called": True, "reason": verdict["reason"]}
