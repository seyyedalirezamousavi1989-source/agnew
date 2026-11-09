"""Find forex symbols and timeframes in chat messages (Persian and English).

Deliberately conservative: a message only counts as a forex request when it names a
real pair (EURUSD, EUR/USD, «یورو دلار» …). A bare word such as «دلار» or «طلا» is never
enough, because in Persian those usually mean the local market, not the FX pair.
"""
import re

# market-convention order: the currency that comes first is the base of the pair
RANK = ["XAU", "XAG", "EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY",
        "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "TRY", "ZAR", "MXN", "SGD", "HKD", "CNH"]
CURRENCIES = set(RANK)
METALS = {"XAU", "XAG"}
MAX_SYMBOLS = 3

_PERSIAN_FIXES = str.maketrans({
    "ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "\u200c": " ", "\u200f": "", "\u200e": "",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})


def normalize(text):
    return (text or "").translate(_PERSIAN_FIXES)


def canonical_pair(a, b):
    """Market-convention symbol for two currency codes, plus whether the input was reversed."""
    if RANK.index(a) <= RANK.index(b):
        return a + b, False
    return b + a, True


# ---- English / code patterns ----------------------------------------------------------

_CODE = "|".join(sorted(CURRENCIES, key=len, reverse=True))
# EURUSD / EUR/USD / EUR-USD / EUR_USD / "EUR vs USD" — case-insensitive; a plain
# space is NOT a separator here (see _PAIR_SPACED)
_PAIR_STRICT = re.compile(
    rf"(?<![A-Za-z])({_CODE})(?:\s*[/\-_]\s*|\s+vs\.?\s+|)({_CODE})(?![A-Za-z])", re.I)
# "EUR USD" with only spaces. The English word «try» is also the code TRY, so a
# lower-case «try» next to a code («try USD/JPY») is not treated as a pair.
_PAIR_SPACED = re.compile(rf"(?<![A-Za-z])({_CODE})[ ]+({_CODE})(?![A-Za-z])", re.I)

_GOLD = re.compile(r"(?<![A-Za-z])XAU\s*[/\-_]?\s*USD(?![A-Za-z])|اونس\s+طلا|طلای\s+جهانی", re.I)
_SILVER = re.compile(r"(?<![A-Za-z])XAG\s*[/\-_]?\s*USD(?![A-Za-z])|اونس\s+نقره|نقره(?:ی)?\s+جهانی", re.I)

# ---- Persian names --------------------------------------------------------------------

_FA_NAMES = [
    ("دلار استرالیا", "AUD"), ("دلار استرالیایی", "AUD"),
    ("دلار نیوزیلند", "NZD"), ("دلار نیوزلند", "NZD"),
    ("دلار کانادا", "CAD"), ("دلار کانادایی", "CAD"),
    ("فرانک سوئیس", "CHF"), ("فرانک", "CHF"),
    ("یورو", "EUR"), ("پوند", "GBP"), ("دلار", "USD"), ("ین", "JPY"),
]
_FA_ALT = "|".join(name for name, _ in sorted(_FA_NAMES, key=lambda item: -len(item[0])))
_FA_CODE = {name: code for name, code in _FA_NAMES}
_FA_B = r"(?<![\u0600-\u06FF])"
_FA_E = r"(?![\u0600-\u06FF])"
# explicit separator: «یورو/دلار», «دلار به ین», «پوند در برابر دلار»
_FA_PAIR_SEP = re.compile(
    rf"{_FA_B}({_FA_ALT})\s*(?:/|-|\bبه\b|در\s+برابر|مقابل)\s*({_FA_ALT}){_FA_E}")
# plain space or glued («یورودلار»): only accepted in conventional order (یورو دلار, دلار ین …)
_FA_PAIR_PLAIN = re.compile(rf"{_FA_B}({_FA_ALT})\s*({_FA_ALT}){_FA_E}")


def _persian_pairs(text):
    found = []
    for match in _FA_PAIR_SEP.finditer(text):
        a, b = _FA_CODE[match.group(1)], _FA_CODE[match.group(2)]
        if a != b:
            found.append((match.start(), a, b))
    covered = [(m.start(), m.end()) for m in _FA_PAIR_SEP.finditer(text)]
    for match in _FA_PAIR_PLAIN.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        a, b = _FA_CODE[match.group(1)], _FA_CODE[match.group(2)]
        if a != b and RANK.index(a) < RANK.index(b):
            found.append((match.start(), a, b))
    return found


def detect_symbols(text, limit=MAX_SYMBOLS):
    """Symbols named in ``text`` in order of appearance: [{"symbol", "reversed"}]."""
    text = normalize(text)
    hits = []  # (position, base, quote)
    taken = []
    for pattern in (_PAIR_STRICT, _PAIR_SPACED):
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in taken):
                continue
            if pattern is _PAIR_SPACED and "try" in (match.group(1), match.group(2)):
                continue
            a, b = match.group(1).upper(), match.group(2).upper()
            if a == b:
                continue
            if a in METALS and b != "USD":
                continue
            if b in METALS and a != "USD":
                continue
            taken.append((match.start(), match.end()))
            hits.append((match.start(), a, b))
    for pattern, code in ((_GOLD, "XAU"), (_SILVER, "XAG")):
        for match in pattern.finditer(text):
            if not any(start <= match.start() < end for start, end in taken):
                taken.append((match.start(), match.end()))
                hits.append((match.start(), code, "USD"))
    hits.extend(_persian_pairs(text))
    hits.sort(key=lambda item: item[0])
    symbols, seen = [], set()
    for _position, a, b in hits:
        if a in METALS or b in METALS:
            symbol, reversed_ = (a + b, False) if a in METALS else (b + a, True)
        else:
            symbol, reversed_ = canonical_pair(a, b)
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append({"symbol": symbol, "reversed": reversed_})
        if len(symbols) >= limit:
            break
    return symbols


# ---- cues for follow-up questions -----------------------------------------------------

_STRONG_CUE = re.compile(
    r"(?:تحلیل|سیگنال|روند|حمایت|مقاومت|نقطه\s*(?:ورود|خروج)|استاپ|حد\s*(?:ضرر|سود)|تارگت|"
    r"پیش\s*بینی|اندیکاتور|تایم\s*فریم|بخرم|بفروشم|بخریم|بفروشیم|لانگ|شورت|ست\s*آپ|اسپرد|پیوت|"
    r"analy[sz]|signal|\btrend|support|resistance|\bentry\b|\btargets?\b|stop\s*loss|\btp\b|\bsl\b|"
    r"forecast|outlook|indicator|timeframe|\b(?:buy|sell)\b|go\s+(?:long|short)|"
    r"(?:long|short)\s+(?:position|trade|setup)|\bsetup\b|\bspread\b|pivot|"
    r"bullish|bearish|\brsi\b|\bmacd\b|\bema\b|\batr\b)", re.I)


def has_forex_cue(text):
    return bool(_STRONG_CUE.search(normalize(text)))


# ---- timeframes -----------------------------------------------------------------------

DEFAULT_TIMEFRAMES = ("M15", "H1", "H4", "D1")
_TF_CODE = re.compile(r"(?<![A-Za-z0-9])(?:(M(?:1|5|15|30))|(H(?:1|4))|(D1)|(W1))(?![A-Za-z0-9])", re.I)
_TF_SHORT = re.compile(r"(?<![A-Za-z0-9])(1|5|15|30)(?:m|\s*min(?:ute)?s?)(?![A-Za-z])|"
                       r"(?<![A-Za-z0-9])(1|4)(?:h|\s*(?:hr|hour)s?)(?![A-Za-z])", re.I)
_TF_FA = [
    (re.compile(r"(?<!\d)1\s*دقیقه|(?<![\u0600-\u06FF])یک\s*دقیقه"), "M1"),
    (re.compile(r"(?<!\d)5\s*دقیقه|(?<![\u0600-\u06FF])پنج\s*دقیقه"), "M5"),
    (re.compile(r"(?<!\d)15\s*دقیقه|(?<![\u0600-\u06FF])پانزده\s*دقیقه|(?<![\u0600-\u06FF])ربع\s*ساعت"), "M15"),
    (re.compile(r"(?<!\d)30\s*دقیقه|(?<![\u0600-\u06FF])سی\s*دقیقه|نیم\s*ساعت"), "M30"),
    (re.compile(r"(?<!\d)60\s*دقیقه|شصت\s*دقیقه"), "H1"), (re.compile(r"(?<!\d)240\s*دقیقه"), "H4"),
    # «تایم فریم 15» / «تایم‌فریم ۵» / «tf 30» — a bare number after the word timeframe means minutes
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*(?:=\s*)?1(?!\d)(?!\s*(?:ساعت|h))", re.I), "M1"),
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*5(?!\d)(?!\s*(?:ساعت|h))", re.I), "M5"),
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*15(?!\d)(?!\s*(?:ساعت|h))", re.I), "M15"),
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*30(?!\d)(?!\s*(?:ساعت|h))", re.I), "M30"),
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*60(?!\d)", re.I), "H1"),
    (re.compile(r"(?:تایم\s*فریم|تایمفریم|time\s*frame|\btf)\s*:?\s*240(?!\d)", re.I), "H4"),
    (re.compile(r"(?:\b1|یک)\s*ساعته|ساعتی"), "H1"), (re.compile(r"(?<!\d)4\s*ساعته|چهار\s*ساعته"), "H4"),
    (re.compile(r"روزانه|دیلی"), "D1"), (re.compile(r"هفتگی"), "W1"),
]
_TF_EN = [(re.compile(r"\bdaily\b", re.I), "D1"), (re.compile(r"\bweekly\b", re.I), "W1"),
          (re.compile(r"\bhourly\b", re.I), "H1")]
_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]


def detect_timeframes(text):
    """Timeframes the user asked for explicitly (sorted short → long)."""
    text = normalize(text)
    found = set()
    for match in _TF_CODE.finditer(text):
        found.add(match.group(0).upper())
    for match in _TF_SHORT.finditer(text):
        if match.group(1):
            found.add("M" + match.group(1))
        else:
            found.add("H" + match.group(2))
    for pattern, name in _TF_FA + _TF_EN:
        if pattern.search(text):
            found.add(name)
    return [name for name in _TF_ORDER if name in found]


def merge_timeframes(requested, limit=6):
    """Default set plus what the user asked for, capped to keep the prompt small."""
    chosen = list(DEFAULT_TIMEFRAMES)
    for name in requested:
        if name not in chosen:
            chosen.append(name)
    chosen.sort(key=_TF_ORDER.index)
    if len(chosen) > limit:
        # drop defaults that the user did not ask for, from the shortest timeframe up
        for name in list(chosen):
            if len(chosen) <= limit:
                break
            if name not in requested and name != "D1":
                chosen.remove(name)
    return chosen


# ---- whole conversation ---------------------------------------------------------------

CARRY_MESSAGES = 6


def symbols_for_messages(messages):
    """(symbols, carried) for the newest user message.

    A follow-up such as «تحلیل H4 و سطح‌هاش چیه؟» has no pair of its own: when it has a
    clear trading cue, the pair(s) of the last few user messages are used again.
    """
    users = [m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if not users:
        return [], False
    current = detect_symbols(users[-1])
    if current:
        return current, False
    if not has_forex_cue(users[-1]):
        return [], False
    for earlier in reversed(users[:-1][-CARRY_MESSAGES:]):
        previous = detect_symbols(earlier)
        if previous:
            return previous, True
    return [], False


# ---- trading profile: the timeframe the user trades on ---------------------------------
#
# «برای ترید در تایم فریم ۱۵ دقیقه» must not be forgotten on the next message. The client
# sends the whole conversation with every request, so the profile is derived from the
# history (newest statement wins) and costs no stored state and no extra prompt text.

LADDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
DEFAULT_TRADING_TF = "M15"

# timeframes shown to the analyst for each trading timeframe: one lower "trigger" frame,
# the trading frame itself and up to three higher frames for bias and structure
TF_PLAN = {
    "M1": ["M1", "M5", "M15", "H1"],
    "M5": ["M1", "M5", "M15", "H1", "H4"],
    "M15": ["M5", "M15", "H1", "H4", "D1"],
    "M30": ["M15", "M30", "H1", "H4", "D1"],
    "H1": ["M15", "H1", "H4", "D1", "W1"],
    "H4": ["H1", "H4", "D1", "W1"],
    "D1": ["H4", "D1", "W1"],
    "W1": ["D1", "W1"],
}
# frames that vote for the higher-timeframe bias of the probability model
HTF_CHAIN = {
    "M1": ["M5", "M15"], "M5": ["M15", "H1"], "M15": ["H1", "H4"], "M30": ["H1", "H4"],
    "H1": ["H4", "D1"], "H4": ["D1", "W1"], "D1": ["W1"], "W1": [],
}
STYLE_OF = {"M1": "scalp", "M5": "scalp", "M15": "intraday", "M30": "intraday",
            "H1": "intraday", "H4": "swing", "D1": "swing", "W1": "position"}
STYLE_LABEL = {"scalp": "اسکالپ", "intraday": "اینترادی", "swing": "سوینگ", "position": "پوزیشن"}

_TRADE_CUE = re.compile(
    r"(?:ترید|معامله|اسکالپ|ورود|سیگنال|ست\s*آپ|trade|trading|scalp|entry|setup|signal)", re.I)
_STYLE_WORDS = [
    (re.compile(r"اسکالپ|scalp", re.I), "M5"),
    (re.compile(r"سوینگ|swing", re.I), "H4"),
    (re.compile(r"اینترادی|intra\s*day|دی\s*ترید|day\s*trad", re.I), "M15"),
]


def plan_timeframes(trading_tf, extra=()):
    """Timeframes to fetch for a trading timeframe (plus anything the user asked for)."""
    chosen = list(TF_PLAN.get(trading_tf, TF_PLAN[DEFAULT_TRADING_TF]))
    for name in extra:
        if name in LADDER and name not in chosen and len(chosen) < 6:
            chosen.append(name)
    return sorted(chosen, key=LADDER.index)


def _pick_trading_tf(text, found):
    """The trading timeframe among the timeframes named in one message."""
    if len(found) == 1:
        return found[0]
    normalized = normalize(text)
    cues = [m.start() for m in _TRADE_CUE.finditer(normalized)]
    if cues:
        best, best_distance = None, None
        for name in found:
            for pattern in _timeframe_patterns(name):
                for match in pattern.finditer(normalized):
                    distance = min(abs(match.start() - cue) for cue in cues)
                    if best_distance is None or distance < best_distance:
                        best, best_distance = name, distance
        if best:
            return best
    return found[0]   # several frames without a trading cue: the lowest one is the one traded


def _timeframe_patterns(name):
    patterns = [pattern for pattern, tf in _TF_FA + _TF_EN if tf == name]
    patterns.append(re.compile(rf"(?<![A-Za-z0-9]){name}(?![A-Za-z0-9])", re.I))
    short = {"M1": r"1\s*m(?:in\w*)?", "M5": r"5\s*m(?:in\w*)?", "M15": r"15\s*m(?:in\w*)?",
             "M30": r"30\s*m(?:in\w*)?", "H1": r"1\s*h(?:r|our)?", "H4": r"4\s*h(?:r|our)?"}.get(name)
    if short:
        patterns.append(re.compile(rf"(?<![A-Za-z0-9]){short}(?![A-Za-z])", re.I))
    return patterns


def trading_profile(messages, scan=40):
    """What the user said about how they trade: {"timeframe", "explicit", "style", "source"}.

    Newest statement wins; an explicit timeframe beats a style word («اسکالپ»). Without
    any statement the default timeframe is returned with ``explicit`` False.
    """
    users = [m["content"] for m in messages
             if m.get("role") == "user" and isinstance(m.get("content"), str)][-scan:]
    for text in reversed(users):
        found = detect_timeframes(text)
        if found:
            tf = _pick_trading_tf(text, found)
            return {"timeframe": tf, "explicit": True, "style": STYLE_OF[tf], "source": "timeframe"}
        normalized = normalize(text)
        for pattern, tf in _STYLE_WORDS:
            if pattern.search(normalized):
                return {"timeframe": tf, "explicit": True, "style": STYLE_OF[tf], "source": "style"}
    return {"timeframe": DEFAULT_TRADING_TF, "explicit": False,
            "style": STYLE_OF[DEFAULT_TRADING_TF], "source": "default"}
