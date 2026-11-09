"""Check, on YOUR broker's real history, where the probability model has an edge.

    python -m local_chat.forex_validate EURUSD GBPUSD XAUUSD --tf M15 H1 --out report.json

For every symbol × timeframe the desk fetches deep history from MetaTrader 5, runs the
walk-forward test and prints: how many past states were tested, Brier skill against the
plain base rate (> 0 means the model beats it), calibration error (how far stated
percentages are from what really happened) and the verdict. Pairs whose verdict is
"بدون لبه" should not be traded on these percentages.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

if __package__:
    from . import forex_desk, mt5_data
    from . import settings as app_settings
else:  # pragma: no cover - script usage
    import forex_desk
    import mt5_data
    import settings as app_settings


def validate(symbols, timeframes, config, analyze=None, now=None):
    """Rows of results; ``analyze`` is injectable for tests."""
    analyze = analyze or forex_desk.analyze_symbol
    quiet = {**config, "news_enabled": False}          # news never changes the walk-forward result
    rows = []
    for symbol in symbols:
        for tf in timeframes:
            row = {"symbol": symbol, "timeframe": tf}
            try:
                report = analyze(symbol, tf, quiet, now=now or datetime.now(timezone.utc), deep=True)
            except (mt5_data.MT5Error, forex_desk.DeskError) as exc:
                row["error"] = exc.message
                rows.append(row)
                continue
            quant = report["quant"]
            if quant is None:
                row["error"] = "تاریخچهٔ کافی نبود."
            else:
                validation = quant["validation"] or {}
                row.update({
                    "history_bars": quant["history_bars"],
                    "tested": validation.get("tested"),
                    "skill_pct": None if not validation else round(validation["skill"] * 100, 2),
                    "ece_pct": None if not validation or validation.get("ece") is None else round(validation["ece"] * 100, 1),
                    "favourite_hit_pct": None if not validation else round(validation["favourite_hit_rate"] * 100, 1),
                    "grade": quant["grade"],
                })
            rows.append(row)
    return rows


def render(rows):
    head = f"{'symbol':<9}{'tf':<5}{'bars':>6}{'tested':>8}{'skill%':>8}{'ECE%':>7}{'hit%':>7}  verdict"
    lines = [head, "-" * len(head)]
    for row in rows:
        if "error" in row:
            lines.append(f"{row['symbol']:<9}{row['timeframe']:<5}  {row['error']}")
            continue
        lines.append(f"{row['symbol']:<9}{row['timeframe']:<5}{row['history_bars']:>6}{row['tested'] or 0:>8}"
                     f"{row['skill_pct'] if row['skill_pct'] is not None else '-':>8}"
                     f"{row['ece_pct'] if row['ece_pct'] is not None else '-':>7}"
                     f"{row['favourite_hit_pct'] if row['favourite_hit_pct'] is not None else '-':>7}  {row['grade']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="+", help="e.g. EURUSD GBPUSD XAUUSD")
    parser.add_argument("--tf", nargs="+", default=["M15"], help="trading timeframes, e.g. M5 M15 H1 H4")
    parser.add_argument("--out", help="also write the rows as JSON to this file")
    args = parser.parse_args(argv)
    rows = validate([s.upper() for s in args.symbols], [t.upper() for t in args.tf], app_settings.forex_settings())
    print(render(rows))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
