"""Separate watcher: re-checks the market on every closed candle and acts when the scenario happens.

    python -m local_chat.forex_watch EURUSD --tf M15                 # alert only (default, no orders)
    python -m local_chat.forex_watch EURUSD --tf M15 --mode demo     # orders on a DEMO account only
    python -m local_chat.forex_watch EURUSD --tf M15 --mode demo --agent --model qwen3:8b   # the local model takes/skips
    python -m local_chat.forex_watch EURUSD --tf M15 --mode live --i-accept-live-risk

Flow (once per closed candle of the trading timeframe)
1. Fresh deep analysis + the rule-based decision (``forex_decision``; no language model).
2. No armed plan and the decision is ``plan`` → arm it: trigger, invalidation, expiry.
3. Armed plan: a candle closes beyond the trigger → re-check the fresh decision; if it still
   says ``plan`` in the same direction, the scenario "happened": notify (alert) or place the
   order (demo/live) with a mandatory stop-loss and target. Invalidated or expired → cancel.

Safety gates for orders: demo account unless live is explicitly accepted, algo-trading enabled in
the terminal, one position per symbol, risk per trade (default 0.5 % of equity), max trades and max
loss per day, STOP file kill switch, no order when the decision has any blocker (weak model,
news, spread, closed market). It is a tool, not financial advice; results can lose money.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

if __package__:
    from . import forex_analysis as fa
    from . import forex_agent, forex_decision, forex_desk, mt5_data, mt5_trade
    from . import settings as app_settings
else:  # pragma: no cover
    import forex_analysis as fa
    import forex_agent
    import forex_decision
    import forex_desk
    import mt5_data
    import mt5_trade
    import settings as app_settings

MAGIC = 770011
MODES = ("alert", "demo", "live")
POLL_SECONDS = 20
BAR_GRACE_SECONDS = 4


class WatchLog:
    """Console + JSON-lines log (one event per line)."""

    def __init__(self, path=None, echo=print):
        self.path, self.echo = path, echo

    def event(self, kind, message, **data):
        row = {"time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "event": kind, "message": message, **data}
        self.echo(f"[{row['time']}] {kind}: {message}")
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class PlanTracker:
    """The scenario state machine shared by the live watcher and the backtest (identical rules).

    ``advance(decision, close)`` is called once per closed candle and returns ``(outcome, detail)``:
    idle / armed / waiting / cancelled / expired / rejected / confirmed. For ``confirmed`` the detail
    is the *fresh* plan (levels recomputed on the confirming candle) that must be traded.
    """

    def __init__(self, plan=None, age=0, wide=False):
        self.plan, self.age, self.wide = plan, age, wide

    def _usable(self, decision):
        """A decision that may be armed/confirmed: ``plan``; in wide mode also softly-blocked ``watch``."""
        return decision["action"] == "plan" or (self.wide and decision["action"] == "watch"
                                                and decision.get("soft_only", False) and decision["plan"] is not None)

    def _clear(self):
        self.plan, self.age = None, 0

    def advance(self, decision, close):
        plan = self.plan
        if plan:
            self.age += 1
            if forex_decision.invalidated(plan, close):
                self._clear()
                return "cancelled", None
            if forex_decision.trigger_hit(plan, close):
                direction = plan["direction"]
                self._clear()
                fresh = decision["plan"]
                if not self._usable(decision) or fresh is None or fresh["direction"] != direction:
                    return "rejected", None
                return "confirmed", fresh
            if self.age >= plan["expires_bars"]:
                self._clear()
                return "expired", None
            return "waiting", None
        if self._usable(decision):
            self.plan, self.age = decision["plan"], 0
            return "armed", decision["plan"]
        return "idle", None


class Watcher:
    def __init__(self, symbol, tf, config, mode="alert", risk_pct=0.5, max_trades=3, max_daily_loss_pct=2.0,
                 stop_file="STOP", state_path=None, log=None, analyze=None, trade=mt5_trade, clock=time.time, agent=None,
                 ai_scenario=False, ai_every=4):
        if mode not in MODES:
            raise ValueError("mode must be one of " + ", ".join(MODES))
        self.symbol, self.tf, self.config, self.mode = symbol, tf, config, mode
        self.risk_pct, self.max_trades, self.max_daily_loss_pct = risk_pct, max_trades, max_daily_loss_pct
        self.stop_file, self.state_path = stop_file, state_path
        self.log = log or WatchLog()
        self.analyze = analyze or (lambda: forex_desk.analyze_symbol(symbol, tf, config, deep=True))
        self.trade, self.clock, self.agent = trade, clock, agent
        self.ai_scenario, self.ai_every, self._bars_seen = bool(ai_scenario and agent), max(1, int(ai_every)), 0
        self.live_ok = False
        self.state = {"plan": None, "age": 0, "bar": None}
        self._load()

    # -- persistence -------------------------------------------------------------------
    def _load(self):
        if self.state_path and os.path.isfile(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and data.get("symbol") == self.symbol and data.get("tf") == self.tf:
                    self.state.update(plan=data.get("plan"), age=int(data.get("age", 0)), bar=data.get("bar"))
            except (OSError, ValueError):
                pass

    def _save(self):
        if self.state_path:
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump({"symbol": self.symbol, "tf": self.tf, **self.state}, handle, ensure_ascii=False)

    # -- one decision cycle (called once per new closed candle) ---------------------------
    def bar_id(self):
        return int(self.clock() // fa.TIMEFRAME_SECONDS[self.tf])

    def step(self):
        report = self.analyze()
        decision = report.get("decision")
        digits = report["snapshot"]["digits"]
        close = report["snapshot"]["timeframes"][self.tf]["close"]
        if decision is None:
            self.log.event("skip", "تحلیل عمیق نتیجه نداد.")
            return "skip"
        tracker = PlanTracker(self.state["plan"], self.state["age"], wide=bool(self.agent and self.agent.mode == "wide"))
        if self.ai_scenario:        # the model's own scenario is tracked; the rule-based decision is not used
            decision = forex_agent.synthetic_decision(self.state["plan"])
        outcome, detail = tracker.advance(decision, close)
        self.state.update(plan=tracker.plan, age=tracker.age)
        self._save()
        plan = detail if isinstance(detail, dict) else None
        if outcome == "cancelled":
            self.log.event("cancel", f"سناریو باطل شد؛ کندل {close:.{digits}f} از سطح ابطال گذشت.")
        elif outcome == "expired":
            self.log.event("expire", "سناریو در مهلت خود تأیید نشد و منقضی شد.")
        elif outcome == "waiting":
            p = tracker.plan
            self.log.event("waiting", f"در انتظار تأیید ({tracker.age}/{p['expires_bars']}) | "
                                      f"قیمت {close:.{digits}f} | شرط {p['trigger']:.{digits}f}")
        elif outcome == "armed":
            p = tracker.plan
            self.log.event("armed", f"سناریو {forex_decision.WORD[p['direction']]} فعال شد: کندل {self.tf} باید "
                                    f"{'بالای' if p['direction'] == 'long' else 'پایین'} {p['trigger']:.{digits}f} ببندد؛ "
                                    f"ابطال {p['invalidation']:.{digits}f}", plan=p)
        elif outcome == "rejected":
            self.log.event("rejected", "شرط تأیید رخ داد ولی تحلیل تازه دیگر همان سناریو را با شرایط کامل تأیید نمی‌کند: "
                                       + decision["headline"])
        elif outcome == "idle":
            self.log.event("idle", decision["headline"])
        elif outcome == "confirmed":
            if self.ai_scenario and report["snapshot"]["market"] != "open":
                self.log.event("refused", "شرط رخ داد ولی بازار باز/زنده نیست.")
                return "refused"
            return self._confirmed(report, decision, plan, close, digits)
        if self.ai_scenario and tracker.plan is None and outcome != "armed":
            self._bars_seen += 1
            if (self._bars_seen - 1) % self.ai_every == 0 and report["snapshot"]["market"] == "open":
                proposal, note = self.agent.propose(report)
                if proposal is None:
                    self.log.event("ai_idle", f"مدل سناریویی نچید: {note}")
                else:
                    proposal["reference_price"] = close
                    self.state.update(plan=proposal, age=0)
                    self._save()
                    self.log.event("armed", f"سناریو {forex_decision.WORD[proposal['direction']]} را مدل چید: کندل {self.tf} باید "
                                            f"{'بالای' if proposal['direction'] == 'long' else 'پایین'} {proposal['trigger']:.{digits}f} "
                                            f"ببندد؛ حد ضرر {proposal['stop']:.{digits}f}؛ هدف {proposal['target1']:.{digits}f}؛ "
                                            f"p≈{proposal['p_target'] * 100:.0f}% — {proposal['reason']}", plan=proposal)
                    return "armed"
        return outcome

    def _confirmed(self, report, decision, fresh, close, digits):
        word = forex_decision.WORD[fresh["direction"]]
        message = (f"سناریوی {word} رخ داد: کندل {close:.{digits}f} | حد ضرر {fresh['stop']:.{digits}f} | "
                   f"هدف {fresh['target1']:.{digits}f} | احتمال هدف {fresh['p_target'] * 100:.0f}%")
        risk_mult = 1.0
        if self.agent:
            verdict = self.agent.decide(report, decision)
            self.log.event("agent", f"مدل {self.agent.model}: {'می‌گیرد' if verdict['action'] == 'take' else 'رد می‌کند'} "
                                    f"(اطمینان {verdict['confidence']}) — {verdict['reason']}", verdict=verdict)
            if verdict["action"] != "take":
                return "agent_skip"
            risk_mult = verdict["risk_mult"]
            message += f" | تصمیم مدل: بگیر (ریسک ×{risk_mult:g})"
        if self.mode == "alert":
            self.log.event("signal", message + " — حالت هشدار است، سفارشی ارسال نشد.", plan=fresh)
            return "signal"
        return self._execute(report, fresh, message, risk_mult)

    # -- order gates -----------------------------------------------------------------
    def _execute(self, report, plan, message, risk_mult=1.0):
        cfg = self.config
        reasons = []
        if os.path.exists(self.stop_file):
            reasons.append(f"فایل {self.stop_file} وجود دارد (کلید توقف)")
        account = self.trade.account_state(cfg["terminal_path"])
        if self.mode == "demo" and account["mode"] != "demo":
            reasons.append("حالت demo فقط روی حساب دمو مجاز است؛ حساب فعلی " + account["mode"] + " است")
        if self.mode == "live" and not self.live_ok:
            reasons.append("حالت live بدون --i-accept-live-risk مجاز نیست")
        if not (account["terminal_trade_allowed"] and account["account_trade_allowed"]):
            reasons.append("معامله‌گری خودکار در ترمینال/حساب فعال نیست (دکمهٔ Algo Trading را روشن کن)")
        if reasons:
            return self._refuse(message, reasons)
        if self.trade.own_positions(self.symbol, cfg["symbol_suffix"], MAGIC, cfg["terminal_path"]):
            return self._refuse(message, ["همین نماد از قبل پوزیشن باز دارد"])
        today = self.trade.today_result(MAGIC, cfg["terminal_path"], self.clock())
        if today["trades"] >= self.max_trades:
            return self._refuse(message, [f"سقف {self.max_trades} معامله در روز پر شده"])
        if today["pnl"] <= -account["equity"] * self.max_daily_loss_pct / 100:
            return self._refuse(message, [f"سقف ضرر روزانه ({self.max_daily_loss_pct}%) رسیده"])
        info = self.trade.symbol_trade_info(self.symbol, cfg["symbol_suffix"], cfg["terminal_path"])
        stop_distance = abs(plan["reference_price"] - plan["stop"])
        volume = forex_decision.position_volume(
            account["equity"], self.risk_pct * risk_mult, stop_distance, info["tick_size"], info["tick_value"],
            info["vol_min"], info["vol_max"], info["vol_step"])
        if volume is None:
            return self._refuse(message, ["حداقل حجم مجاز بروکر بیشتر از ریسک تعیین‌شده است"])
        try:
            sent = self.trade.send_market_order(
                self.symbol, cfg["symbol_suffix"], plan["direction"], volume, plan["stop"], plan["target1"],
                MAGIC, "LocalChat " + self.tf, cfg["terminal_path"])
        except mt5_data.MT5Error as exc:
            return self._refuse(message, [exc.message])
        self.log.event("order", message + f" | سفارش ارسال شد: حجم {sent['volume']} قیمت {sent['price']}",
                       plan=plan, order=sent)
        return "order"

    def _refuse(self, message, reasons):
        self.log.event("refused", message + " — ولی سفارش ارسال نشد: " + "؛ ".join(reasons))
        return "refused"

    # -- main loop -------------------------------------------------------------------------
    def run(self, sleep=time.sleep, max_cycles=None):
        self.log.event("start", f"پایش {self.symbol} {self.tf} | حالت {self.mode} | هر کندل بسته یک‌بار تحلیل می‌شود.")
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            current = self.bar_id()
            if self.state["bar"] != current:
                if self.state["bar"] is not None:
                    sleep(BAR_GRACE_SECONDS)        # let the broker publish the candle that just closed
                try:
                    self.step()
                except (mt5_data.MT5Error, forex_desk.DeskError) as exc:
                    self.log.event("error", exc.message)
                cycles += 1
                self.state["bar"] = current
                self._save()
            sleep(POLL_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol")
    parser.add_argument("--tf", default="M15")
    parser.add_argument("--mode", choices=MODES, default="alert")
    parser.add_argument("--risk", type=float, default=0.5, help="percent of equity risked per trade (default 0.5)")
    parser.add_argument("--max-trades", type=int, default=3)
    parser.add_argument("--max-daily-loss", type=float, default=2.0, help="percent of equity")
    parser.add_argument("--agent", action="store_true", help="the local language model decides take/skip on confirmed scenarios")
    parser.add_argument("--model", default="", help="Ollama model for --agent (default: text_model from settings)")
    parser.add_argument("--agent-mode", choices=forex_agent.MODES, default="veto")
    parser.add_argument("--ai-scenario", action="store_true",
                        help="no edge filters: the model itself builds the scenario, the script watches the trigger, the model activates it")
    parser.add_argument("--ai-every", type=int, default=4, help="candles between the model's scenario proposals while idle")
    parser.add_argument("--i-accept-live-risk", action="store_true")
    parser.add_argument("--data-dir", default="watch_data")
    args = parser.parse_args(argv)
    if not 0 < args.risk <= 2:
        parser.error("--risk must be in (0, 2]")
    agent = None
    if args.ai_scenario:
        args.agent = True
    if args.agent:
        model = args.model or app_settings.forex_settings().get("text_model", "")
        if not model:
            parser.error("--agent needs --model (or a text_model in the forex settings)")
        agent = forex_agent.AgentDecider(forex_agent.ollama_chat, model, args.agent_mode)
    os.makedirs(args.data_dir, exist_ok=True)
    symbol, tf = args.symbol.upper(), args.tf.upper()
    watcher = Watcher(
        symbol, tf, app_settings.forex_settings(), mode=args.mode, risk_pct=args.risk, max_trades=args.max_trades,
        max_daily_loss_pct=args.max_daily_loss, stop_file=os.path.join(args.data_dir, "STOP"),
        state_path=os.path.join(args.data_dir, f"state_{symbol}_{tf}.json"),
        log=WatchLog(os.path.join(args.data_dir, "watch_log.jsonl")), agent=agent,
        ai_scenario=args.ai_scenario, ai_every=args.ai_every)
    watcher.live_ok = args.i_accept_live_risk
    print(f"کلید توقف: فایلی به نام STOP در پوشهٔ {args.data_dir} بساز. خروج: Ctrl+C")
    try:
        watcher.run()
    except KeyboardInterrupt:
        print("متوقف شد.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
