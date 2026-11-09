"""Order placement for the watcher. Every call runs on the shared MT5 worker thread.

Nothing here decides *whether* to trade; ``forex_watch`` does. This module only talks to the
terminal: account facts, own open positions, today's closed result and one market order with
mandatory stop-loss and take-profit (checked with ``order_check`` before ``order_send``).
"""
from datetime import datetime, timezone

if __package__:
    from . import mt5_data
else:  # pragma: no cover
    import mt5_data

MT5Error = mt5_data.MT5Error
RETCODE_DONE = (10008, 10009)      # placed / done
ACCOUNT_MODES = {0: "demo", 1: "contest", 2: "real"}


def _call(function, *args):
    return mt5_data._WORKER.call(function, *args)


def _account_impl(terminal_path):
    with mt5_data._LOCK:
        mt5 = mt5_data._import_mt5()
        terminal = mt5_data._connect_impl(terminal_path)
        account = mt5.account_info()
        if account is None:
            raise MT5Error("اطلاعات حساب از MT5 خوانده نشد؛ در ترمینال وارد حساب شو.", "no_account")
        return {
            "mode": ACCOUNT_MODES.get(int(getattr(account, "trade_mode", 2)), "real"),
            "login": int(getattr(account, "login", 0)),
            "equity": float(getattr(account, "equity", 0.0)),
            "balance": float(getattr(account, "balance", 0.0)),
            "currency": str(getattr(account, "currency", "")),
            "account_trade_allowed": bool(getattr(account, "trade_allowed", False)),
            "terminal_trade_allowed": bool(getattr(terminal, "trade_allowed", False)),
        }


def account_state(terminal_path=""):
    return _call(_account_impl, terminal_path)


def _positions_impl(terminal_path, symbol, suffix, magic):
    with mt5_data._LOCK:
        mt5 = mt5_data._import_mt5()
        mt5_data._connect_impl(terminal_path)
        name = mt5_data.resolve_symbol(mt5, symbol, suffix)
        rows = mt5.positions_get(symbol=name) or ()
        return [{"ticket": int(p.ticket), "volume": float(p.volume), "type": int(p.type), "profit": float(p.profit)}
                for p in rows if int(getattr(p, "magic", 0)) == magic]


def own_positions(symbol, suffix, magic, terminal_path=""):
    return _call(_positions_impl, terminal_path, symbol, suffix, magic)


def _today_impl(terminal_path, magic, now_epoch):
    with mt5_data._LOCK:
        mt5 = mt5_data._import_mt5()
        mt5_data._connect_impl(terminal_path)
        day_start = datetime.fromtimestamp(now_epoch, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, datetime.fromtimestamp(now_epoch + 86400, timezone.utc)) or ()
        mine = [d for d in deals if int(getattr(d, "magic", 0)) == magic]
        closes = [d for d in mine if int(getattr(d, "entry", 0)) == 1]      # DEAL_ENTRY_OUT
        pnl = sum(float(d.profit) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0)) for d in closes)
        opens = sum(1 for d in mine if int(getattr(d, "entry", 0)) == 0)
        return {"pnl": pnl, "trades": opens}


def today_result(magic, terminal_path="", now_epoch=None):
    import time
    return _call(_today_impl, terminal_path, magic, time.time() if now_epoch is None else now_epoch)


def _symbol_impl(terminal_path, symbol, suffix):
    with mt5_data._LOCK:
        mt5 = mt5_data._import_mt5()
        mt5_data._connect_impl(terminal_path)
        name = mt5_data.resolve_symbol(mt5, symbol, suffix)
        info = mt5.symbol_info(name)
        if info is None:
            raise MT5Error(f"اطلاعات نماد {name} خوانده نشد.", "symbol_not_found")
        return {
            "name": name, "tick_size": float(info.trade_tick_size), "tick_value": float(info.trade_tick_value_loss
                                                                                       if getattr(info, "trade_tick_value_loss", 0) else info.trade_tick_value),
            "vol_min": float(info.volume_min), "vol_max": float(info.volume_max), "vol_step": float(info.volume_step),
            "stops_level": int(getattr(info, "trade_stops_level", 0)), "point": float(info.point),
            "digits": int(info.digits),
        }


def symbol_trade_info(symbol, suffix, terminal_path=""):
    return _call(_symbol_impl, terminal_path, symbol, suffix)


def _send_impl(terminal_path, symbol, suffix, direction, volume, stop, target, magic, comment, deviation):
    with mt5_data._LOCK:
        mt5 = mt5_data._import_mt5()
        mt5_data._connect_impl(terminal_path)
        name = mt5_data.resolve_symbol(mt5, symbol, suffix)
        tick = mt5.symbol_info_tick(name)
        info = mt5.symbol_info(name)
        if tick is None or info is None:
            raise MT5Error("قیمت لحظه‌ای برای ارسال سفارش در دسترس نبود.", "no_tick")
        long = direction == "long"
        digits = int(info.digits)
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": name, "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if long else mt5.ORDER_TYPE_SELL,
            "price": float(tick.ask if long else tick.bid), "sl": round(stop, digits), "tp": round(target, digits),
            "deviation": int(deviation), "magic": int(magic), "comment": comment[:30],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        last = None
        for filling in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
            request["type_filling"] = filling
            check = mt5.order_check(request)
            if check is not None and int(check.retcode) == 0:
                result = mt5.order_send(request)
                if result is not None and int(result.retcode) in RETCODE_DONE:
                    return {"ticket": int(getattr(result, "order", 0)), "price": float(getattr(result, "price", 0.0)),
                            "volume": float(volume)}
                last = result
            else:
                last = check
        code = getattr(last, "retcode", None)
        text = getattr(last, "comment", "") if last is not None else mt5_data._last_error(mt5)
        raise MT5Error(f"بروکر سفارش را نپذیرفت (کد {code}: {text}).", "order_rejected")


def send_market_order(symbol, suffix, direction, volume, stop, target, magic, comment, terminal_path="", deviation=20):
    """Market order with SL and TP. Returns {"ticket", "price", "volume"} or raises MT5Error."""
    return _call(_send_impl, terminal_path, symbol, suffix, direction, volume, stop, target, magic, comment, deviation)
