"""Regression tests for inline Forex mode in the normal chat composer."""
from local_chat import server, settings, forex_service


def test_forex_chat_flag_is_optional_and_boolean():
    base = {"model": "qwen3:latest", "messages": [{"role": "user", "content": "سلام"}]}
    assert server.ChatHandler._validate_chat(base) is None
    assert server.ChatHandler._validate_chat({**base, "forex": True}) is None
    assert server.ChatHandler._validate_chat({**base, "forex": "yes"}) is not None


def test_forex_prompt_is_only_added_when_enabled():
    assert "حالت تحلیلگر فارکس فعال است" not in server._system_prompt()
    assert "حالت تحلیلگر فارکس فعال است" in server._system_prompt(forex=True)


def test_forex_settings_defaults_and_validation():
    cfg = settings.forex_settings()
    assert cfg["enabled"] is True
    assert cfg["default_timeframe"] in settings.FOREX_TIMEFRAMES
    try:
        settings.set_forex_settings({"default_timeframe": "BAD"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timeframe should be rejected")


def test_forex_context_degrades_safely_without_mt5(monkeypatch):
    monkeypatch.setattr(forex_service.app_settings, "forex_settings",
                        lambda: {"enabled": True, "default_timeframe": "M15", "symbol_suffix": ""})
    monkeypatch.setattr(forex_service.mt5_data, "fetch_history",
                        lambda *a, **k: (_ for _ in ()).throw(forex_service.mt5_data.MT5Error("MT5 unavailable", "not_installed")))
    result = forex_service.build_context([{"role": "user", "content": "تحلیل EURUSD در M15"}], 3000)
    assert result.ok is False
    assert "هیچ قیمت" in result.context or "قیمت" in result.context
