"""Tool-calling architecture: the main model decides chart vs analysis vs chat in ONE request.

Ollama is replaced by a scripted fake; MT5 is replaced by fakes. These tests prove the app's
routing/streaming logic, NOT real MT5 behaviour (that needs a Windows machine with a terminal).
"""
import io
import json
import unittest
from unittest.mock import patch

from local_chat import server as sv
from local_chat import mt5_data


def _ndjson(*packets):
    return [json.dumps(p, ensure_ascii=False).encode("utf-8") + b"\n" for p in packets]


class _FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)


class _Handler:
    _chat_with_model = sv.ChatHandler._chat_with_model
    _forex_tool_turn = sv.ChatHandler._forex_tool_turn
    _emit = sv.ChatHandler._emit
    _validate_chat = staticmethod(sv.ChatHandler._validate_chat)

    def __init__(self):
        self.wfile = io.BytesIO()
        self._session_id = "s"
        self.errors = []

    def _begin_ndjson(self):
        pass

    def _heartbeat(self):
        pass

    def _send_json(self, status, payload, **kw):
        self.errors.append((status, payload))

    def packets(self):
        return [json.loads(x) for x in self.wfile.getvalue().splitlines() if x.strip()]


CHART_CALL = _ndjson(
    {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "render_forex_chart",
        "arguments": {"symbol": "XAUUSD", "timeframe": "M15", "overlays": ["fvg"], "fvg_scope": "most_important"}}}]},
     "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"})
ANALYSIS_CALL = _ndjson(
    {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "get_market_analysis", "arguments": {"symbols": ["XAUUSD"], "timeframe": "H1"}}}]}, "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": True})
TEXT_ANSWER = _ndjson(
    {"message": {"role": "assistant", "content": "پاسخ متنی"}, "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"})

FOREX_CFG = {"enabled": True, "default_timeframe": "M15", "symbol_suffix": "", "terminal_path": "",
             "news_enabled": False}


class _Store:
    def prompt_context(self):
        return ""


def _payload(text, history=()):
    return {"model": "toolmodel", "mode": "general", "forex": True,
            "messages": [*history, {"role": "user", "content": text}]}


def _report(png=b"png"):
    return {"snapshot": {"broker_symbol": "XAUUSD", "market": "open", "fetched_utc": "t"},
            "chart_png": png, "fvgs": [{"kind": "bull", "top": 2005.0, "bottom": 2000.0, "filled": False,
                                        "touched": False, "time": 1}]}


class ToolRoutingTests(unittest.TestCase):
    def run_chat(self, text, caps, scripted, analyze=None, history=()):
        calls = {"payloads": []}
        scripted = list(scripted)

        def fake_open(payload):
            calls["payloads"].append(payload)
            return _FakeStream(scripted.pop(0))

        handler = _Handler()
        ctx = sv.forex_service.ForexResult("DIGEST", ["n1"], True)
        with patch.object(sv, "_model_capabilities", return_value=caps), \
                patch.object(sv, "_open_chat_stream", side_effect=fake_open), \
                patch.object(sv, "_memory_store", return_value=_Store()), \
                patch.object(sv.app_settings, "forex_settings", return_value=FOREX_CFG), \
                patch.object(sv.forex_desk, "analyze_symbol", side_effect=analyze or (lambda *a, **k: _report())) as an, \
                patch.object(sv.forex_service, "build_context", return_value=ctx) as bc:
            handler._chat_with_model(_payload(text, history))
        return handler, calls, an, bc

    def test_clear_chart_command_needs_no_model_call_and_no_prefetch(self):
        # A clear «take a screenshot of X and mark …» command is routed by the app itself (zero model calls, no
        # tool schema in the prompt, no chance of a text-only answer). Ambiguous wording still goes to the model
        # (see test_ambiguous_chart_wording_still_reaches_the_model).
        h, calls, analyze, build = self.run_chat("اسکرین‌شات xauusd بگیر و مهمترین fvg رو مشخص کن",
                                                 ["completion", "tools"], [CHART_CALL])
        self.assertEqual(len(calls["payloads"]), 0)
        build.assert_not_called()            # no analysis fetch just to draw a picture
        analyze.assert_called_once()
        kw = analyze.call_args.kwargs
        self.assertEqual(kw["overlays"], {"fvg": True, "fvg_count": 1})
        self.assertFalse(kw["deep"])
        out = h.packets()
        self.assertEqual(sum(1 for p in out if p.get("chartImages")), 1)
        self.assertEqual(sum(1 for p in out if p.get("done")), 1)
        self.assertFalse(any((p.get("message") or {}).get("content") for p in out))
        notice = next(p["notice"] for p in out if p.get("chartImages"))
        self.assertIn("FVG", notice)
        self.assertEqual(sv._METRICS[-1]["model_calls"], 0)
        self.assertEqual(sv._METRICS[-1]["route"], "direct-chart")

    def test_ambiguous_chart_wording_still_reaches_the_model(self):
        h, calls, analyze, build = self.run_chat("میشه یه نگاه به شکل بازار طلا بندازم؟", ["completion", "tools"],
                                                 [CHART_CALL])
        self.assertEqual(len(calls["payloads"]), 1)
        self.assertIn("tools", calls["payloads"][0])
        self.assertEqual(sv._METRICS[-1]["model_calls"], 1)

    def test_analysis_request_uses_tool_then_one_answer_call(self):
        h, calls, analyze, build = self.run_chat("تحلیل طلا چیه؟", ["completion", "tools"],
                                                 [ANALYSIS_CALL, TEXT_ANSWER])
        self.assertEqual(len(calls["payloads"]), 2)
        self.assertNotIn("tools", calls["payloads"][1])      # the answer call cannot loop
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["symbols"], ["XAUUSD"])
        self.assertEqual(build.call_args.kwargs["timeframe"], "H1")
        analyze.assert_not_called()                          # no chart for an analysis question
        out = h.packets()
        self.assertTrue(any((p.get("message") or {}).get("content") == "پاسخ متنی" for p in out))
        self.assertEqual(sv._METRICS[-1]["model_calls"], 2)

    def test_plain_chat_touches_no_tool_and_no_mt5(self):
        h, calls, analyze, build = self.run_chat("FVG یعنی چی؟", ["completion", "tools"], [TEXT_ANSWER])
        self.assertEqual(len(calls["payloads"]), 1)
        build.assert_not_called()
        analyze.assert_not_called()
        self.assertTrue(any((p.get("message") or {}).get("content") == "پاسخ متنی" for p in h.packets()))

    def test_tool_failure_is_reported_once_with_stage_and_no_model_followup(self):
        cases = [("not_installed", "package"), ("binary_load_failed", "dll"),
                 ("init_failed", "terminal"), ("not_connected", "connection")]
        for code, stage in cases:
            def boom(*a, **k):
                raise mt5_data.MT5Error("پیام " + code, code)
            h, calls, _a, _b = self.run_chat("چارت xauusd", ["completion", "tools"], [CHART_CALL], analyze=boom)
            self.assertEqual(len(calls["payloads"]), 1, code)          # model never asked to comment
            out = h.packets()
            errors = [p for p in out if p.get("error")]
            self.assertEqual(len(errors), 1, code)
            self.assertEqual(errors[0]["errorStage"], stage)
            self.assertIn("پیام " + code, errors[0]["error"])
            self.assertEqual(sum(1 for p in out if p.get("done")), 1, code)

    def test_model_without_tool_calling_uses_rule_fallback_with_zero_model_calls(self):
        h, calls, analyze, build = self.run_chat("اسکرین‌شات xauusd بگیر و نشون بده", ["completion"], [])
        self.assertEqual(calls["payloads"], [])
        analyze.assert_called_once()
        self.assertTrue(any(p.get("chartImages") for p in h.packets()))

    def test_model_without_tool_calling_still_gets_analysis_context(self):
        h, calls, analyze, build = self.run_chat("تحلیل XAUUSD بده", ["completion"], [TEXT_ANSWER])
        build.assert_called_once()
        self.assertNotIn("tools", calls["payloads"][0])
        analyze.assert_not_called()

    def test_intent_endpoint_and_extra_model_call_are_gone(self):
        self.assertFalse(hasattr(sv.ChatHandler, "_forex_intent_classify"))
        import inspect
        self.assertNotIn("/api/forex/intent", inspect.getsource(sv))

    def test_mt5_stage_mapping(self):
        self.assertEqual(mt5_data.error_stage("not_installed"), "package")
        self.assertEqual(mt5_data.error_stage("binary_load_failed"), "dll")
        self.assertEqual(mt5_data.error_stage("init_failed"), "terminal")
        self.assertEqual(mt5_data.error_stage("not_connected"), "connection")


if __name__ == "__main__":
    unittest.main()
