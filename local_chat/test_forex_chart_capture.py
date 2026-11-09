"""Model-independent MT5 chart capture endpoint tests."""
import unittest
from unittest.mock import patch

from local_chat.server import ChatHandler


class _FakeHandler:
    _forex_chart_capture = ChatHandler._forex_chart_capture

    def __init__(self, payload):
        self.payload = payload
        self.response = None

    def _read_json_body(self, max_size=16_384):
        return self.payload

    def _send_json(self, status, payload, **kwargs):
        self.response = (status, payload)


class ForexChartCaptureTests(unittest.TestCase):
    @patch("local_chat.server.forex_desk.analyze_symbol")
    @patch("local_chat.server.app_settings.forex_settings", return_value={
        "enabled": True, "default_timeframe": "M15", "symbol_suffix": "", "terminal_path": ""
    })
    def test_chart_is_created_from_mt5_without_selected_llm(self, _settings, analyze):
        analyze.return_value = {
            "snapshot": {"broker_symbol": "EURUSD.a", "market": "open", "fetched_utc": "2026-11-08T00:00:00Z"},
            "chart_png": b"fake-png",
        }
        handler = _FakeHandler({"text": "از چارت EURUSD در تایم‌فریم H1 اسکرین‌شات بگیر"})
        handler._forex_chart_capture()
        self.assertEqual(handler.response[0], 200)
        chart = handler.response[1]["charts"][0]
        self.assertEqual(chart["symbol"], "EURUSD")
        self.assertEqual(chart["broker_symbol"], "EURUSD.a")
        self.assertEqual(chart["timeframe"], "H1")
        self.assertTrue(chart["image"].startswith("data:image/png;base64,"))
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.kwargs["deep"], False)
        self.assertEqual(analyze.call_args.kwargs["with_chart"], True)


    @patch("local_chat.server.forex_desk.analyze_symbol")
    @patch("local_chat.server.app_settings.forex_settings", return_value={
        "enabled": True, "default_timeframe": "M15", "symbol_suffix": "", "terminal_path": ""
    })
    def test_persian_gold_chart_request_resolves_to_xauusd(self, _settings, analyze):
        analyze.return_value = {
            "snapshot": {"broker_symbol": "XAUUSD", "market": "open", "fetched_utc": "2026-11-08T00:00:00Z"},
            "chart_png": b"fake-png",
        }
        handler = _FakeHandler({"text": "یه تصویر از چارت طلا بگیر نشونم بده"})
        handler._forex_chart_capture()
        self.assertEqual(handler.response[0], 200)
        self.assertEqual(handler.response[1]["charts"][0]["symbol"], "XAUUSD")
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[0], "XAUUSD")

    def test_rejects_request_without_a_pair(self):
        handler = _FakeHandler({"text": "از چارت عکس بگیر"})
        handler._forex_chart_capture()
        self.assertEqual(handler.response[0], 400)


if __name__ == "__main__":
    unittest.main()

class ExplicitChartIntentTests(unittest.TestCase):
    def test_direct_chart_command_with_symbol_is_routed(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [{"role": "user", "content": "اسکرینشات از xauusd بگیر نشونم بده"}]
        self.assertEqual(_explicit_forex_chart_request(messages), messages[0]["content"])

    def test_chart_followup_carries_previous_symbol(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [
            {"role": "user", "content": "یه تحلیل از XAUUSD بده"},
            {"role": "assistant", "content": "تحلیل نماد طلا"},
            {"role": "user", "content": "اسکرینشات بگیر"},
        ]
        self.assertEqual(_explicit_forex_chart_request(messages), "اسکرینشات بگیر XAUUSD")

    def test_analysis_question_is_not_chart_capture(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [{"role": "user", "content": "به نظرت XAUUSD صعودی میشه؟"}]
        self.assertIsNone(_explicit_forex_chart_request(messages))

    def test_screenshot_word_without_action_is_not_chart_capture(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [{"role": "user", "content": "اسکرینشات XAUUSD رو تحلیل کن"}]
        self.assertIsNone(_explicit_forex_chart_request(messages))


class ForexChartContextRetentionTests(unittest.TestCase):
    def test_chart_followup_carries_symbol_and_timeframe(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [
            {"role": "user", "content": "تحلیل XAUUSD در تایم‌فریم H1 بده"},
            {"role": "assistant", "content": "تحلیل انجام شد"},
            {"role": "user", "content": "اسکرین‌شات بگیر"},
        ]
        result = _explicit_forex_chart_request(messages)
        self.assertIn("XAUUSD", result)
        self.assertIn("H1", result)

    def test_context_survives_more_than_twelve_messages_and_model_switch(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [{"role": "user", "content": "تحلیل EURUSD در تایم‌فریم M30 بده"}]
        for index in range(7):
            messages.extend([
                {"role": "assistant", "content": f"پاسخ {index}"},
                {"role": "user", "content": f"سؤال عمومی شماره {index}"},
            ])
        messages.append({"role": "user", "content": "اسکرین‌شات بگیر"})
        result = _explicit_forex_chart_request(messages)
        self.assertIn("EURUSD", result)
        self.assertIn("M30", result)

    def test_explicit_new_timeframe_overrides_previous_timeframe(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [
            {"role": "user", "content": "XAUUSD در H1"},
            {"role": "user", "content": "اسکرین‌شات در M5 بگیر"},
        ]
        result = _explicit_forex_chart_request(messages)
        self.assertIn("M5", result)
        self.assertNotIn(" H1", result)


class FvgOverlayTests(unittest.TestCase):
    def test_fvg_annotation_followup_is_routed_as_chart_request(self):
        from local_chat.server import _explicit_forex_chart_request, _chart_overlays
        messages = [
            {"role": "user", "content": "یه اسکرینشات از xauusd بگیر در همین صفحه چت نمایش بده"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "من گفتم fvg رو توی تصویر با خط چین یا یه طرح دیگه مشخص کن"},
        ]
        request = _explicit_forex_chart_request(messages)
        self.assertIsNotNone(request)
        self.assertIn("XAUUSD", request.upper())
        self.assertTrue(_chart_overlays(request).get("fvg"))

    def test_most_important_fvg_draws_one_gap(self):
        from local_chat.server import _chart_overlays
        self.assertEqual(_chart_overlays("مهمترین fvg رو مشخص کن XAUUSD")["fvg"], True)
        self.assertEqual(_chart_overlays("مهمترین fvg رو مشخص کن XAUUSD")["fvg_count"], 1)

    def test_plain_question_about_fvg_is_not_a_chart_request(self):
        from local_chat.server import _explicit_forex_chart_request
        messages = [{"role": "user", "content": "FVG یعنی چی؟"}]
        self.assertIsNone(_explicit_forex_chart_request(messages))

    def test_find_fvgs_and_render(self):
        from local_chat import forex_chart
        n = 40
        o, h, l, c = [2000.0] * n, [2001.0] * n, [1999.0] * n, [2000.5] * n
        h[19], l[19] = 2000.0, 1998.0
        o[20], c[20], h[20], l[20] = 1999.5, 2012.0, 2012.3, 1999.2
        o[21], c[21], h[21], l[21] = 2012.0, 2013.0, 2014.0, 2005.0
        for i in range(22, n):
            o[i], c[i], h[i], l[i] = 2013.0, 2013.2, 2014.0, 2012.0
        rates = {"time": list(range(n)), "open": o, "high": h, "low": l, "close": c}
        gaps = forex_chart.find_fvgs(rates, atr_value=2.0, limit=3)
        self.assertEqual((gaps[0]["kind"], gaps[0]["bottom"], gaps[0]["top"]), ("bull", 2000.0, 2005.0))
        self.assertFalse(gaps[0]["filled"])
        png = forex_chart.render(rates, digits=2, fvgs=gaps)
        self.assertTrue(png.startswith(b"\x89PNG"))
