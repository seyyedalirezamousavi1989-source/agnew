import math
import random
import struct
import unittest
import zlib

from local_chat import forex_chart, server


def _rates(n=120):
    random.seed(7)
    price, o, h, l, c = 2650.0, [], [], [], []
    for i in range(n):
        close = price + random.uniform(-3, 3) + 0.2 * math.sin(i / 9)
        o.append(price); c.append(close)
        h.append(max(price, close) + random.uniform(0, 2)); l.append(min(price, close) - random.uniform(0, 2))
        price = close
    return {"time": list(range(n)), "open": o, "high": h, "low": l, "close": c}


def _pixels(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(png):
        size, kind = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + size]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            idat += data
        pos += 12 + size
    raw, stride = zlib.decompress(idat), width * 3 + 1
    return width, height, [raw[y * stride + 1:(y + 1) * stride] for y in range(height)]


class ChartThemeAndAnnotationTests(unittest.TestCase):
    def test_light_theme_has_white_background_and_dark_default_unchanged(self):
        rates = _rates()
        _, _, light = _pixels(forex_chart.render(rates, theme="light", digits=2))
        _, _, dark = _pixels(forex_chart.render(rates, digits=2))
        self.assertEqual(tuple(light[2][3:6]), (255, 255, 255))
        self.assertEqual(tuple(dark[2][3:6]), forex_chart.BG)

    def test_annotations_change_the_picture_and_bad_items_are_dropped(self):
        rates = _rates()
        last = rates["close"][-1]
        notes = [{"type": "arrow", "price": last, "bar": 0, "to_price": last + 20, "to_bar": 12, "label": "up فارسی"},
                 {"type": "marker", "price": last - 5, "bar": -2, "shape": "up", "label": "buy"},
                 {"type": "hline", "price": last - 12, "label": "sl"},
                 {"type": "arrow", "price": last}, {"type": "nope", "price": 1}, "x", {"type": "text", "price": 1}]
        clean = forex_chart.sanitize_annotations(notes)
        self.assertEqual([n["type"] for n in clean], ["arrow", "marker", "hline"])
        self.assertEqual(clean[0]["label"], "UP")   # Persian letters are not drawable and are removed
        plain = forex_chart.render(rates, digits=2)
        drawn = forex_chart.render(rates, digits=2, annotations=notes)
        self.assertNotEqual(plain, drawn)

    def test_theme_words_in_text(self):
        self.assertEqual(server._theme_from_text("یه اسکرینشات از xauusd با زمینه سفید بگیر"), "light")
        self.assertEqual(server._theme_from_text("چارت رو با بک‌گراند مشکی بده"), "dark")
        self.assertIsNone(server._theme_from_text("چارت xauusd بگیر"))
        self.assertEqual(server._chart_overlays("white chart of XAUUSD").get("theme"), "light")

    def test_tool_arguments_become_overlays(self):
        args = {"symbol": "XAUUSD", "theme": "light",
                "annotations": [{"type": "arrow", "price": 2650, "to_price": 2670, "to_bar": 10}]}
        overlays = server._chart_overlays_from_args(args)
        self.assertEqual(overlays["theme"], "light")
        self.assertEqual(len(overlays["annotations"]), 1)
        props = server._FOREX_TOOLS[0]["function"]["parameters"]["properties"]
        self.assertIn("theme", props)
        self.assertIn("annotations", props)

    def test_forex_profile_gives_longer_answers_to_every_model_size(self):
        profile = server._forex_profile({"num_ctx": 8192, "num_predict": 2560, "budget": 11000})
        self.assertGreaterEqual(profile["num_predict"], 4096)
        self.assertGreaterEqual(profile["num_ctx"], 12288)
        large = server._forex_profile({"num_ctx": 4096, "num_predict": 2048, "budget": 4000})
        self.assertEqual((large["num_ctx"], large["num_predict"]), (8192, 3072))
        self.assertGreater(large["budget"], 4000)

    def test_scenario_arrow_is_built_by_the_app_from_its_own_analysis(self):
        from local_chat import forex_desk
        report = {"lean": "balanced",
                  "quant": {"probs": {"up": 0.25, "down": 0.31, "range": 0.44}, "horizon_bars": 16},
                  "scenarios": [
                      {"direction": "long", "entry": 4070.5, "stop": 4060.0, "target1": 4085.0, "target2": 4095.0},
                      {"direction": "short", "entry": 4070.5, "stop": 4080.0, "target1": 4058.0, "target2": 4050.0}]}
        notes = forex_desk.scenario_annotations(report, 2)
        arrow = next(n for n in notes if n["type"] == "arrow")
        self.assertEqual(arrow["to_price"], 4058.0)          # the more probable side (down) is drawn
        self.assertEqual(report["scenario_drawn"]["direction"], "short")
        self.assertTrue(forex_chart.sanitize_annotations(notes))
        empty = {}
        self.assertEqual(forex_desk.scenario_annotations(empty, 2), [])
        self.assertTrue(empty["scenario_note"])

    def test_screenshot_with_arrow_request_keeps_white_theme_and_scenario(self):
        text = "فقط یه دونه اسکرینشات میخام که محتملترین سناریوی تحلیلیتو با فلش توش مشخص کنی با زمینه چارت سفید"
        messages = [{"role": "user", "content": "سه اسکرینشات از xauusd بگیر با زمینه سفید"},
                    {"role": "assistant", "content": "تایم‌فریم؟"},
                    {"role": "user", "content": text}]
        request = server._explicit_forex_chart_request(messages)
        self.assertIsNotNone(request)
        overlays = server._chart_overlays(request)
        self.assertEqual(overlays.get("theme"), "light")
        self.assertTrue(overlays.get("scenario"))
        args = {"symbol": "XAUUSD", "overlays": ["scenario"], "theme": "light"}
        self.assertTrue(server._chart_overlays_from_args(args).get("scenario"))


class ContinueKeepsTheConversationTests(unittest.TestCase):
    def _chat(self):
        question = "سناریوی محتمل XAUUSD رو توضیح بده " + "الف" * 200
        half_answer = "شروع پاسخ. " + ("متن طولانی تحلیل " * 600) + " و آخرین جملهٔ ناتمام"
        return [{"role": "user", "content": "پیام خیلی قدیمی " * 50},
                {"role": "assistant", "content": "پاسخ قدیمی " * 50},
                {"role": "user", "content": question},
                {"role": "assistant", "content": half_answer},
                {"role": "user", "content": "ادامه بده"}], question, half_answer

    def test_old_behaviour_forgot_everything_when_the_half_answer_did_not_fit(self):
        messages, _, _ = self._chat()
        fitted = server._fit_messages("S" * 700, messages, 6000)
        self.assertEqual([m["content"] for m in fitted[1:]], ["ادامه بده"])

    def test_protected_tail_keeps_question_and_end_of_the_half_answer(self):
        messages, question, half_answer = self._chat()
        fitted = server._fit_messages("S" * 700, messages, 6000, protect_tail=3)
        contents = [m["content"] for m in fitted[1:]]
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0], question)
        self.assertTrue(contents[1].startswith("شروع پاسخ."))
        self.assertTrue(contents[1].endswith("و آخرین جملهٔ ناتمام"))
        self.assertEqual(contents[2], "ادامه بده")
        self.assertLessEqual(sum(len(m["content"]) for m in fitted), 6000 + 200)

    def test_continue_prompt_is_compact_and_flag_is_validated(self):
        self.assertLess(len(server.CONTINUE_SYSTEM_PROMPT), 900)
        base = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "سلام"}]}
        self.assertIsNone(server.ChatHandler._validate_chat({**base, "continue": True}))
        self.assertIsNotNone(server.ChatHandler._validate_chat({**base, "continue": "yes"}))


if __name__ == "__main__":
    unittest.main()
