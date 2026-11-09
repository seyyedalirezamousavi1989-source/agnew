"""Regression tests for the 2.0.4 fixes (from a real conversation with the agent):

1. «یه اسکرینشات از xauusd بگیر و توش محتمل‌ترین سناریو رو با فلش نشون بده» was answered with the canned
   «بله، برنامه قابلیت … را دارد» (capability shortcut) instead of a chart.
2. After a chart the history held only the status notice; the model copied it back as text without any picture
   («پس تصویرش کو؟»).
3. «سقف‌ها و کف‌ها رو مشخص کن» / «ادیتش کن» only produced a plain chart (no swing overlay, previous overlays lost).
4. Context: nothing may be forgotten silently, and the prompt must stay lean.

Ollama and MT5 are replaced by fakes; these tests prove the app's routing/context logic, not real MT5 behaviour.
"""
import json
import random
import unittest
from unittest.mock import patch

from local_chat import context_manager as cm
from local_chat import forex_chart, forex_desk
from local_chat import server as sv
from local_chat.test_forex_tool_routing import (FOREX_CFG, TEXT_ANSWER, _Handler, _FakeStream, _Store, _payload,
                                                _report)

XAU = "یه اسکرینشات از xauusd بگیر و توش محتملترین سناریو رو با فلش نشون بده"


def chart_message(**overrides):
    state = {"symbol": "XAUUSD", "timeframe": "M15", "theme": "dark", "overlays": [], "market": "closed"}
    state.update(overrides)
    return {"role": "assistant", "content": "نمودار مستقیماً از داده‌های کندلی MT5 ساخته شد. بازار بسته است.",
            "chart": state}


def user(text):
    return {"role": "user", "content": text}


class CapabilityShortcutTests(unittest.TestCase):
    def test_tasks_are_never_answered_with_the_canned_capability_text(self):
        for text in (
            XAU,
            "یه اسکرینشات از xauusd بگیر و با فلش محتملترین سناریو تحلیلیتو نشون بده",
            "با فلش محتملترین سناریو تحلیلیتو نشون بده",
            "سقفها و کفهارو مشخص کن توی تصویر",
            "پس تصویرش کو؟",
            "چارت eurusd رو بکش و سناریو رو روش نشون بده",
        ):
            with self.subTest(text=text):
                self.assertIsNone(sv._capability_question_answer(text))

    def test_real_capability_questions_are_still_answered(self):
        for text in ("تحلیل فارکس بلدی؟", "می‌تونی تحلیل فارکس کنی؟", "فارکس تحلیل می‌تونی بکنی؟", "قابلیت تحلیل فارکس داری؟"):
            with self.subTest(text=text):
                answer = sv._capability_question_answer(text)
                self.assertIsNotNone(answer)
                self.assertIn("گفت‌وگوی تحلیل فارکس", answer)

    def test_forex_mode_message_naming_a_market_skips_the_shortcut(self):
        class Fake:
            replied = None

            def _reply_text(self, model, reply):
                self.replied = reply

        fake = Fake()
        payload = {"model": "m", "forex": True, "messages": [user("xauusd رو تحلیل می‌تونی بکنی؟")]}
        self.assertFalse(sv.ChatHandler._local_shortcut(fake, payload))
        self.assertIsNone(fake.replied)


class ChartRoutingTests(unittest.TestCase):
    def test_first_request_from_the_transcript_is_a_chart_with_a_scenario_arrow(self):
        routed = sv._route_chart_request([user(XAU)], native_tools=True)
        self.assertEqual(routed[0], "text")
        self.assertTrue(sv._chart_overlays(routed[1]).get("scenario"))

    def test_edit_keeps_symbol_timeframe_theme_and_adds_swings(self):
        history = [user("اسکرین‌شات xauusd بگیر"), chart_message(), user("سقفها و کفهارو مشخص کن توی تصویر")]
        kind, args = sv._route_chart_request(history, native_tools=True)
        self.assertEqual(kind, "args")
        self.assertEqual((args["symbol"], args["timeframe"], args["theme"]), ("XAUUSD", "M15", "dark"))
        self.assertEqual(args["overlays"], ["swings"])
        self.assertTrue(sv._chart_overlays_from_args(args).get("swings"))

    def test_edit_accumulates_overlays_and_can_remove_them(self):
        history = [user("اسکرین‌شات xauusd بگیر"), chart_message(overlays=["swings"]),
                   user("با فلش محتملترین سناریو رو هم نشون بده")]
        args = sv._chart_edit_args(history)
        self.assertEqual(args["overlays"], ["scenario", "swings"])
        history = [user("x"), chart_message(overlays=["scenario", "swings"]), user("فلش سناریو رو بردار")]
        self.assertEqual(sv._chart_edit_args(history)["overlays"], ["swings"])
        history = [user("x"), chart_message(overlays=["scenario", "swings"]), user("فقط سقف و کف رو نشون بده")]
        self.assertEqual(sv._chart_edit_args(history)["overlays"], ["swings"])

    def test_redo_and_theme_change_reuse_the_previous_chart(self):
        base = [user("x"), chart_message(overlays=["scenario"])]
        args = sv._chart_edit_args(base + [user("پس تصویرش کو؟")])
        self.assertEqual((args["symbol"], args["overlays"], args["theme"]), ("XAUUSD", ["scenario"], "dark"))
        args = sv._chart_edit_args(base + [user("سفیدش کن")])
        self.assertEqual((args["theme"], args["overlays"]), ("light", ["scenario"]))

    def test_questions_analysis_other_markets_and_stale_charts_are_not_edits(self):
        base = [user("x"), chart_message()]
        for text in ("سقف و کف یعنی چی؟", "تحلیل کن و سقف و کف رو مشخص کن", "الان روند چطوره؟"):
            with self.subTest(text=text):
                self.assertIsNone(sv._chart_edit_args(base + [user(text)]))
        # another market → a fresh request handled by the explicit path
        self.assertIsNone(sv._chart_edit_args(base + [user("یه اسکرینشات از eurusd بگیر")]))
        kind, text = sv._route_chart_request(base + [user("یه اسکرینشات از eurusd بگیر")], native_tools=True)
        self.assertEqual(kind, "text")
        # three turns later the user is no longer editing that picture
        stale = base + [user("بازار چطوره؟"), {"role": "assistant", "content": "خوبه"}, user("سقف و کف رو نشون بده")]
        self.assertIsNone(sv._chart_edit_args(stale))

    def test_tool_models_hand_chart_plus_analysis_requests_to_the_model(self):
        text = "اسکرین‌شات xauusd بگیر و بگو چرا افت کرد"
        self.assertIsNone(sv._route_chart_request([user(text)], native_tools=True))
        self.assertEqual(sv._route_chart_request([user(text)], native_tools=False)[0], "text")

    def test_swing_word_is_not_matched_inside_other_words(self):
        self.assertIsNone(sv._SWING_WORD.search("کفش و سقفی"))
        self.assertTrue(sv._SWING_WORD.search("سقف ها و کف ها"))
        self.assertTrue(sv._SWING_WORD.search("سقف‌ها و کف‌ها".replace("‌", " ")))


class SwingTests(unittest.TestCase):
    @staticmethod
    def _zigzag():
        random.seed(3)
        highs, lows, close = [], [], []
        base = [2650 + (6 if (i // 12) % 2 == 0 else -6) * ((i % 12) / 12 if (i // 12) % 2 == 0 else 1 - (i % 12) / 12)
                for i in range(120)]
        for value in base:
            close.append(value)
            highs.append(value + random.uniform(0.2, 0.8))
            lows.append(value - random.uniform(0.2, 0.8))
        return {"time": list(range(120)), "open": close[:], "high": highs, "low": lows, "close": close}

    def test_find_swings_returns_ordered_confirmed_extremes(self):
        rates = self._zigzag()
        swings = forex_chart.find_swings(rates, atr_value=1.0)
        self.assertTrue(swings["highs"] and swings["lows"])
        self.assertLessEqual(len(swings["highs"]), 3)
        for side in swings.values():
            self.assertEqual([i["index"] for i in side], sorted(i["index"] for i in side))
            self.assertTrue(all(i["index"] <= 119 - 3 for i in side))   # the newest candles cannot be confirmed
        for item in swings["highs"]:
            self.assertEqual(item["price"], rates["high"][item["index"]])
            self.assertGreater(item["price"], max(rates["high"][item["index"] - 3:item["index"]]))
        for item in swings["lows"]:
            self.assertEqual(item["price"], rates["low"][item["index"]])

    def test_annotations_render_and_report_what_was_drawn(self):
        rates = self._zigzag()
        report = {}
        notes = forex_desk.swing_annotations(report, rates, 1.0, 2)
        self.assertTrue(notes)
        self.assertTrue(all(n["type"] == "marker" and n["label"][0] in "HL" and n["bar"] <= 0 for n in notes))
        self.assertEqual(len(forex_chart.sanitize_annotations(notes)), len(notes))
        self.assertTrue(report["swings_drawn"]["highs"] and report["swings_drawn"]["lows"])
        plain = forex_chart.render(rates, digits=2)
        marked = forex_chart.render(rates, digits=2, annotations=notes)
        self.assertNotEqual(plain, marked)

    def test_flat_market_reports_nothing_found(self):
        flat = {"time": list(range(60)), "open": [1.0] * 60, "high": [1.001] * 60, "low": [0.999] * 60, "close": [1.0] * 60}
        report = {}
        self.assertEqual(forex_desk.swing_annotations(report, flat, 0.01, 5), [])
        self.assertIn("swing_note", report)


class ChatFlowTests(unittest.TestCase):
    def run_chat(self, text, history, caps=("completion", "tools"), scripted=None, report=None):
        calls = {"payloads": []}
        scripted = list(scripted or [])

        def fake_open(payload):
            calls["payloads"].append(payload)
            return _FakeStream(scripted.pop(0))

        handler = _Handler()
        ctx = sv.forex_service.ForexResult("DIGEST", [], True)
        analyze_calls = []

        def analyze(*args, **kwargs):
            analyze_calls.append((args, kwargs))
            return report or _report()

        with patch.object(sv, "_model_capabilities", return_value=list(caps)), \
                patch.object(sv, "_open_chat_stream", side_effect=fake_open), \
                patch.object(sv, "_memory_store", return_value=_Store()), \
                patch.object(sv.app_settings, "forex_settings", return_value=FOREX_CFG), \
                patch.object(sv.forex_desk, "analyze_symbol", side_effect=analyze), \
                patch.object(sv.forex_service, "build_context", return_value=ctx):
            handler._chat_with_model(_payload(text, history))
        return handler, calls, analyze_calls

    def test_edit_request_draws_swings_without_the_model_and_returns_state(self):
        report = _report()
        report["swings_drawn"] = {"highs": [2655.1, 2658.4], "lows": [2641.0]}
        history = [user("یه اسکرینشات از xauusd بگیر"), chart_message()]
        handler, calls, analyze_calls = self.run_chat("سقفها و کفهارو مشخص کن توی تصویر", history, report=report)
        self.assertEqual(calls["payloads"], [])
        (_, kwargs), = analyze_calls
        self.assertTrue(kwargs["overlays"].get("swings"))
        out = handler.packets()
        packet = next(p for p in out if p.get("chartImages"))
        self.assertIn("سقف‌ها 2655.1، 2658.4", packet["notice"])
        state = packet["chartImages"][0]["state"]
        self.assertEqual(state["overlays"], ["swings"])
        self.assertEqual(state["swings"]["lows"], [2641.0])
        self.assertEqual(sv._METRICS[-1]["model_calls"], 0)

    def test_history_gives_tool_models_a_tool_call_pair_not_the_status_notice(self):
        state = {"symbol": "XAUUSD", "timeframe": "M15", "theme": "dark", "overlays": ["scenario"],
                 "scenario": {"direction": "long", "prob": 0.58, "entry": 2650.0, "stop": 2640.0, "target1": 2665.0}}
        history = [user("اسکرین‌شات xauusd با فلش سناریو"), chart_message(**state),
                   user("خب قبل از ورود چی رو چک کنم؟")]
        history, question = history[:-1], history[-1]["content"]
        handler, calls, _ = self.run_chat(question, history, scripted=[TEXT_ANSWER])
        sent = calls["payloads"][0]["messages"]
        self.assertIn("tools", calls["payloads"][0])
        roles = [m["role"] for m in sent]
        self.assertEqual(roles[:5], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(sent[2]["tool_calls"][0]["function"]["name"], "render_forex_chart")
        self.assertIn("2640", sent[3]["content"])                        # the exact SL the app drew is remembered
        self.assertNotIn("نمودار مستقیماً", json.dumps(sent, ensure_ascii=False))
        self.assertFalse(any("chart" in m for m in sent))               # browser memory aid never reaches Ollama

    def test_models_without_tools_get_the_factual_summary_as_plain_text(self):
        history = [user("اسکرین‌شات xauusd"), chart_message(overlays=["swings"], swings={"highs": [2655.0], "lows": [2641.0]})]
        handler, calls, _ = self.run_chat("خب حالا توضیح بده", history, caps=("completion",), scripted=[TEXT_ANSWER])
        sent = calls["payloads"][0]["messages"]
        self.assertNotIn("tools", calls["payloads"][0])
        assistant = next(m for m in sent if m["role"] == "assistant")
        self.assertIn("سقف‌ها 2655", assistant["content"])
        self.assertNotIn("نمودار مستقیماً", assistant["content"])

    def test_old_saved_chats_without_chart_state_never_feed_the_notice_back(self):
        history = [user("اسکرین‌شات xauusd"),
                   {"role": "assistant", "content": "نمودار مستقیماً از داده‌های کندلی MT5 ساخته شد. بازار بسته است؛ آخرین ..."}]
        handler, calls, _ = self.run_chat("سقف و کف یعنی چی؟", history, scripted=[TEXT_ANSWER])
        sent = json.dumps(calls["payloads"][0]["messages"], ensure_ascii=False)
        self.assertNotIn("نمودار مستقیماً", sent)

    def test_routing_prompt_is_lean_and_analysis_call_gets_the_full_rules(self):
        lean = sv._system_prompt(forex=True, native_tools=True)
        full = sv._system_prompt(forex=True, native_tools=False)
        self.assertNotIn("حالت تحلیلگر فارکس فعال است", lean)
        self.assertIn("حالت تحلیلگر فارکس فعال است", full)
        self.assertIn("render_forex_chart", lean)
        self.assertLess(len(lean), len(full))

    def test_chat_validation_accepts_chart_state_only_on_assistant_messages(self):
        validate = sv.ChatHandler._validate_chat
        ok = {"model": "m", "messages": [user("a"), chart_message(), user("b")]}
        self.assertIsNone(validate(ok))
        bad_role = {"model": "m", "messages": [{"role": "user", "content": "a", "chart": {"symbol": "X"}}]}
        self.assertIsNotNone(validate(bad_role))
        huge = chart_message()
        huge["chart"]["scenario"] = {"direction": "long", "x": "y" * 5000}
        self.assertIsNotNone(validate({"model": "m", "messages": [huge]}))


class ChartMemoryTests(unittest.TestCase):
    def test_sanitize_rejects_garbage_and_clamps_lists(self):
        self.assertIsNone(cm.sanitize_chart_state({"symbol": "../x", "timeframe": "M15"}))
        self.assertIsNone(cm.sanitize_chart_state({"symbol": "XAUUSD", "timeframe": "M7"}))
        state = cm.sanitize_chart_state({"symbol": "xauusd", "timeframe": "m15", "overlays": ["swings", "evil"],
                                         "swings": {"highs": [1.0, "x", float("nan")] + [2.0] * 20, "lows": []}})
        self.assertEqual(state["symbol"], "XAUUSD")
        self.assertEqual(state["overlays"], ["swings"])
        self.assertLessEqual(len(state["swings"]["highs"]), 6)
        self.assertNotIn("evil", json.dumps(state))

    def test_summary_names_symbol_timeframe_and_every_drawn_level(self):
        state = cm.sanitize_chart_state({
            "symbol": "XAUUSD", "timeframe": "M15", "theme": "light", "overlays": ["scenario", "swings", "fvg"],
            "scenario": {"direction": "short", "prob": 0.61, "entry": 2650.5, "stop": 2660.0, "target1": 2640.25},
            "swings": {"highs": [2655.0], "lows": [2641.5]},
            "fvgs": [{"kind": "bull", "top": 2646.0, "bottom": 2644.0, "filled": False}], "market": "closed"})
        text = cm.chart_summary(state)
        for needle in ("XAUUSD", "M15", "سفید", "نزولی", "61", "2660", "2640.25", "2655", "2641.5", "2644", "بازار بسته"):
            self.assertIn(needle, text)
        self.assertLessEqual(len(text), 520)

    def test_last_chart_state_only_while_the_picture_is_current(self):
        msgs = [user("a"), chart_message()]
        self.assertIsNotNone(cm.last_chart_state(msgs))
        self.assertIsNone(cm.last_chart_state(msgs + [user("b"), {"role": "assistant", "content": "x"}, user("c")]))


class ContextFittingTests(unittest.TestCase):
    def test_dropped_and_older_turns_survive_as_a_digest_within_budget(self):
        older = [user("کد مخفی پروژه ALPHA-7731 است"), {"role": "assistant", "content": "یادداشت شد: ALPHA-7731"}]
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"پیام {i} " + "ی" * 1500}
                   for i in range(14)]
        history.append(user("آخرین سؤال"))
        fitted = sv._fit_messages("SYS", history, 9000, older=older)
        self.assertLessEqual(sum(len(m["content"]) for m in fitted), 9000)
        self.assertEqual(fitted[-1]["content"], "آخرین سؤال")
        system = fitted[0]["content"]
        self.assertTrue(system.startswith("SYS"))
        self.assertIn("ALPHA-7731", system)                      # the oldest facts are not gone
        self.assertIn("پیام 0", system)                          # nor the turns dropped by this very call

    def test_no_digest_when_everything_fits(self):
        fitted = sv._fit_messages("SYS", [user("سلام"), {"role": "assistant", "content": "سلام!"}, user("چطوری؟")], 9000)
        self.assertEqual(fitted[0]["content"], "SYS")

    def test_tool_result_is_never_left_without_its_tool_call(self):
        pair = cm.chart_tool_messages(cm.sanitize_chart_state({"symbol": "XAUUSD", "timeframe": "M15"}))
        history = pair + [{"role": "assistant" if i % 2 else "user", "content": "z" * 2500} for i in range(8)]
        history.append(user("آخر"))
        fitted = sv._fit_messages("SYS", history, 8000)
        roles = [m["role"] for m in fitted]
        self.assertNotIn("tool", roles)
        self.assertFalse(any(m.get("tool_calls") for m in fitted))

    def test_tool_schema_is_part_of_the_budget_and_is_compact(self):
        size = len(json.dumps(sv._FOREX_TOOLS, ensure_ascii=False))
        self.assertLess(size, 3000)                              # was 3660 before 2.0.4
        self.assertEqual(sv._TOOLS_BUDGET_CHARS, int(size * 0.6))

    def test_digest_respects_its_cap_and_keeps_newest_lines(self):
        messages = [user(f"سؤال شمارهٔ {i} " + "و" * 200) for i in range(60)]
        digest = cm.history_digest(messages, 600)
        self.assertLessEqual(len(digest), 600)
        self.assertIn("سؤال شمارهٔ 59", digest)
        self.assertNotIn("سؤال شمارهٔ 0 ", digest)
        self.assertEqual(cm.history_digest(messages, 100), "")


if __name__ == "__main__":
    unittest.main()
