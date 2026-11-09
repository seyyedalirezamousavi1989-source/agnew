"""Context accounting: real token estimates, per-model capacity, 75 % compression, pinned instructions,
checkpoint recovery and the «never send an incomplete request» guard."""
import unittest
from unittest.mock import patch

from local_chat import context_manager as cm
from local_chat import server as srv
from local_chat.test_features import FeatureTestCase


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


class TokenAccountingTests(unittest.TestCase):
    def test_persian_costs_more_tokens_per_char_than_english(self):
        fa = "این یک جملهٔ فارسی برای آزمایش است " * 20
        en = "this is an english sentence for testing " * 18
        self.assertGreater(cm.estimate_tokens(fa) / len(fa), cm.estimate_tokens(en) / len(en))
        self.assertEqual(cm.estimate_tokens(""), 0)

    def test_budget_counts_every_component(self):
        budget = cm.ContextBudget(8192, 2048, system_tokens=500, tools_tokens=700, history_tokens=1500,
                                  file_tokens=800, memory_tokens=300)
        self.assertEqual(budget.input_tokens, 3800)
        self.assertEqual(budget.used, 3800 + 2048)
        self.assertAlmostEqual(budget.utilization, (3800 + 2048) / 8192)

    def test_compression_starts_between_70_and_80_percent(self):
        self.assertGreaterEqual(cm.COMPRESS_AT, 0.70)
        self.assertLessEqual(cm.COMPRESS_AT, 0.80)
        below = cm.ContextBudget(10000, 1000, history_tokens=5500)     # 65 %
        above = cm.ContextBudget(10000, 1000, history_tokens=6600)     # 76 %
        self.assertFalse(below.should_compress)
        self.assertTrue(above.should_compress)
        self.assertLess(above.input_target, above.input_limit)

    def test_capacity_is_read_from_ollama_show(self):
        self.assertEqual(cm.capacity_from_show({"model_info": {"qwen3.context_length": 40960}}), 40960)
        self.assertEqual(cm.capacity_from_show({"model_info": {"general.architecture": "x"}}), None)
        self.assertEqual(cm.capacity_from_show(None), None)


class CapacityClampTests(unittest.TestCase):
    def profile(self, capacity, ctx=8192, predict=2560):
        base = {"num_ctx": ctx, "num_predict": predict, "budget": int((ctx - predict) * 2.1)}
        with patch.object(srv, "_model_context_capacity", return_value=capacity):
            return srv._clamp_profile(base, "m")

    def test_window_never_exceeds_what_the_model_supports(self):
        for capacity in (2048, 4096, 8192, 32768, 131072, None):
            with self.subTest(capacity=capacity):
                result = self.profile(capacity)
                if capacity:
                    self.assertLessEqual(result["num_ctx"], capacity)
                self.assertGreater(result["budget"], 0)
                self.assertLess(result["num_predict"], result["num_ctx"])

    def test_small_window_shrinks_budget_and_output(self):
        small, large = self.profile(2048), self.profile(131072)
        self.assertEqual(large["num_ctx"], 8192)                      # asked-for size is kept, not inflated
        self.assertLess(small["budget"], large["budget"])
        self.assertLessEqual(small["num_predict"], 1024)

    def test_unknown_capacity_keeps_profile_untouched(self):
        base = {"num_ctx": 8192, "num_predict": 2560, "budget": 11827}
        with patch.object(srv, "_model_context_capacity", return_value=None):
            self.assertEqual(srv._clamp_profile(base, "m"), base)


class PinnedAndCheckpointTests(unittest.TestCase):
    def test_instructions_and_open_tasks_survive_compression(self):
        history = [user("همیشه جواب‌ها را کوتاه و فقط فارسی بنویس."),
                   assistant("باشه."),
                   user("فایل server.py هنوز باقی مانده؛ بعداً بررسیش کن.")]
        for index in range(60):
            history += [user(f"سؤال معمولی شمارهٔ {index} دربارهٔ موضوعی کاملاً بی‌ربط " * 4),
                        assistant(f"پاسخ شمارهٔ {index} " * 30)]
        history.append(user("خب حالا چی؟"))
        fitted = srv._fit_messages("SYS", history, 6000)
        system = fitted[0]["content"]
        self.assertIn("همیشه جواب‌ها را کوتاه", system)
        self.assertIn("server.py", system)
        self.assertLessEqual(sum(len(m["content"]) for m in fitted), 6200)
        self.assertEqual(fitted[-1]["content"], "خب حالا چی؟")

    def test_checkpoint_roundtrip_restores_state(self):
        messages = [user("پروژه را به Flutter منتقل کن. از این به بعد همیشه تست بنویس."), assistant("شروع کردم"),
                    user("ادامه بده")]
        checkpoint = cm.make_checkpoint(messages)
        text = cm.checkpoint_system_text(checkpoint)
        self.assertIn("همیشه تست بنویس", text)
        self.assertIn("ادامه بده", text)
        self.assertEqual(cm.checkpoint_system_text(None), "")


class IncompleteRequestGuardTests(FeatureTestCase):
    def test_guard_function_flags_overflow(self):
        ok, usage = cm.guard_request([user("ا" * 30000)], 4096, 1024)
        self.assertFalse(ok)
        self.assertFalse(usage.fits)
        ok, _ = cm.guard_request([user("سلام")], 4096, 1024)
        self.assertTrue(ok)

    def test_oversized_request_is_refused_not_truncated(self):
        before = len(self.ollama.chat_requests())
        with patch.object(srv, "_model_context_capacity", return_value=1024):
            status, body = self.chat("text:8b", "ب" * 20000)
        self.assertEqual(status, 413)
        self.assertIn("ارسال نشد", body.decode())
        self.assertEqual(len(self.ollama.chat_requests()), before)    # nothing reached Ollama

    def test_normal_request_uses_model_capacity_in_options(self):
        with patch.object(srv, "_model_context_capacity", return_value=4096):
            status, _ = self.chat("text:8b", "سلام، حالت چطوره؟")
        self.assertEqual(status, 200)
        options = self.ollama.chat_requests()[-1]["options"]
        self.assertLessEqual(options["num_ctx"], 4096)
        self.assertLess(options["num_predict"], options["num_ctx"])


if __name__ == "__main__":
    unittest.main()
