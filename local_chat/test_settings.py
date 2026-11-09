import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_chat import settings


class ModelsDirSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.dict(
            os.environ, {"LOCALAPPDATA": self.tmp.name}, clear=False
        )
        self.patch.start()
        os.environ.pop("OLLAMA_MODELS", None)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_portable_data_dir_overrides_localappdata(self):
        portable = Path(self.tmp.name) / "portable-data"
        with patch.dict(os.environ, {"LOCALCHAT_DATA_DIR": str(portable)}):
            self.assertEqual(settings.app_data_dir(), portable.resolve())

    def test_default_when_nothing_configured(self):
        self.assertEqual(settings.effective_models_dir(), settings.default_models_dir())
        self.assertNotIn("OLLAMA_MODELS", settings.ollama_environment())

    def test_env_var_used_when_no_app_setting(self):
        with patch.dict(os.environ, {"OLLAMA_MODELS": "/from/env"}):
            self.assertEqual(settings.effective_models_dir(), Path("/from/env"))

    def test_app_setting_wins_and_is_passed_to_ollama(self):
        folder = Path(self.tmp.name) / "m"
        folder.mkdir()
        with patch.dict(os.environ, {"OLLAMA_MODELS": "/from/env"}):
            settings.set_models_dir(str(folder))
            self.assertEqual(settings.effective_models_dir(), folder)
            self.assertEqual(settings.ollama_environment()["OLLAMA_MODELS"], str(folder))

    def test_quotes_are_stripped_and_empty_resets(self):
        folder = Path(self.tmp.name) / "m2"
        folder.mkdir()
        settings.set_models_dir(f'"{folder}"')
        self.assertEqual(settings.configured_models_dir(), str(folder))
        settings.set_models_dir("  ")
        self.assertEqual(settings.configured_models_dir(), "")

    def test_rejects_missing_relative_and_non_string(self):
        for bad in ("nope", str(Path(self.tmp.name) / "missing"), None, 5):
            with self.assertRaises(ValueError):
                settings.set_models_dir(bad)

    def test_corrupt_settings_file_is_ignored(self):
        path = settings.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(settings.configured_models_dir(), "")

    def test_inspect_counts_models(self):
        folder = Path(self.tmp.name) / "o"
        (folder / "manifests" / "reg").mkdir(parents=True)
        (folder / "blobs").mkdir()
        (folder / "manifests" / "reg" / "a").write_text("x")
        (folder / "manifests" / "reg" / "b").write_text("x")
        info = settings.inspect_models_dir(folder)
        self.assertTrue(info["looks_like_ollama"])
        self.assertEqual(info["model_count"], 2)
        self.assertFalse(settings.inspect_models_dir(folder / "zzz")["exists"])


if __name__ == "__main__":
    unittest.main()
