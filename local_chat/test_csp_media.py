from pathlib import Path


def test_csp_allows_blob_audio_playback():
    source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")
    assert "media-src 'self' blob:;" in source
    assert "default-src 'self';" in source
