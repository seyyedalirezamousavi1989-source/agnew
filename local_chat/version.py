"""Single source of truth for the LocalChat version.

Everything that shows or checks a version (server, launcher, installer scripts, UI) reads it from here.
Bump ONLY this file (and packaging/installer.iss / build_installer.bat are checked against it by
test_version.py) whenever you make a new build.
"""
APP_VERSION = "2.0.5"
BUILD_NAME = "VoiceFix_MediaSearch_VersionPanel"
BUILD_DATE = "2026-11-09"
# short, human readable list shown in the «درباره و نسخه» panel (newest first)
CHANGELOG = (
    "پاسخ صوتی: Edge TTS خروجی MP3 می‌دهد و پخش‌کننده فقط WAV را می‌پذیرفت؛ اصلاح شد + عیب‌یابی صدا",
    "پنل «درباره و نسخه» در رابط کاربری با نسخهٔ سرور، ساخت، مسیر و شناسهٔ پردازش",
    "جست‌وجوی هوشمند: فیلم، ویدیو، آهنگ و عکس با لینک مستقیم؛ نمایش لینک‌ها بعد از پاسخ مدل",
)
