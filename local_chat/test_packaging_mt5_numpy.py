from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_onedir_spec_uses_mt5_collection_and_numpy_builtin_hook():
    spec = (ROOT / "packaging" / "localchat.spec").read_text(encoding="utf-8")
    assert 'collect_all("MetaTrader5")' in spec
    assert 'collect_all("numpy")' not in spec
    assert "numpy.core.multiarray" in spec
    assert "numpy.core._multiarray_umath" in spec
    assert "*mt5_binaries" in spec
    assert '"numpy"' not in spec.split("excludes=", 1)[1].split(")", 1)[0]


def test_onefile_spec_uses_mt5_collection_and_numpy_builtin_hook():
    spec = (ROOT / "packaging" / "localchat_onefile.spec").read_text(encoding="utf-8")
    assert 'collect_all("MetaTrader5")' in spec
    assert 'collect_all("numpy")' not in spec
    assert "numpy.core.multiarray" in spec
    assert "numpy.core._multiarray_umath" in spec
    assert "*mt5_binaries" in spec
    assert '"numpy"' not in spec.split("excludes=", 1)[1].split("]", 1)[0]


def test_installer_validates_numpy_multiarray_before_freezing():
    script = (ROOT / "build_installer.bat").read_text(encoding="utf-8")
    assert '"numpy>=1.26.4,<2.0"' in script
    assert "from numpy.core import multiarray" in script
    assert "import MetaTrader5" in script


def test_pyinstaller_specs_create_nested_copy_destination_parents():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("packaging/localchat.spec", "packaging/localchat_onefile.spec"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "def _copyfile_with_parent(src, dst, *args, **kwargs):" in source
        assert "os.makedirs(_dst_parent, exist_ok=True)" in source
        assert "_shutil.copyfile = _copyfile_with_parent" in source


def test_specs_use_extended_length_paths_for_collect_copies():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("packaging/localchat.spec", "packaging/localchat_onefile.spec"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "def _long_path(p):" in source
        assert "_shutil.copystat = _copystat_long" in source


def test_installer_builds_payload_in_short_path():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bat = (root / "build_installer.bat").read_text(encoding="utf-8")
    iss = (root / "packaging" / "installer.iss").read_text(encoding="utf-8")
    assert '%LOCALAPPDATA%\\LCB' in bat
    assert '/DPayloadDir=%APPDIR%' in bat
    assert '{#PayloadDir}' in iss


def test_verify_bundle_flags_missing_mt5_and_numpy_files(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("verify_bundle", root / "packaging" / "verify_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "LocalChat.exe").write_text("x")
    assert len(module.check(tmp_path)) >= 4          # nothing bundled yet
    internal = tmp_path / "_internal"
    (internal / "MetaTrader5").mkdir(parents=True)
    (internal / "MetaTrader5" / "_core.pyd").write_text("x")
    (internal / "numpy" / "core").mkdir(parents=True)
    (internal / "numpy" / "core" / "_multiarray_umath.cp310-win_amd64.pyd").write_text("x")
    (internal / "numpy.libs").mkdir()
    (internal / "numpy.libs" / "libopenblas64.dll").write_text("x")
    (internal / "python310.dll").write_text("x")
    assert module.check(tmp_path) == []


def test_build_script_verifies_bundle_before_inno():
    from pathlib import Path

    bat = (Path(__file__).resolve().parents[1] / "build_installer.bat").read_text(encoding="utf-8")
    assert bat.index("verify_bundle.py") < bat.index("ISCC=")
