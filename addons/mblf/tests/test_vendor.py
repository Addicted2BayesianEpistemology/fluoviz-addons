"""The vendored mblf-python: it imports without cmdstanpy / xarray, its Stan programs are there, the job script imports nothing of FluoViz, and the engine is located in plain words."""
import ast
import subprocess
import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
PKG = ADDON / "fluoviz_addon_mblf"


def test_the_copy_imports_with_numpy_and_scipy_only():
    code = f"import sys; sys.path.insert(0, {str(PKG / 'vendor')!r}); import mblf; assert mblf.__version__ == '1.0.0'; assert 'cmdstanpy' not in sys.modules and 'xarray' not in sys.modules; print('ok')"
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-600:]


def test_the_stan_programs_and_the_licence_are_there():
    for f in ("stan/mblf.stan", "stan/mblf_ml.stan", "LICENSE"):
        assert (PKG / "vendor" / "mblf" / f).is_file(), f
    assert "MIT" in (PKG / "vendor" / "mblf" / "LICENSE").read_text(encoding="utf-8")


def test_the_job_script_is_standalone():
    tree = ast.parse((PKG / "engine_job.py").read_text(encoding="utf-8"))
    mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("fluoviz", "fluoviz_addon")) for m in mods), mods


def test_the_engine_says_what_is_missing(tmp_path, monkeypatch):
    from fluoviz_addon_mblf.core import engine as E
    monkeypatch.delenv("FLUOVIZ_MBLF_ENGINE", raising=False)
    monkeypatch.delenv("FLUOVIZ_MBLF_ENGINE_DIR", raising=False)
    eng = E.locate(tmp_path / "data")
    assert eng.state in ("missing", "unavailable") and eng.message
    if eng.state == "missing":
        assert "450 MB" in eng.message and "not changed" in eng.message
    spaced = E.locate(tmp_path / "with space")
    assert spaced.state == "unavailable" and ("space" in spaced.message or "installer" in spaced.message)
    monkeypatch.setenv("FLUOVIZ_MBLF_ENGINE", str(tmp_path / "nothing"))
    assert E.locate(tmp_path).state == "unavailable"
