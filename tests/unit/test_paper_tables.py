"""The generated tables and number macros are current and well formed."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "paper" / "generated"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    argv, sys.argv = sys.argv, ["make_tables.py", "--out", str(out)]
    try:
        _load("make_tables").main()
    finally:
        sys.argv = argv
    return out


def test_committed_tables_equal_a_fresh_run(fresh):
    names = sorted(p.name for p in fresh.glob("*.tex"))
    assert names == sorted(p.name for p in GENERATED.glob("*.tex")) and len(names) == 16
    for name in names:
        assert (fresh / name).read_text() == (GENERATED / name).read_text(), name


def test_every_row_of_a_table_has_the_declared_number_of_columns(fresh):
    for path in fresh.glob("tab_*.tex"):
        text = path.read_text()
        spec = re.search(r"\\begin\{tabular\}\{(.*)\}\n", text).group(1)
        n_cols = len(re.findall(r"[lrc]|p\{[^}]*\}", spec))
        for line in text.splitlines():
            if line.endswith("\\\\") and "\\multicolumn" not in line and "&" in line:
                assert line.count("&") == n_cols - 1, (path.name, line)


def test_macros_quote_the_endpoint_file(fresh):
    e = json.loads((ROOT / "results" / "S5_endpoints.json").read_text())
    text = (fresh / "numbers.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))
    assert macros["PEoneITT"] == f"{e['PE1']['itt']['pooled_ratio']:.2f}"
    assert macros["PEtwoMean"] == f"{e['PE2']['mean_coverage']:.3f}"
    assert macros["PEthreeRate"] == f"{100 * e['PE3']['bad']['rate']:.2f}\\%"
    worst = e["PE4"]["pooled"]["rules"]
    assert macros["RegretPolicy"] == f"{worst['policy_debiased']['max_excess_srmse']:.3f}"
    assert macros["nCells"] == f"{e['cells']:,}".replace(",", "{,}")
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in macros)      # valid macro names


def _assert_same(a, b, where: str) -> None:
    """Equal structure, labels and counts; floats to 1e-9 (the last digit of a sum of 12,012
    cells depends on the numpy and pandas versions)."""
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), where
        for k in b:
            _assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{where}[{i}]")
    elif isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), where
    else:
        assert a == b, where


def test_posthoc_file_is_what_the_script_computes_from_the_cell_tables():
    post = _load("posthoc_descriptives")
    import glob

    import pandas as pd

    cells = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(str(ROOT / "results" / "compact" / "S5" / "primary" / "*.cells.parquet")))],
        ignore_index=True)
    committed = json.loads((ROOT / "results" / "S5_posthoc_descriptives.json").read_text())
    fresh = json.loads(json.dumps(post.cell_descriptives(cells), default=float))
    for key in ("cells", "pools", "best_single", "excess_srmse", "srmse", "mse_ratios"):
        _assert_same(fresh[key], committed[key], key)
    shares = committed["best_single"]["overall"]
    assert sum(shares.values()) == pytest.approx(1.0)
