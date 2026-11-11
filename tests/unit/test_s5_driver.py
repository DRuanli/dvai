"""The S5 driver takes its grids from the frozen protocol file."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_s5", ROOT / "scripts" / "run_s5.py")
run_s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_s5)
FROZEN = yaml.safe_load((ROOT / "configs" / "protocol" / "frozen.yaml").read_text())


@pytest.mark.parametrize("grid, cells, reps", [
    ("primary", 21, 1000), ("small_budget", 7, 500), ("large_n", 2, 500), ("designs", 30, 500),
    ("mixtures", 6, 500), ("verification_error", 50, 500)])
def test_grids_follow_the_frozen_protocol(grid, cells, reps):
    specs, n_rep = run_s5.cell_specs(grid, FROZEN, "c", "d", "c__g__d")
    assert (len(specs), n_rep) == (cells, reps)
    if grid != "designs":
        assert all(s.design == FROZEN["audit"]["default_design"] for s in specs)
    assert all(s.tags == {"pool": "c__g__d"} for s in specs)


def test_build_pool_selects_rows_and_drops_unscorable():
    feat = pd.DataFrame({"doc_id": list("abcdef"), "a": [0, 0, 0, 1, 1, 1],
                         "source": ["human"] * 3 + ["g1", "g1", "g2"], "length": 10,
                         "y_t": 0.5, "sc": [0.1, 0.2, None, 0.9, 0.8, 0.7],
                         "fl": [0, 0, None, 1, 1, 1]})
    cfg = {"human": {"source": "human"}, "machine_groups": {"one": {"source": "g1"}},
           "detectors": {"x": {"score": "sc", "flag": "fl"}}, "outcomes": ["t"]}
    pool, info = run_s5.build_pool(feat, cfg, "one", "x")
    assert pool["doc_id"].tolist() == ["a", "b", "d", "e"]
    assert info == {"human": 2, "machine": 2, "unscorable_dropped": 1,
                    "flag_rate_human": 0.0, "flag_rate_machine": 1.0}
    assert set(pool["generator"]) == {"human", "g1"}


def test_anes_config_matches_the_protocol_pools():
    cfg = yaml.safe_load((ROOT / "configs" / "s5" / "anes.yaml").read_text())
    assert len(cfg["machine_groups"]) * len(cfg["detectors"]) == 9
    assert len(cfg["outcomes"]) == 4


def test_heldout_configs_give_one_pool_per_domain_group_and_detector():
    raid = yaml.safe_load((ROOT / "configs" / "s5" / "raid.yaml").read_text())
    plan = run_s5.pool_plan(raid)
    assert len(plan) == 5 * (4 + 2 * 3) * 2 and len({name for name, *_ in plan}) == len(plan)
    assert ("raid__poetry-llama-chat-homoglyph__fdgpt", "poetry", "llama-chat-homoglyph",
            "fdgpt") in plan
    beemo = yaml.safe_load((ROOT / "configs" / "s5" / "beemo.yaml").read_text())
    assert [n for n, *_ in run_s5.pool_plan(beemo)][:2] == ["beemo__machine__fdgpt",
                                                           "beemo__machine__tfidf"]
    assert len(run_s5.pool_plan(beemo)) == 8


def test_data_dependent_shortcuts_of_a_configuration_are_resolved_from_the_table():
    feat = pd.DataFrame({"domain": ["law", "baike", "law"], "a": [0, 1, 1],
                         "source": ["human", "g2", "g1"]})
    cfg = run_s5.resolve({"corpus": "c", "domains": "all", "machine_groups": {"by": "source"},
                          "detectors": {"d": {}}}, feat)
    assert cfg["domains"] == ["baike", "law"]
    assert cfg["machine_groups"] == {"g1": {"source": "g1"}, "g2": {"source": "g2"}}
    assert len(run_s5.pool_plan(cfg)) == 4
    for name, pools in (("m4", 24), ("hc3zh", None), ("daigt", None)):
        c = yaml.safe_load((ROOT / "configs" / "s5" / f"{name}.yaml").read_text())
        assert c["corpus"] == name
        if pools:
            assert len(run_s5.pool_plan(c)) == pools


def test_driver_records_empty_pools_and_stops_at_the_time_budget(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_h, n_m = 1000, 500
    feat = pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(n_h + n_m)], "a": [0.0] * n_h + [1.0] * n_m,
        "source": ["human"] * n_h + ["g1"] * n_m, "domain": "x",
        "length": rng.integers(50, 400, n_h + n_m), "y_o": rng.normal(size=n_h + n_m),
        "score_d": np.r_[rng.normal(0, 1, n_h), rng.normal(2, 1, n_m)]})
    feat["flag_d"] = (feat["score_d"] > 1).astype(int)
    feat.to_parquet(tmp_path / "f.parquet")
    cfg = {"corpus": "toy", "features": str(tmp_path / "f.parquet"), "outcomes": ["o"],
           "human": {"source": "human"},
           "machine_groups": {"g1": {"source": "g1"}, "g2": {"source": "g2"}},
           "detectors": {"d": {"score": "score_d", "flag": "flag_d"}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(ROOT)
    argv = ["run_s5.py", str(tmp_path / "c.yaml"), "--grid", "large_n", "--reps", "2",
            "--jobs", "1", "--out-root", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", [*argv, "--max-minutes", "-1"])
    run_s5.main()                                     # budget already spent: nothing runs
    man = json.loads((tmp_path / "out" / "large_n" / "toy.manifest.json").read_text())
    assert man["pools"] == {}
    monkeypatch.setattr("sys.argv", argv)
    run_s5.main()
    man = json.loads((tmp_path / "out" / "large_n" / "toy.manifest.json").read_text())
    assert man["pools"]["toy__g2__d"]["empty"] and man["pools"]["toy__g2__d"]["cells_run"] == 0
    assert man["pools"]["toy__g1__d"]["cells_skipped"]          # N = 2,000 > pool: skipped
    assert not (tmp_path / "out" / "large_n" / "toy__g2__d.parquet").exists()


def test_reports_from_compact_files_equal_reports_from_raw_files(tmp_path, monkeypatch):
    import subprocess
    import sys

    rng = np.random.default_rng(1)
    n_h, n_m = 1000, 500
    feat = pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(n_h + n_m)], "a": [0.0] * n_h + [1.0] * n_m,
        "source": ["human"] * n_h + ["g1"] * n_m, "domain": "x",
        "length": rng.integers(50, 400, n_h + n_m),
        "y_o": np.r_[rng.normal(0, 1, n_h), rng.normal(0.5, 1, n_m)],
        "score_d": np.r_[rng.normal(0, 1, n_h), rng.normal(2, 1, n_m)]})
    feat["flag_d"] = (feat["score_d"] > 1).astype(int)
    feat.to_parquet(tmp_path / "f.parquet")
    cfg = {"corpus": "toy", "features": str(tmp_path / "f.parquet"), "outcomes": ["o"],
           "human": {"source": "human"}, "machine_groups": {"g1": {"source": "g1"}},
           "detectors": {"d": {"score": "score_d", "flag": "flag_d"}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("sys.argv", ["run_s5.py", str(tmp_path / "c.yaml"), "--grid",
                                     "small_budget", "--reps", "4", "--jobs", "1",
                                     "--out-root", str(tmp_path / "raw")])
    run_s5.main()
    py, scripts = sys.executable, ROOT / "scripts"

    def run(*args):
        subprocess.run([py, *map(str, args)], check=True, capture_output=True, cwd=ROOT)

    run(scripts / "compact_s5.py", tmp_path / "raw" / "small_budget",
        tmp_path / "compact" / "small_budget")
    for src in ("raw", "compact"):
        run(scripts / "report_endpoints.py", tmp_path / src / "small_budget", "--out",
            tmp_path / f"{src}.json", "--boot", "200")
    a, b = (json.loads((tmp_path / f"{s}.json").read_text()) for s in ("raw", "compact"))
    a.pop("source"), b.pop("source")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
