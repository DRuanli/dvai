"""The S5 driver takes its grids from the frozen protocol file."""

import importlib.util
from pathlib import Path

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
