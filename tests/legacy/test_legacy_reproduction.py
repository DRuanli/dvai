import numpy as np
import pytest

pytest.importorskip("ppi_py")

from d2vi.estimators.legacy import legacy_replication  # noqa: E402


def _pools(pool):
    df = pool.df
    h, m = df[df["a"] == 0], df[df["a"] == 1]

    def cols(part):
        return {"score": part["score"].to_numpy(), "flag": part["flag"].to_numpy(),
                "length": part["length"].to_numpy(), "y_sent": part["y_sent"].to_numpy()}

    return cols(h), cols(m)


def test_v1_fallback_reports_flag_ppi_when_rc_ppi_abstains(pool):
    human, machine = _pools(pool)
    rows = []
    for seed in range(8):
        rows += legacy_replication(human, machine, 900, 0.05, seed, 0.05, ["sent"])
    fallback = [r for r in rows if np.isnan(r["rc_strat_raw"])]
    assert all(r["rc_strat"] == r["ppi"] for r in fallback)
    # the two v1 audits do not share a budget
    assert len({r["m_random"] for r in rows}) > 1
    assert all(r["m_strat"] == 45 for r in rows)
