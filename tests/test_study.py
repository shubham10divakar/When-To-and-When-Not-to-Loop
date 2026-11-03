"""Difficulty selection rule for the one-command local study."""

import pandas as pd

from loopbounds.pipeline import choose_difficulty


def _rep(ps):
    return pd.DataFrame([{"difficulty": lv, "arm": "naive", "p_hat": p} for lv, p in ps.items()])


def test_hardest_level_in_band_for_all_models():
    reps = {"a": _rep({"L0": 0.99, "L1": 0.93, "L2": 0.80, "L3": 0.6}),
            "b": _rep({"L0": 0.98, "L1": 0.90, "L2": 0.88, "L3": 0.7})}
    assert choose_difficulty(reps)[0] == "L1"


def test_falls_back_to_easy_enough_then_best_worst_case():
    too_easy = {"a": _rep({"L0": 0.99, "L1": 0.98}), "b": _rep({"L0": 0.995, "L1": 0.975})}
    assert choose_difficulty(too_easy)[0] == "L1"
    too_hard = {"a": _rep({"L0": 0.7, "L1": 0.4}), "b": _rep({"L0": 0.8, "L1": 0.5})}
    assert choose_difficulty(too_hard)[0] == "L0"
