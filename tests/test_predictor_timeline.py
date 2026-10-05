"""Regression tests for the frozen predictor's information/target timeline.

Run from the package root: python tests/test_predictor_timeline.py
These tests use synthetic archives and the published frozen checkpoint. They
do not train a model or read private production data. The optional audit output
describes test evidence, not scientific replay outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

from controller_replay import LockedModelBank
from model_lopo import EXOG, WINDOW, sequences


class LastObservedPlusOne(torch.nn.Module):
    """Synthetic oracle: row-index thickness at k forecasts k+1."""

    def __init__(self):
        super().__init__()
        self.seen_windows = None

    def forward(self, x):
        self.seen_windows = x.detach().cpu().numpy().copy()
        return x[:, -1, 0] + 1.0


def stub_bank(window=4):
    bank = LockedModelBank.__new__(LockedModelBank)
    bank.device = torch.device("cpu")
    bank.features = ["exit_thickness_dev"]
    bank.window = window
    bank.mu = np.zeros((1, 1, 1), dtype=np.float32)
    bank.sd = np.ones((1, 1, 1), dtype=np.float32)
    bank.model = LastObservedPlusOne()
    return bank


def synthetic_frame(n, segments=None):
    values = np.arange(n, dtype=float)
    frame = pd.DataFrame({"exit_thickness_dev": values})
    for i, feature in enumerate(EXOG):
        frame[feature] = 0.01 * (i + 1) * values + (i + 1)
    frame["effective_dt_s"] = 0.5 + 0.01 * values
    frame["distance_step_m"] = 0.8 + 0.002 * values
    frame["segment_id"] = np.zeros(n, dtype=int) if segments is None else segments
    return frame


def prepare(bank, frame):
    bank.prepare(
        frame,
        frame["effective_dt_s"].to_numpy(),
        frame["distance_step_m"].to_numpy(),
        frame["segment_id"].to_numpy(),
    )


class PredictorTimelineTests(unittest.TestCase):
    def test_decision_k_predicts_target_k_plus_one(self):
        frame = synthetic_frame(20)
        bank = stub_bank()
        prepare(bank, frame)
        current = np.array([100.0, -20.0, 300.0])
        prediction = bank.predict(12, current, 10)
        np.testing.assert_array_equal(prediction[:, 1], np.full(10, 13.0))
        np.testing.assert_array_equal(prediction[:, [0, 2]], np.tile(current[[0, 2]], (10, 1)))
        self.assertEqual(bank.target_rows[12], 13)
        # The input ending at decision k=12 is rows 9..12, never rows 8..11.
        position = np.flatnonzero(np.isfinite(bank.cached)).tolist().index(12)
        np.testing.assert_array_equal(bank.model.seen_windows[position, :, 0], [9, 10, 11, 12])

    def test_training_input_and_target_alignment(self):
        frame = synthetic_frame(22)
        inputs, targets, naive = sequences(frame, "exit_thickness_dev")
        bank = stub_bank(WINDOW)
        bank.features = ["exit_thickness_dev", *EXOG]
        bank.mu = np.zeros((1, 1, len(bank.features)), dtype=np.float32)
        bank.sd = np.ones((1, 1, len(bank.features)), dtype=np.float32)
        prepare(bank, frame)
        rows = np.flatnonzero(np.isfinite(bank.cached))
        np.testing.assert_array_equal(rows, np.arange(WINDOW, len(frame) - 1))
        np.testing.assert_array_equal(bank.model.seen_windows, inputs)
        np.testing.assert_array_equal(bank.cached[rows], targets)
        np.testing.assert_array_equal(naive, rows)
        np.testing.assert_array_equal(bank.target_rows[rows], rows + 1)

    def test_segment_boundaries_and_short_segments_fall_back(self):
        # Long first/last segments surround a segment too short for any forecast.
        frame = synthetic_frame(24, np.repeat([0, 1, 2], [10, 3, 11]))
        bank = stub_bank()
        prepare(bank, frame)
        valid = np.array([4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22])
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(bank.cached)), valid)
        current = np.array([1.0, 987.0, 3.0])
        for row in [0, 3, 9, 10, 11, 12, 13, 16, 23]:
            np.testing.assert_array_equal(bank.predict(row, current, 3), np.tile(current, (3, 1)))
            self.assertEqual(bank.target_rows[row], -1)
        for row in valid:
            self.assertEqual(frame.segment_id.iloc[row], frame.segment_id.iloc[row + 1])
            self.assertEqual(bank.cached[row], row + 1)

    def test_future_feature_values_cannot_change_current_forecast(self):
        frame = synthetic_frame(24)
        reference = stub_bank()
        prepare(reference, frame)
        modified = frame.copy()
        feature_columns = ["exit_thickness_dev", *EXOG]
        modified.loc[13:, feature_columns] += 10000.0
        changed = stub_bank()
        prepare(changed, modified)
        np.testing.assert_array_equal(reference.cached[:13], changed.cached[:13])
        self.assertEqual(changed.cached[12], 13.0)
        self.assertNotEqual(reference.cached[13], changed.cached[13])

    def test_recurrent_segment_ids_do_not_bridge_intervening_segment(self):
        frame = synthetic_frame(20, np.repeat([0, 1, 0], [8, 4, 8]))
        bank = stub_bank()
        prepare(bank, frame)
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(bank.cached)), [4, 5, 6, 16, 17, 18])

    def test_empty_and_single_row_archives_have_no_next_target(self):
        for n in [0, 1]:
            bank = stub_bank()
            prepare(bank, synthetic_frame(n))
            self.assertEqual(len(bank.cached), n)
            self.assertFalse(np.isfinite(bank.cached).any())
        current = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(bank.predict(0, current, 2), np.tile(current, (2, 1)))

    def test_invalid_indices_and_input_lengths_are_rejected(self):
        bank = stub_bank()
        frame = synthetic_frame(12)
        prepare(bank, frame)
        for row in [-1, 12, 100]:
            with self.assertRaises(IndexError):
                bank.predict(row, np.ones(3), 1)
        with self.assertRaises(TypeError):
            bank.predict(1.2, np.ones(3), 1)
        with self.assertRaises(ValueError):
            bank.prepare(frame, np.ones(11), np.ones(12), np.zeros(12))
        with self.assertRaises(ValueError):
            bank.prepare(frame, np.ones(12), np.ones(12), np.zeros(11))

    def test_frozen_gru_cache_matches_direct_training_windows(self):
        bank = LockedModelBank(str(PACKAGE_ROOT / "models" / "final_thickness_model.pt"))
        self.assertEqual(bank.window, WINDOW)
        self.assertEqual(bank.features, ["exit_thickness_dev", *EXOG])
        frame = synthetic_frame(34, np.repeat([0, 1], [18, 16]))
        inputs, targets, _ = sequences(frame, "exit_thickness_dev")
        normalized = (inputs - bank.mu) / bank.sd
        with torch.no_grad():
            direct = bank.model(torch.tensor(normalized, device=bank.device)).cpu().numpy()
        prepare(bank, frame)
        # Synthetic target values equal row indices, so this establishes exact
        # information-row / next-target mapping with the actual frozen GRU.
        rows = targets.astype(int) - 1
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(bank.cached)), rows)
        np.testing.assert_allclose(bank.cached[rows], direct, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(bank.target_rows[rows], targets.astype(int))
        self.assertTrue(np.isnan(bank.cached[[0, 7, 17, 18, 25, 33]]).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PredictorTimelineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.audit_json:
        checkpoint = PACKAGE_ROOT / "models" / "final_thickness_model.pt"
        audit = {
            "protocol": "observation/control row k -> target archived row k+1",
            "feature_information_cutoff": "row k inclusive; no target feature values",
            "eligibility": "training-compatible same-segment warm-up, window and next target",
            "fallback": "current-value persistence at ineligible rows and last archive row",
            "planning_stages": "repeat one-update forecast, not recursive multi-step prediction",
            "scope": "synthetic regression tests and direct inference with frozen CPU GRU; no private data or training",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "passed": result.wasSuccessful(),
        }
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
