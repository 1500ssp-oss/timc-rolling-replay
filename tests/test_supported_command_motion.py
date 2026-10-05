"""Regression tests for supported command transitions and reset motion.

Uses synthetic row logs and the actual metric entry points; no private records,
model training, controller trajectories, or published outputs are written.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

import command_audit
import controller_replay
import protocol
from supported_command_motion import (
    reset_command_motion,
    supported_command_deltas,
    supported_total_variation,
    supported_transition_mask,
)


def log_frame(values, labels=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = np.column_stack([values, np.zeros((len(values), 2))])
    if labels is None:
        labels = np.zeros(len(values), dtype=int)
    labels = np.asarray(labels)
    starts = np.r_[True, labels[1:] != labels[:-1]]
    logged_du = np.vstack([values[0], np.diff(values, axis=0)])
    logged_du[starts] = values[starts]
    frame = pd.DataFrame({
        "k": np.arange(len(values)), "time_s": np.arange(len(values), dtype=float),
        "segment_id": labels, "control": "probe", "pass_id": "probe", "file": "probe",
        "timestamp_dt_s": 1., "distance_step_m": np.where(starts, 0., 1.),
        "mpc_share": .5, "ff_share": 1., "sat_flag": 0, "phase": "steady",
    })
    for channel_i, channel in enumerate(("speed", "gap", "shape")):
        for prefix, factor in (("u", 1.), ("u_preclamp", 3.), ("u_target_unclipped", 4.),
                               ("u_mpc", 2.), ("u_mpc_component", 1.),
                               ("u_ff", 1.), ("u_pid", 1.), ("u_adrc", 1.)):
            frame[f"{prefix}_{channel}"] = values[:, channel_i] * factor
        frame[f"du_{channel}"] = logged_du[:, channel_i]
    for channel, error in (("T", .1), ("h", .2), ("S", .3)):
        frame[f"{channel}_error"] = error
        frame[f"{channel}_control_error"] = error
    return frame


class SupportedCommandMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(PACKAGE_ROOT / "code" / "engine_core"))
        cls.suite = protocol.load_module("suite_supported_motion_test", PACKAGE_ROOT / "code" / "engine_core" / "15_realdata_closed_loop_suite.py")
        cls.p = SimpleNamespace(pass_file="probe", tension_scale=1., thickness_scale=1., flatness_scale=1.)
        cls.support = pd.Series({
            f"u_{channel}_{bound}": value
            for channel in ("speed", "gap", "shape")
            for bound, value in (("p005", -10.), ("p995", 10.))
        })

    def assert_metric_paths(self, frame, expected_tv, expected_startup, expected_restart):
        original_du = frame[["du_speed", "du_gap", "du_shape"]].to_numpy().copy()
        original_commands = frame[["u_speed", "u_gap", "u_shape"]].to_numpy().copy()
        mechanism = controller_replay.add_mechanism_metrics({}, frame, self.p)
        audit = command_audit.audit_group(frame, self.support, self.support)
        extra = protocol.add_extra_metrics({}, frame, self.p, SimpleNamespace(reference_kind="none"), {})
        legacy = self.suite.control_metrics(frame, self.p, {}, "C0", "S5", 1)
        self.assertAlmostEqual(mechanism["postclamp_total_variation"], expected_tv)
        self.assertAlmostEqual(mechanism["preclamp_total_variation"], 3 * expected_tv)
        self.assertAlmostEqual(mechanism["mpc_raw_total_variation"], 2 * expected_tv)
        self.assertAlmostEqual(mechanism["mpc_component_total_variation"], expected_tv)
        self.assertAlmostEqual(audit["post_projection_tv"], expected_tv)
        self.assertAlmostEqual(audit["pre_projection_tv"], 4 * expected_tv)
        self.assertAlmostEqual(legacy["control_total_variation"], expected_tv)
        for metrics in (mechanism, audit, legacy):
            self.assertAlmostEqual(metrics["startup_motion"], expected_startup)
            self.assertAlmostEqual(metrics["restart_motion"], expected_restart)
            self.assertAlmostEqual(metrics["startup_restart_motion"], expected_startup + expected_restart)
        self.assertAlmostEqual(legacy["composite_normalized_RMS"], .2)
        np.testing.assert_array_equal(original_du, frame[["du_speed", "du_gap", "du_shape"]].to_numpy())
        np.testing.assert_array_equal(original_commands, frame[["u_speed", "u_gap", "u_shape"]].to_numpy())
        return mechanism, audit, extra, legacy

    def test_two_constant_segments_exclude_join(self):
        _, audit, extra, legacy = self.assert_metric_paths(log_frame([0., 0., 1., 1.], [0, 0, 1, 1]), 0., 0., 1.)
        self.assertEqual(audit["TV_L_per_100m"], 0.)
        self.assertEqual(extra["delta_u_over_peak_action"], 0.)
        self.assertEqual(legacy["control_delta_rms"], 0.)

    def test_two_single_row_segments_have_no_supported_transition(self):
        _, audit, extra, legacy = self.assert_metric_paths(log_frame([.3, .5], [0, 1]), 0., .3, .5)
        self.assertTrue(np.isnan(audit["TV_L_per_100m"]))
        self.assertEqual(extra["delta_u_over_peak_action"], 0.)
        self.assertEqual(legacy["control_delta_rms"], 0.)

    def test_single_row_file_has_only_startup(self):
        _, audit, extra, legacy = self.assert_metric_paths(log_frame([.4]), 0., .4, 0.)
        self.assertTrue(np.isnan(audit["TV_L_per_100m"]))
        self.assertEqual(extra["delta_u_over_peak_action"], 0.)
        self.assertEqual(legacy["control_delta_rms"], 0.)

    def test_repeated_label_does_not_reconnect_runs(self):
        self.assert_metric_paths(log_frame([0., 0., 1., 1., 2., 2.], [0, 0, 1, 1, 0, 0]), 0., 0., 3.)

    def test_command_audit_restores_row_order_without_label_sort(self):
        frame = log_frame([0., .1, .4, .4, .2, .3], [0, 0, 1, 1, 0, 0]).iloc[[4, 2, 0, 5, 1, 3]]
        _, audit, _, _ = self.assert_metric_paths(frame.sort_values("k"), .2, 0., .6)
        shuffled_audit = command_audit.audit_group(frame, self.support, self.support)
        self.assertAlmostEqual(shuffled_audit["post_projection_tv"], audit["post_projection_tv"])
        self.assertAlmostEqual(shuffled_audit["restart_motion"], .6)

    def test_single_continuous_segment_keeps_adjacent_variation(self):
        _, audit, extra, legacy = self.assert_metric_paths(log_frame([0., .2, .1]), .3, 0., 0.)
        self.assertAlmostEqual(audit["TV_L_per_100m"], 15.)
        self.assertAlmostEqual(extra["delta_u_over_peak_action"], .2 / (.2 + 1e-9))
        self.assertAlmostEqual(legacy["control_delta_rms"], np.sqrt((.2 ** 2 + .1 ** 2) / 6.))

    def test_nonzero_initial_command_is_not_supported_tv(self):
        _, audit, _, _ = self.assert_metric_paths(log_frame([.4, .5]), .1, .4, 0.)
        self.assertAlmostEqual(audit["TV_t_per_s"], .05)
        self.assertAlmostEqual(audit["TV_L_per_100m"], 10.)

    def test_multichannel_l1_norm_and_signed_changes(self):
        values = np.array([[.1, -.2, .3], [.2, -.1, .1], [-.4, .2, -.1], [-.4, .1, -.2]])
        self.assert_metric_paths(log_frame(values, [0, 0, 1, 1]), .6, .6, .7)
        np.testing.assert_allclose(supported_command_deltas(values, [0, 0, 1, 1]), [[.1, .1, -.2], [0., -.1, -.1]])

    def test_no_labels_means_one_continuous_run(self):
        frame = log_frame([.4, .2, .5]).drop(columns="segment_id")
        self.assertAlmostEqual(controller_replay.total_variation_from_cols(frame, ["u_speed", "u_gap", "u_shape"]), .5)
        self.assertAlmostEqual(supported_total_variation([.4, .2, .5]), .5)
        self.assertEqual(reset_command_motion([.4, .2, .5])["restart_motion"], 0.)

    def test_empty_commands_have_zero_motion(self):
        self.assertEqual(supported_total_variation(np.empty((0, 3)), []), 0.)
        self.assertEqual(supported_command_deltas(np.empty((0, 3)), []).shape, (0, 3))
        self.assertEqual(reset_command_motion(np.empty((0, 3)), [])["startup_restart_motion"], 0.)

    def test_invalid_segment_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            supported_total_variation([0., 1.], [0])
        with self.assertRaises(ValueError):
            supported_transition_mask([[0], [0]], 2)

    def test_peak_change_excludes_large_restart_join(self):
        frame = log_frame([.1, .15, -.5, -.45], [0, 0, 1, 1])
        _, _, extra, _ = self.assert_metric_paths(frame, .1, .1, .5)
        self.assertAlmostEqual(extra["delta_u_over_peak_action"], .05 / (.5 + 1e-9))

    def test_protocol_suite_metrics_preserves_reset_fields(self):
        frame = log_frame([.1, .15, -.5, -.45], [0, 0, 1, 1])
        rp = SimpleNamespace(p=self.p, source="synthetic", pass_id="probe", condition_id="probe", trial_id="probe", file="probe")
        diag = {"seed": 1, "control": "C0", "sat_count": 0, "sat_ratio": 0., "invalid_command_count": 0}
        metrics = protocol.suite_metrics(self.suite, rp, frame, diag, "C0", protocol.ScenarioSpec("probe", "nominal"))
        self.assertAlmostEqual(metrics["control_total_variation"], .1)
        self.assertAlmostEqual(metrics["startup_motion"], .1)
        self.assertAlmostEqual(metrics["restart_motion"], .5)
        self.assertAlmostEqual(metrics["startup_restart_motion"], .6)

    def test_protocol_summary_retains_new_motion_columns(self):
        summary = protocol.aggregate_metrics(pd.DataFrame({
            "control": ["probe", "probe"], "control_delta_rms": [.1, .3],
            "startup_motion": [.2, .4], "restart_motion": [.5, .7], "startup_restart_motion": [.7, 1.1],
        }), ["control"])
        self.assertAlmostEqual(summary["control_delta_rms_mean"].iat[0], .2)
        self.assertAlmostEqual(summary["startup_motion_mean"].iat[0], .3)
        self.assertAlmostEqual(summary["restart_motion_mean"].iat[0], .6)
        self.assertAlmostEqual(summary["startup_restart_motion_mean"].iat[0], .9)

    def test_peak_slew_excludes_startup_with_zero_interval(self):
        frame = log_frame([.4, .4])
        frame.loc[0, "timestamp_dt_s"] = 0.
        original_du = frame[["du_speed", "du_gap", "du_shape"]].to_numpy().copy()
        metrics = command_audit.audit_group(frame, self.support, self.support)
        self.assertEqual(metrics["peak_slew_rate_per_s"], 0.)
        self.assertAlmostEqual(metrics["startup_motion"], .4)
        np.testing.assert_array_equal(original_du, frame[["du_speed", "du_gap", "du_shape"]].to_numpy())

    def test_peak_slew_uses_only_positive_finite_supported_intervals(self):
        frame = log_frame([0., .1, .2, .3, .4])
        frame["timestamp_dt_s"] = [1., 0., float("nan"), float("inf"), 2.]
        metrics = command_audit.audit_group(frame, self.support, self.support)
        self.assertAlmostEqual(metrics["peak_slew_rate_per_s"], .05)

    def test_peak_slew_without_supported_transitions_is_zero(self):
        frame = log_frame([.3, -.5], [0, 1])
        frame["timestamp_dt_s"] = 0.
        metrics = command_audit.audit_group(frame, self.support, self.support)
        self.assertEqual(metrics["peak_slew_rate_per_s"], 0.)
        self.assertAlmostEqual(metrics["startup_restart_motion"], .8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
