"""Regenerate the manuscript figure assets from the replay pipeline outputs.

Figure file mapping (LaTeX package names):
    Figure_S1.png -> Fig. S1 maturity distributions
    Figure_2.png -> Fig. 2   paired effects with CI and ROPE
    Figure_S2.png -> Fig. S2 condition-level recorded-band-distance effects
    Figure_3.png -> Fig. 3   response-family variants
    Figure_4.png -> Fig. 4   expanded four-batch effects
    Figure_S3.png -> Fig. S3 gap-vs-control-time separation
Figure_1.png -> Fig. 1   evidence-flow diagram
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

import implementation_sensitivity as impl  # noqa: E402

CANON = ROOT / "outputs" / "canonical" / "03_nominal"
SENS = ROOT / "outputs" / "sensitivity"
EXPANDED = ROOT / "outputs" / "expanded"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
ORIGINAL_FIG = ROOT.parent / "FIGURES_ORIGINAL"  # set by caller if available

COLORS = {"PID": "#0072B2", "OC-PID": "#009E73", "C2": "#E69F00", "C7-Core": "#D55E00"}
MARKERS = {"PID": "o", "OC-PID": "s", "C2": "^", "C7-Core": "D"}


def style_axes(ax) -> None:
    ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig1_evidence_flow() -> None:
    """Render the reader-facing evidence-flow diagram from vector primitives.

    The compact canvas and deliberately large type are sized for a two-column
    manuscript page.  Text is wrapped explicitly so each node retains generous
    internal padding after the figure is reduced to ``\\textwidth``.
    """
    fig, ax = plt.subplots(figsize=(10.6, 7.0), dpi=300)
    ax.set_xlim(0, 1060)
    ax.set_ylim(700, 0)
    ax.axis("off")

    def box(x, y, w, h, face, edge="#30373d", radius=7, lw=1.25):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw,
        )
        ax.add_patch(patch)

    def arrow(x1, y1, x2, y2, color, lw=1.7, scale=18):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=scale, linewidth=lw, color=color,
            shrinkA=0, shrinkB=0,
        ))

    node_text_bounds = []

    def node(x, y, w, h, face, title, body, *, title_y=19, body_y=73):
        """Draw a node with consistent padding and a clear type hierarchy."""
        box(x, y, w, h, face)
        title_text = ax.text(
            x + 15, y + title_y, title,
            fontsize=13.2, fontweight="bold", va="top",
            linespacing=1.18, color="#15191c",
        )
        body_text = ax.text(
            x + 15, y + body_y, body,
            fontsize=12.0, va="top", linespacing=1.20,
            color="#20262b",
        )
        node_text_bounds.append((title, x, y, w, h, title_text, body_text))

    # Evidence-source lane.
    box(20, 18, 1020, 306, "#f4f8fb", "#7fa4bf", 9, 1.35)
    ax.text(
        43, 50, "DATA1-DERIVED OBJECTS AND VERSIONED SETTINGS",
        color="#24587b", fontsize=15.5, fontweight="bold", va="center",
    )
    node(43, 82, 132, 154, "#dbe8f2",
         "Data1\narchive", "13\nproduction\npasses", body_y=76)
    arrow(176, 159, 201, 159, "#3d6786")
    node(205, 82, 211, 154, "#e9eff3",
         "Integrity and\npasswise scale\nextraction",
         "units and timestamps\nrobust scales for\nsix channels", body_y=91)
    arrow(417, 159, 442, 159, "#3d6786")
    node(446, 82, 275, 154, "#dceee8",
         "Data1-derived\nobjects",
         "predictor bank\ninner-PID grid\nrecorded operating band\n13-pass scale catalogue",
         body_y=70)
    node(750, 82, 265, 154, "#e9ecf2",
         "Versioned engineering\nsettings",
         "controller-family constants\namplitude–slew projection\nresponse law",
         body_y=70)

    # Data-derived objects and versioned settings converge on one locked replay.
    merge_x = 584
    ax.plot([584, 584, merge_x], [237, 249, 249], color="#5b7d66", lw=1.45)
    ax.plot([882, 882, merge_x], [237, 249, 249], color="#5b7d66", lw=1.45)
    arrow(merge_x, 249, merge_x, 262, "#5b7d66", 1.45, 15)
    box(380, 258, 408, 50, "#edf4ef", "#5b7d66", 25, 1.15)
    ax.text(
        584, 283, "Fixed objects and settings held\nwithin released replay",
        color="#4f6f59", fontsize=12.0, fontweight="bold", linespacing=1.05,
        ha="center", va="center",
    )
    # End above the next lane so its filled background cannot hide the head.
    arrow(584, 309, 584, 350, "#5b7d66", 1.45, 17)

    # Reserved assessment lane.
    box(20, 354, 1020, 326, "#fffaf5", "#cb9160", 9, 1.35)
    ax.text(
        43, 391, "DATA2 RESERVED RETROSPECTIVE REPLAY AND EVALUATION",
        color="#a85308", fontsize=15.5, fontweight="bold", va="center",
    )
    node(43, 431, 145, 175, "#f6e5d6",
         "Data2\narchive", "10 files\n8 conditions", body_y=84)
    arrow(189, 519, 211, 519, "#95602c")
    node(215, 431, 166, 175, "#f1ece5",
         "Time and\ndistance QA", "row-wise\ntimebase\nstate reset\nat gaps", body_y=87)
    arrow(382, 519, 403, 519, "#95602c")
    node(407, 431, 194, 175, "#f8edcc",
         "Policy replay\nwith common\ninputs",
         "same retained trace\nsame innovation\nsequence", body_y=99)
    arrow(602, 519, 623, 519, "#95602c")
    node(627, 431, 179, 175, "#dbe8f2",
         "Applied\ncommand\nresponse",
         "bounded\nprojection\nversioned law", body_y=99)
    arrow(807, 519, 828, 519, "#95602c")
    node(832, 431, 183, 175, "#dfeee6",
         "Metrics and\nlogs",
         "replay error and\nmotion\nband distance and\nprovenance", body_y=84)
    ax.text(
        530, 648,
        "Archived-policy comparison only; no physical closed-loop or safety claim.",
        color="#4a5055", fontsize=12.0, ha="center", va="center",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Validate actual font-rendered extents, not estimated character counts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for label, x, y, w, h, title_text, body_text in node_text_bounds:
        title_bound = title_text.get_window_extent(renderer).transformed(ax.transData.inverted())
        body_bound = body_text.get_window_extent(renderer).transformed(ax.transData.inverted())
        for bound in (title_bound, body_bound):
            if bound.x0 < x + 10 or bound.x1 > x + w - 10 or max(bound.y0, bound.y1) > y + h - 8:
                issues.append(f"Text escapes node {label!r}: {bound.bounds}")
        if max(title_bound.y0, title_bound.y1) + 10 > min(body_bound.y0, body_bound.y1):
            issues.append(f"Title/body gap too small in node {label!r}")
    if issues:
        raise ValueError("\n".join(issues))
    fig.savefig(FIG_DIR / "Figure_1.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure_1.svg", facecolor="white")
    plt.close(fig)


def fig_s1_maturity() -> None:
    src = ROOT / "outputs" / "canonical" / "10_appendix_audit" / "10_maturity_distributions.png"
    if src.exists():
        shutil.copy2(src, FIG_DIR / "Figure_S1.png")

def fig2_effects() -> None:
    effects = pd.read_csv(RESULTS / "paired_effect_bootstrap.csv")
    metric_map = {
        "composite_normalized_RMS": r"$\mathrm{RMS}_c$",
        "TV_L_per_100m": "TV/100 m",
        "S_excess_mean": "Mean two-sided\nband distance",
        "S_excess_cvar95": "Upper-tail\nband distance",
    }
    rope_map = {"composite_normalized_RMS": 0.5, "TV_L_per_100m": 5.0, "S_excess_mean": 5.0, "S_excess_cvar95": 5.0}
    baselines = ["PID", "OC-PID", "C2"]
    panel_metrics = [["composite_normalized_RMS"], list(metric_map)[1:]]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8),
                             gridspec_kw={"height_ratios": [1, 3]})
    for panel, (ax, metrics) in enumerate(zip(axes, panel_metrics)):
        order = [(metric, f"C7 vs {baseline}")
                 for metric in metrics for baseline in baselines]
        y = np.arange(len(order))[::-1]
        labels = []
        for i, (metric, comparison) in enumerate(order):
            selected = effects[effects.metric.eq(metric) & effects.comparison.eq(comparison)]
            if len(selected) != 1:
                raise ValueError(f"Expected one published effect for {(metric, comparison)!r}")
            row = selected.iloc[0]
            lower = row.point_pct < 0
            color = "#D55E00" if lower else "#0072B2"
            marks = ax.errorbar(
                row.point_pct, y[i],
                xerr=[[row.point_pct - row.ci_low_pct], [row.ci_high_pct - row.point_pct]],
                fmt="o" if lower else "s", color=color, ms=5.5, capsize=3, lw=1.3,
            )
            marks.lines[0].set_gid(f"point:{metric}:{comparison}")
            marks.lines[2][0].set_gid(f"ci:{metric}:{comparison}")
            rope = rope_map[metric]
            band = ax.fill_betweenx([y[i] - 0.38, y[i] + 0.38], -rope, rope,
                                    color="#D9DDE3", zorder=0)
            band.set_gid(f"rope:{metric}:{comparison}")
            baseline = comparison.removeprefix("C7 vs ")
            labels.append(baseline if panel == 0 else f"{metric_map[metric]} ({baseline})")
        ax.set_yticks(y, labels)
        ax.set_ylim(-0.65, len(order) - 0.35)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.axvline(0, color="#333333", lw=0.9)
        style_axes(ax)
    # The focused RMS axis is independent of the other metrics' percent axis.
    axes[0].set_xlim(-0.75, 2.75)
    axes[0].set_xticks(np.arange(-0.5, 2.6, 0.5))
    axes[0].set_title(r"(a) $\mathrm{RMS}_c$  |  row-wise ROPE: $\pm$0.5%",
                      loc="left", fontsize=11.0, pad=10)
    axes[0].set_xlabel("Paired effect (%)", fontsize=10.0)
    axes[1].set_xlim(-38, 24)
    axes[1].set_xticks(np.arange(-30, 21, 10))
    axes[1].set_title("(b) TV and recorded-band distance  |  row-wise ROPE: ±5%",
                      loc="left", fontsize=11.0, pad=10)
    axes[1].set_xlabel("C7-Core paired effect relative to baseline (%)", fontsize=10.0)
    fig.suptitle("Paired effects and 95% condition-hierarchical bootstrap CIs",
                 fontsize=12.0, y=0.975)
    fig.text(0.5, 0.932, "Independent horizontal scales in panels (a) and (b)",
             ha="center", fontsize=10.0, color="#30373d")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color="#D55E00",
                   label="C7-Core lower than baseline"),
            Line2D([0], [0], marker="s", linestyle="none", color="#0072B2",
                   label="C7-Core higher than baseline"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
        frameon=False, fontsize=9.5,
    )
    fig.subplots_adjust(left=0.335, right=0.97, top=0.865, bottom=0.15, hspace=0.35)
    fig.savefig(FIG_DIR / "Figure_2.png", dpi=600, facecolor="white")
    plt.close(fig)


def fig_s2_conditions() -> None:
    conditions = pd.read_csv(RESULTS / "condition_level_effects.csv")
    forest = conditions[(conditions.metric.eq("S_excess_mean")) & conditions.comparison.isin(["C7 vs PID", "C7 vs OC-PID"])].copy()
    condition_order = sorted(forest.condition.unique())
    code = {c: f"C{i+1}" for i, c in enumerate(condition_order)}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(condition_order))
    for offset, baseline in [(-0.12, "C7 vs PID"), (0.12, "C7 vs OC-PID")]:
        vals = [float(forest[(forest.condition.eq(c)) & (forest.comparison.eq(baseline))].relative_effect_pct.iloc[0]) for c in condition_order]
        is_pid = baseline == "C7 vs PID"
        color = "#0072B2" if is_pid else "#D55E00"
        ax.scatter(vals, y + offset, marker="o" if is_pid else "s",
                   facecolor=color if is_pid else "none", edgecolor=color,
                   linewidth=1.2, label=baseline.replace("C7 vs", "C7-Core vs"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [code[c] for c in condition_order])
    ax.set_xlabel("Condition-level mean two-sided band-distance effect (%)")
    ax.set_ylabel("Thickness condition")
    ax.set_title("Condition-level recorded-band-distance effects")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_S2.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def fig3_response_family() -> None:
    # Rebuild from the published values so the reader-facing terminology
    # cannot inherit an obsolete label from an intermediate analysis figure.
    effects = pd.read_csv(RESULTS / "response_family_effects.csv")
    plot = effects[effects.metric.eq("S_excess_mean")]
    scenarios = ["nominal", "low_response_corner", "high_response_corner", "diagonal_response", "alternative_linear", "alternative_weakened"]
    labels = ["Nominal", "Low response", "High response", "Diagonal", "Linear alt.", "Weakened alt."]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(scenarios))
    for offset, baseline in [(-0.17, "PID"), (0.17, "OC-PID")]:
        vals = [float(plot[(plot.scenario.eq(s)) & plot.comparison.eq(f"C7-Core vs {baseline}")].effect_pct.iloc[0]) for s in scenarios]
        ax.bar(x + offset, vals, width=0.32, label=f"vs {baseline}", color=COLORS[baseline], edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("C7-Core effect on mean band distance (%)")
    ax.set_title("Effect direction across locked response-family variants")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_expanded() -> None:
    expanded = pd.read_csv(RESULTS / "expanded_batch_weighted_absolute.csv").set_index("policy")
    omit = pd.read_csv(RESULTS / "data2_leave_one_batch_out.csv")
    metric_map = {
        "S_excess_mean": ("Mean two-sided band distance", -1.0, True),
        "S_excess_cvar95": ("Upper-tail band distance", -1.0, True),
        "composite_normalized_RMS": (r"$\mathrm{RMS}_c$", 1.0, True),
        "TV_L_per_100m": ("TV/100 m", 1.0, True),
    }
    baselines = ["PID", "OC-PID"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), sharey=False)
    panel_titles = {
        "S_excess_mean": "(a) Mean two-sided\nband distance",
        "S_excess_cvar95": "(b) Upper-tail\nband distance",
        "composite_normalized_RMS": r"(c) $\mathrm{RMS}_c$",
        "TV_L_per_100m": "(d) TV/100 m",
    }
    for ax, (metric, (label, sign, pct)) in zip(axes.flat, metric_map.items()):
        x = np.arange(len(baselines))
        for i, baseline in enumerate(baselines):
            base = float(expanded.loc[baseline, metric])
            c7 = float(expanded.loc["C7", metric])
            effect = 100.0 * (c7 - base) / base
            sub = omit[(omit.metric.eq(metric)) & (omit.comparison.eq(f"C7 vs {baseline}"))]
            lo = float(sub.relative_effect_pct.min()) if len(sub) else effect
            hi = float(sub.relative_effect_pct.max()) if len(sub) else effect
            color = COLORS["C7-Core"] if effect < 0 else COLORS[baseline]
            ax.errorbar(x[i], effect, yerr=[[effect - lo], [hi - effect]], fmt=MARKERS[baseline],
                        color=color, ms=7, capsize=4, lw=1.3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, baselines, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(panel_titles[metric], fontsize=11.5, pad=10)
        ax.set_ylabel("C7-Core effect (%)", fontsize=10.5)
        style_axes(ax)
    fig.suptitle("Expanded four-batch C7-Core effects", fontsize=12.0, y=0.982)
    fig.text(0.5, 0.942,
             "Whiskers: leave-one-target-batch-out ranges (not confidence intervals)",
             ha="center", fontsize=10.5, color="#30373d")
    fig.text(0.5, 0.024, "Independent vertical scales across metrics",
             ha="center", fontsize=10.5, color="#30373d")
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.105, top=0.825,
                        hspace=0.63, wspace=0.30)
    fig.savefig(FIG_DIR / "Figure_4.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure_4.svg", facecolor="white")
    plt.close(fig)


def fig_s3_gaps() -> None:
    qa = pd.read_csv(RESULTS / "time_protocol_numeric_QA.csv")
    # Both panels use the same categorical positions; ties use pass ID order.
    qa_sorted = qa.sort_values(["max_positive_raw_timestamp_gap_s", "pass_id"],
                               ascending=[False, True], kind="stable")
    if qa_sorted.pass_id.duplicated().any():
        raise ValueError("Time-protocol QA must contain one row per pass")
    x = np.arange(len(qa_sorted))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.3), sharex=True)
    raw_bars = axes[0].bar(x, qa_sorted.max_positive_raw_timestamp_gap_s, color="#8DA8CE")
    for bar, pass_id in zip(raw_bars, qa_sorted.pass_id):
        bar.set_gid(f"raw-gap:{pass_id}")
    axes[0].set_yscale("log")
    reference = axes[0].axhline(20, color="#C0504D", lw=1.0, ls="--")
    reference.set_gid("gap-reference-20s")
    axes[0].text(0.99, 20, "20 s", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", color="#A13F3B", fontsize=9.0)
    axes[0].set_title("(a) Maximum positive raw timestamp gap\nretained for audit (log scale)",
                      fontsize=11.0, pad=9)
    axes[0].set_ylabel("Gap (s, log scale)", fontsize=10.0)
    control_bars = axes[1].bar(x, qa_sorted.max_dt_control_s, color="#5B7FA6")
    for bar, pass_id in zip(control_bars, qa_sorted.pass_id):
        bar.set_gid(f"control-interval:{pass_id}")
    axes[1].set_title("(b) Maximum segment-local control interval per pass\n(linear scale)",
                      fontsize=11.0, pad=9)
    axes[1].set_ylabel("Control interval (s)", fontsize=10.0)
    axes[1].set_ylim(0, 10.5)
    for ax in axes:
        ax.set_xticks(x, qa_sorted.pass_id, rotation=90, fontsize=8.5)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_xlim(-0.7, len(qa_sorted) - 0.3)
        style_axes(ax)
    fig.text(0.5, 0.025, "Shared pass order: descending maximum raw timestamp gap",
             ha="center", fontsize=10.0, color="#30373d")
    fig.subplots_adjust(left=0.115, right=0.98, top=0.89, bottom=0.15, hspace=0.90)
    fig.savefig(FIG_DIR / "Figure_S3.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / "Figure_S3.svg", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-fig-dir", type=Path, default=None)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    fig1_evidence_flow()
    fig_s1_maturity()
    fig2_effects()
    fig_s2_conditions()
    fig3_response_family()
    fig4_expanded()
    fig_s3_gaps()
    print("[figures] regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
