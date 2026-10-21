# -*- coding: utf-8 -*-
"""Figure 1: architecture with the actual (reported) computation path in solid lines
and the specified-but-disabled branch dashed. Reviewer-4 item 3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=300)
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 12.6)
ax.axis("off")


def box(x, y, w, h, title, body, fc, ec, dashed=False, title_fs=10, body_fs=8.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       linewidth=1.6, edgecolor=ec, facecolor=fc,
                       linestyle="--" if dashed else "-")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.40, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + (h - 0.55) / 2 + 0.05, body, ha="center", va="center",
            fontsize=body_fs, color="#222222")


# Layer 1 (bottom, full width)
box(0.6, 0.7, 12.2, 2.5,
    "Layer 1  Evidence Extraction (offline, Section 4.2)",
    "CRediT / contribution statements -> declared roles C_apr\n"
    "full text -> method entities K_p;  author histories -> w_ak, pi_ak (Eq. 1)\n"
    "anchors: human-adjudicated set used in reported runs;\n"
    "LLM auto-extraction specified for scale-up only (Section 5.6)",
    "#fee2e2", "#991b1b")

# Layer 2 (middle, full width)
box(0.6, 4.2, 12.2, 2.9,
    "Layer 2  Contribution-Evidence-Constrained Attribution (core, Section 4.3)",
    "multi-evidence sigmoid scorer (Eqs. 2-3)\n"
    "losses: L_coarse + L_anchor (human-adjudicated) + L_hist (Eqs. 4-6)\n"
    "output: Z_hat_apkr -> noisy-or -> role-aggregated participation Z_hat_apk\n"
    "(verified primary output: participation, RQ1)",
    "#fef9c3", "#854d0e")

# Reported path (top-left, solid)
box(0.6, 8.3, 7.3, 3.4,
    "Reported Configuration (Section 4.5)",
    "corpus aggregation -> profile P6(r|a,k)\n"
    "with declared-role-histogram fallback\n"
    "analytic scoring:  score = w_ak x P6 x c_a  (Eq. 9)\n"
    "Task B: proxy-based ranking evaluation\n"
    "(boundary analysis, Sections 5.5, 6.2)",
    "#dbeafe", "#1e40af")

# Specified but not enabled (top-right, dashed)
box(8.6, 8.3, 4.8, 3.4,
    "Specified, Not Enabled",
    "attribution-controlled propagation\n"
    "-> role-resolved repr. h_a,r (Sec. 4.4)\n"
    "joint training, lambda_3 (Sec. 4.5.4)\n"
    "additive variant (Sec. S2)",
    "#f3f4f6", "#6b7280", dashed=True, title_fs=9.5, body_fs=7.6)

# arrows
ax.add_patch(FancyArrowPatch((6.7, 3.34), (6.7, 4.04), arrowstyle="-|>",
                             mutation_scale=16, linewidth=1.8, color="#444444"))
ax.add_patch(FancyArrowPatch((4.2, 7.24), (4.2, 8.14), arrowstyle="-|>",
                             mutation_scale=16, linewidth=1.8, color="#1e40af"))
ax.add_patch(FancyArrowPatch((10.9, 7.24), (10.9, 8.14), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.5, color="#6b7280",
                             linestyle="--"))

ax.text(6.7, 12.35, "solid = reported computation path    dashed = specified but not enabled",
        ha="center", va="center", fontsize=8.5, color="#444444")

plt.tight_layout()
fig.savefig("fig1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved fig1_architecture.png")
