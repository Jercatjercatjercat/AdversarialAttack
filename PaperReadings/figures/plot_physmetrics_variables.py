"""Draw two variable-type comparisons; circle areas do not count channels.

Sources: PhysMetrics arXiv:2606.10642v2 (methods and Appendix A.1),
Otter arXiv:2606.26421v1 (Section 3.1), Aardvark Nature 2025 (main text).
The left set is the six core diagnostic fields, excluding static auxiliaries.
MSLP is deliberately on the right: it can supply an estimate of missing surface
pressure, but it is not the same physical quantity as surface pressure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


DESTINATION = Path(__file__).resolve().parent
INK = "#243746"
BLUE = "#496F8B"
GREEN = "#527D70"


def draw(model, levels, extra, caveat):
    """Place names by set membership; compare gridded weather, not sensor data."""
    fig, ax = plt.subplots(figsize=(12, 8.4), facecolor="#FAFAF7")
    ax.set(xlim=(0, 12), ylim=(-1.6, 6.8), aspect="equal")
    ax.axis("off")
    ax.text(6, 6.35, f"PhysMetrics.Weather and {model}", ha="center",
            fontsize=23, weight="bold", color=INK)
    ax.text(6, 5.94, "Core diagnostic fields versus gridded forecast fields",
            ha="center", fontsize=13, color=INK)

    for center, color in [((4.5, 2.8), BLUE), ((7.5, 2.8), GREEN)]:
        ax.add_patch(Circle(center, 2.7, facecolor=color, alpha=0.17,
                            edgecolor=color, linewidth=2))
        ax.add_patch(Circle(center, 2.7, fill=False, edgecolor=color, linewidth=1.7))

    ax.text(3.65, 4.9, "PhysMetrics", ha="center", color=BLUE,
            fontsize=18, weight="bold")
    ax.text(8.35, 4.9, model, ha="center", color=GREEN,
            fontsize=18, weight="bold")
    ax.text(2.95, 3.55, "Needed, but not\ndirectly predicted", ha="center",
            va="center", fontsize=12, weight="bold", color=BLUE)
    ax.text(2.95, 2.55, "Surface pressure\n(at ground level)", ha="center",
            va="center", fontsize=15, color=INK)
    ax.text(2.95, 1.55, "Can be estimated*", ha="center", fontsize=12,
            color=BLUE, style="italic")

    ax.text(6, 4.05, "Shared fields", ha="center", fontsize=13,
            color=INK, weight="bold")
    ax.text(6, 2.8, "Temperature\nSpecific humidity\nGeopotential\nEast–west wind\nNorth–south wind",
            ha="center", va="center", fontsize=12, color=INK, linespacing=1.9)

    ax.text(9.05, 3.95, "No direct diagnostic\nin the core suite", ha="center",
            va="center", fontsize=12, weight="bold", color=GREEN)
    ax.text(9.05, 2.45, "\n".join(extra), ha="center", va="center",
            fontsize=14, color=INK, linespacing=1.7)
    ax.text(6, 0.32, levels, ha="center", fontsize=12, color=INK,
            bbox=dict(facecolor="#FAFAF7", edgecolor="none", pad=4))

    footer = [
        "* Sea-level pressure + surface geopotential (terrain) can estimate surface pressure (Appendix A.1).",
        "Static terrain fields are excluded from these sets; both models provide surface geopotential.",
        caveat,
        "Overlap means a diagnostic can use the field; it does not certify that field or the whole state as physical.",
    ]
    for y, line in zip([-0.35, -0.68, -1.01, -1.34], footer):
        ax.text(6, y, line, ha="center", va="center", fontsize=10.8, color=INK)
    stem = DESTINATION / f"physmetrics_{model.lower()}_venn"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw("Otter", "Shared upper-air fields: 13 pressure levels in Otter",
         ["Vertical velocity", "2-m temperature", "Sea-level pressure*", "10-m wind (u, v)"],
         "Global conservation checks also need a forecast sequence and vertical integration.")
    draw("Aardvark", "Shared upper-air fields: 200, 500, 700 and 850 hPa in Aardvark",
         ["2-m temperature", "Sea-level pressure*", "10-m wind (u, v)"],
         "Only four vertical levels: full-column budgets need additional assumptions / vertical coverage.")
