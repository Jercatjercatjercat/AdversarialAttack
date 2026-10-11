"""Save figures without needing a graphical desktop. All plotted fields are 2D."""
import os
from pathlib import Path
# A local writable cache also works in restricted notebook/agent environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data.era5 import denormalize


def field_panels(fields, titles, path, differences=(), units="°C"):
    """List of [H,W] arrays -> PNG. Shared temperature scale; signed difference scales."""
    arrays = [np.asarray(f) for f in fields]
    temps = [a for i, a in enumerate(arrays) if i not in differences]
    low = min(a.min() for a in temps) if temps else 0
    high = max(a.max() for a in temps) if temps else 1
    fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 3.5), squeeze=False)
    for i, (a, title, ax) in enumerate(zip(arrays, titles, axes[0])):
        if i in differences:
            limit = max(float(abs(a).max()), 1e-8)
            picture = ax.imshow(a, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            label = "K difference" if units == "°C" else units
        else:
            picture = ax.imshow(a, origin="lower", cmap="coolwarm", vmin=low, vmax=high)
            label = units
        ax.set_title(title)
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")
        fig.colorbar(picture, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def forecast_plot(x, y, prediction, stats, path):
    """Three normalized [1,H,W] tensors -> input/target/prediction/error in physical units."""
    fields = [denormalize(t.detach().cpu().numpy()[0], stats, celsius=True)
              for t in (x, y, prediction)]
    field_panels(fields + [fields[2] - fields[1]],
                 ["Input", "True future", "Predicted future", "Prediction − true future"],
                 path, differences=(3,))


def training_plot(history, path):
    """Epoch records -> separate training MSE and validation RMSE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_mse"] for r in history])
    axes[0].set_ylabel("Training MSE (normalized)")
    axes[1].plot(epochs, [r["validation_rmse_k"] for r in history])
    axes[1].set_ylabel("Validation RMSE (K)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def comparison_plot(rows, path):
    """Comparison rows -> one error-versus-budget panel per norm.

    Average random repetitions; shading shows their standard deviation, not a
    confidence interval. Each norm has its own allowed set of input changes.
    """
    norms = list(dict.fromkeys(row["norm"] for row in rows))
    fig, axes = plt.subplots(1, len(norms), figsize=(6 * len(norms), 4), squeeze=False)
    for ax, norm in zip(axes[0], norms):
        subset = [r for r in rows if r["norm"] == norm]
        for method in dict.fromkeys(r["method"] for r in subset):
            budgets = sorted({r["budget_k"] for r in subset if r["method"] == method})
            means, deviations = [], []
            for budget in budgets:
                errors = [r["changed_rmse_k"] for r in subset
                          if r["method"] == method and r["budget_k"] == budget]
                means.append(np.mean(errors))
                deviations.append(np.std(errors))
            line, = ax.plot(budgets, means, marker="o", label=method)
            ax.fill_between(budgets, np.array(means) - deviations,
                            np.array(means) + deviations, alpha=.15, color=line.get_color())
        ax.axhline(subset[0]["clean_rmse_k"], color="black", linestyle="--", label="clean CNN")
        ax.axhline(subset[0]["persistence_rmse_k"], color="gray", linestyle=":", label="predict no change")
        ax.set_title("L∞: limit at each cell" if norm == "linf" else "L2: shared map budget")
        ax.set_xlabel("Maximum cell change (K)" if norm == "linf" else "RMS input-change bound (K)")
        ax.set_ylabel("Forecast RMSE (K)")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
