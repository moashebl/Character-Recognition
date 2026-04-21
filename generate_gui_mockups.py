"""Generate realistic GUI mockup images for the presentation.

Creates 4 mockup images matching the actual ANNGui Tkinter application:
  - gui_main.png         : The full main window (Draw & Predict tab)
  - gui_training.png     : Training Metrics tab with curves
  - gui_probability.png  : Probability View tab with bar chart
  - gui_prediction.png   : After a prediction (letter A shown)

All saved to outputs/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour system matching the Tkinter GUI ───────────────────────────
BG      = "#f4f7fb"       # Window background
CARD    = "#ffffff"       # Card/panel background
DARK    = "#1f2937"       # Dark text
MED     = "#6b7280"       # Medium text
LIGHT   = "#9ca3af"       # Light text / hints
BLUE    = "#1d4ed8"       # Accent blue (buttons)
LINE    = "#e5e7eb"       # Borders
PROG    = "#3b82f6"       # Progress bar fill
TEAL    = "#2a9d8f"       # Bar chart


def _fig(w=14, h=8.5):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def _panel(ax, facecolor=CARD, edgecolor=LINE, lw=1.0, radius=0.01):
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(lw)


def _label(ax, x, y, text, size=9, color=DARK, weight="normal",
           ha="left", va="center", transform=None):
    t = transform or ax.transData
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, transform=t)


def _button(ax, x, y, w, h, label, bgcolor=BLUE, textcolor="white", size=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005",
                          facecolor=bgcolor, edgecolor="none", transform=ax.transData)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, fontsize=size, color=textcolor,
            ha="center", va="center", fontweight="bold", transform=ax.transData)


def _entry_row(ax, label, value, y, x0=0.03, w_entry=0.45):
    ax.text(x0, y, label, fontsize=8.5, color=MED, ha="left", va="center",
            transform=ax.transAxes)
    rect = FancyBboxPatch((x0 + 0.22, y - 0.025), w_entry, 0.05,
                          boxstyle="round,pad=0.005",
                          facecolor="#f9fafb", edgecolor=LINE, linewidth=0.8,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x0 + 0.22 + 0.01, y, value, fontsize=8, color=DARK, ha="left",
            va="center", transform=ax.transAxes)


def _section_title(ax, text, y, x=0.03):
    rect = FancyBboxPatch((x, y - 0.005), 0.94, 0.055,
                          boxstyle="round,pad=0.003",
                          facecolor="#e5e7eb", edgecolor="none",
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + 0.01, y + 0.022, text, fontsize=9, color=DARK, fontweight="bold",
            ha="left", va="center", transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════
# Mockup 1 – Main Window (Draw & Predict tab, idle state)
# ═══════════════════════════════════════════════════════════════════
def make_main():
    fig = _fig()
    gs = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                  top=0.97, bottom=0.03, wspace=0.012,
                  width_ratios=[1, 2.1])

    # ── Left panel ──────────────────────────────────────────────────
    ax_left = fig.add_subplot(gs[0])
    _panel(ax_left)
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
    ax_left.axis("off")

    # Title bar
    ax_left.add_patch(FancyBboxPatch((0, 0.965), 1, 0.035,
                                      boxstyle="square,pad=0",
                                      facecolor="#1e293b", edgecolor="none",
                                      transform=ax_left.transAxes))
    ax_left.text(0.5, 0.982, "ANN Character Recognition Studio",
                 fontsize=10, color="white", fontweight="bold",
                 ha="center", va="center", transform=ax_left.transAxes)

    # Dataset section
    _section_title(ax_left, "  Dataset", 0.9)
    _entry_row(ax_left, "NPZ file", "character_fonts (with handwritten …", 0.872)

    # Hyperparameters
    _section_title(ax_left, "  Hyperparameters", 0.83)
    hp = [("Hidden layers", "256,128"), ("Learning rate", "0.01"),
          ("Epochs", "20"), ("Batch size", "256"),
          ("Max samples", "100000"), ("Validation size", "0.1"),
          ("Early stopping", "5"), ("Activation", "relu")]
    for i, (lbl, val) in enumerate(hp):
        _entry_row(ax_left, lbl, val, 0.8 - i * 0.05)

    # Buttons
    for j, (lbl, col) in enumerate([("Train Model", BLUE),
                                     ("Load Model", "#374151"),
                                     ("Predict Image", "#374151")]):
        ax_left.add_patch(FancyBboxPatch((0.03 + j * 0.323, 0.38), 0.3, 0.048,
                                          boxstyle="round,pad=0.005",
                                          facecolor=col, edgecolor="none",
                                          transform=ax_left.transAxes))
        ax_left.text(0.03 + j * 0.323 + 0.15, 0.404, lbl, fontsize=8,
                     color="white", fontweight="bold",
                     ha="center", va="center", transform=ax_left.transAxes)

    # Progress bar
    ax_left.text(0.03, 0.365, "Training progress", fontsize=8, color=MED,
                 transform=ax_left.transAxes)
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor="#e5e7eb", edgecolor=LINE,
                                      linewidth=0.5, transform=ax_left.transAxes))
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.0, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor=PROG, edgecolor="none",
                                      transform=ax_left.transAxes))

    # Status log
    _section_title(ax_left, "  Status", 0.32)
    status_lines = [
        "ANN Character Recognition Studio ready",
        "Dataset: character_fonts (with handwritten data).npz",
        "Architecture: 784 → 256 → 128 → 26 (ReLU)",
        "Train the model to start …",
    ]
    for i, line in enumerate(status_lines):
        ax_left.text(0.03, 0.295 - i * 0.032, line, fontsize=7.5,
                     color=DARK if i < 2 else LIGHT, family="monospace",
                     transform=ax_left.transAxes)

    # ── Right panel ─────────────────────────────────────────────────
    ax_right = fig.add_subplot(gs[1])
    _panel(ax_right, facecolor=BG)
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    ax_right.axis("off")

    # Notebook tabs
    for ti, (tab, selected) in enumerate([("Draw and Predict", True),
                                            ("Training Metrics", False),
                                            ("Probability View", False)]):
        tc = CARD if selected else BG
        ax_right.add_patch(FancyBboxPatch((0.01 + ti * 0.25, 0.935), 0.24, 0.04,
                                           boxstyle="round,pad=0.003",
                                           facecolor=tc, edgecolor=LINE,
                                           linewidth=0.8,
                                           transform=ax_right.transAxes))
        ax_right.text(0.01 + ti * 0.25 + 0.12, 0.956, tab, fontsize=8.5,
                      color=BLUE if selected else MED,
                      fontweight="bold" if selected else "normal",
                      ha="center", va="center", transform=ax_right.transAxes)

    # Drawing pad
    ax_right.add_patch(FancyBboxPatch((0.01, 0.12), 0.59, 0.80,
                                       boxstyle="round,pad=0.005",
                                       facecolor=CARD, edgecolor=LINE,
                                       linewidth=0.8, transform=ax_right.transAxes))
    ax_right.text(0.015, 0.905, "Drawing Pad", fontsize=9, fontweight="bold",
                  color=DARK, transform=ax_right.transAxes)
    # White canvas
    ax_right.add_patch(FancyBboxPatch((0.025, 0.18), 0.56, 0.69,
                                       boxstyle="square,pad=0",
                                       facecolor="white", edgecolor="#9ca3af",
                                       linewidth=1.0, transform=ax_right.transAxes))
    ax_right.text(0.305, 0.525, "Draw here", fontsize=14, color="#d1d5db",
                  ha="center", va="center", transform=ax_right.transAxes,
                  style="italic")

    # Brush/controls row
    ax_right.text(0.03, 0.155, "Brush size", fontsize=7.5, color=MED,
                  transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.16, 0.148), 0.25, 0.014,
                                       boxstyle="round,pad=0.002",
                                       facecolor="#e5e7eb", edgecolor=LINE,
                                       linewidth=0.5, transform=ax_right.transAxes))
    ax_right.add_patch(FancyBboxPatch((0.16, 0.148), 0.08, 0.014,
                                       boxstyle="round,pad=0.002",
                                       facecolor=BLUE, edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.add_patch(FancyBboxPatch((0.44, 0.142), 0.09, 0.026,
                                       boxstyle="round,pad=0.003",
                                       facecolor="#374151", edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.text(0.485, 0.155, "Clear", fontsize=7.5, color="white",
                  fontweight="bold", ha="center", va="center",
                  transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.03, 0.125), 0.565, 0.026,
                                       boxstyle="round,pad=0.003",
                                       facecolor=BLUE, edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.text(0.3125, 0.138, "Predict Drawing", fontsize=8.5, color="white",
                  fontweight="bold", ha="center", va="center",
                  transform=ax_right.transAxes)

    # Prediction details panel
    ax_right.add_patch(FancyBboxPatch((0.62, 0.12), 0.37, 0.80,
                                       boxstyle="round,pad=0.005",
                                       facecolor=CARD, edgecolor=LINE,
                                       linewidth=0.8, transform=ax_right.transAxes))
    ax_right.text(0.625, 0.905, "Prediction Details", fontsize=9,
                  fontweight="bold", color=DARK, transform=ax_right.transAxes)
    ax_right.text(0.635, 0.870, "No prediction yet.\nTrain or load a model\nto start.",
                  fontsize=8.5, color=MED, transform=ax_right.transAxes)
    ax_right.text(0.635, 0.78, "Processed 28×28 input", fontsize=8, color=MED,
                  transform=ax_right.transAxes)
    # Preview box
    ax_right.add_patch(FancyBboxPatch((0.635, 0.60), 0.15, 0.17,
                                       boxstyle="square,pad=0",
                                       facecolor="#e5e7eb", edgecolor=LINE,
                                       linewidth=0.5, transform=ax_right.transAxes))
    ax_right.text(0.71, 0.685, "28×28", fontsize=7, color=LIGHT,
                  ha="center", va="center", transform=ax_right.transAxes)
    ax_right.text(0.635, 0.575, "Top predictions", fontsize=8, color=MED,
                  transform=ax_right.transAxes)
    # Top predictions table header
    ax_right.add_patch(FancyBboxPatch((0.635, 0.545), 0.345, 0.026,
                                       boxstyle="square,pad=0",
                                       facecolor="#f3f4f6", edgecolor=LINE,
                                       linewidth=0.5, transform=ax_right.transAxes))
    ax_right.text(0.64, 0.558, "Class", fontsize=7.5, color=DARK,
                  fontweight="bold", transform=ax_right.transAxes)
    ax_right.text(0.93, 0.558, "Prob.", fontsize=7.5, color=DARK,
                  fontweight="bold", ha="right", transform=ax_right.transAxes)

    fig.savefig(str(OUT / "gui_main.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved: outputs/gui_main.png")


# ═══════════════════════════════════════════════════════════════════
# Mockup 2 – After prediction (letter A drawn)
# ═══════════════════════════════════════════════════════════════════
def make_prediction():
    fig = _fig()
    gs = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                  top=0.97, bottom=0.03, wspace=0.012, width_ratios=[1, 2.1])

    # ── Left panel (same as main) ───────────────────────────────────
    ax_left = fig.add_subplot(gs[0])
    _panel(ax_left)
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
    ax_left.axis("off")

    ax_left.add_patch(FancyBboxPatch((0, 0.965), 1, 0.035,
                                      boxstyle="square,pad=0",
                                      facecolor="#1e293b", edgecolor="none",
                                      transform=ax_left.transAxes))
    ax_left.text(0.5, 0.982, "ANN Character Recognition Studio",
                 fontsize=10, color="white", fontweight="bold",
                 ha="center", va="center", transform=ax_left.transAxes)
    _section_title(ax_left, "  Dataset", 0.9)
    _entry_row(ax_left, "NPZ file", "character_fonts (with handwritten …", 0.872)
    _section_title(ax_left, "  Hyperparameters", 0.83)
    hp = [("Hidden layers", "256,128"), ("Learning rate", "0.01"),
          ("Epochs", "20"), ("Batch size", "256"),
          ("Max samples", "100000"), ("Validation size", "0.1"),
          ("Early stopping", "5"), ("Activation", "relu")]
    for i, (lbl, val) in enumerate(hp):
        _entry_row(ax_left, lbl, val, 0.8 - i * 0.05)
    for j, (lbl, col) in enumerate([("Train Model", BLUE),
                                     ("Load Model", "#374151"),
                                     ("Predict Image", "#374151")]):
        ax_left.add_patch(FancyBboxPatch((0.03 + j * 0.323, 0.38), 0.3, 0.048,
                                          boxstyle="round,pad=0.005",
                                          facecolor=col, edgecolor="none",
                                          transform=ax_left.transAxes))
        ax_left.text(0.03 + j * 0.323 + 0.15, 0.404, lbl, fontsize=8,
                     color="white", fontweight="bold",
                     ha="center", va="center", transform=ax_left.transAxes)

    # Full progress bar
    ax_left.text(0.03, 0.365, "Training progress", fontsize=8, color=MED,
                 transform=ax_left.transAxes)
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor="#e5e7eb", edgecolor=LINE,
                                      linewidth=0.5, transform=ax_left.transAxes))
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor=PROG, edgecolor="none",
                                      transform=ax_left.transAxes))

    _section_title(ax_left, "  Status", 0.32)
    status_lines = [
        "Loaded model from models/mlp_az.npz",
        "Estimated dataset polarity: light background",
        "Prediction for Drawing: A (0) | confidence=0.8821",
        "  mode=original (orig=0.8821, inv=0.3104)",
    ]
    for i, line in enumerate(status_lines):
        ax_left.text(0.03, 0.295 - i * 0.032, line, fontsize=7.2,
                     color=DARK if i < 2 else "#1d4ed8",
                     family="monospace", transform=ax_left.transAxes)

    # ── Right panel ─────────────────────────────────────────────────
    ax_right = fig.add_subplot(gs[1])
    _panel(ax_right, facecolor=BG)
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    ax_right.axis("off")

    for ti, (tab, selected) in enumerate([("Draw and Predict", True),
                                            ("Training Metrics", False),
                                            ("Probability View", False)]):
        tc = CARD if selected else BG
        ax_right.add_patch(FancyBboxPatch((0.01 + ti * 0.25, 0.935), 0.24, 0.04,
                                           boxstyle="round,pad=0.003",
                                           facecolor=tc, edgecolor=LINE,
                                           linewidth=0.8,
                                           transform=ax_right.transAxes))
        ax_right.text(0.01 + ti * 0.25 + 0.12, 0.956, tab, fontsize=8.5,
                      color=BLUE if selected else MED,
                      fontweight="bold" if selected else "normal",
                      ha="center", va="center", transform=ax_right.transAxes)

    # Drawing pad with drawn "A"
    ax_right.add_patch(FancyBboxPatch((0.01, 0.12), 0.59, 0.80,
                                       boxstyle="round,pad=0.005",
                                       facecolor=CARD, edgecolor=LINE,
                                       linewidth=0.8, transform=ax_right.transAxes))
    ax_right.text(0.015, 0.905, "Drawing Pad", fontsize=9, fontweight="bold",
                  color=DARK, transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.025, 0.18), 0.56, 0.69,
                                       boxstyle="square,pad=0",
                                       facecolor="white", edgecolor="#9ca3af",
                                       linewidth=1.0, transform=ax_right.transAxes))

    # Draw a big "A" in the canvas area
    cx, cy = 0.305, 0.525
    lw = 18
    ax_right.plot([cx - 0.10, cx], [cy - 0.22, cy + 0.22], color="black",
                  linewidth=lw, solid_capstyle="round", transform=ax_right.transAxes)
    ax_right.plot([cx + 0.10, cx], [cy - 0.22, cy + 0.22], color="black",
                  linewidth=lw, solid_capstyle="round", transform=ax_right.transAxes)
    ax_right.plot([cx - 0.055, cx + 0.055], [cy + 0.02, cy + 0.02], color="black",
                  linewidth=lw, solid_capstyle="round", transform=ax_right.transAxes)

    # Controls
    ax_right.text(0.03, 0.155, "Brush size", fontsize=7.5, color=MED,
                  transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.16, 0.148), 0.25, 0.014,
                                       boxstyle="round,pad=0.002",
                                       facecolor="#e5e7eb", edgecolor=LINE,
                                       linewidth=0.5, transform=ax_right.transAxes))
    ax_right.add_patch(FancyBboxPatch((0.16, 0.148), 0.08, 0.014,
                                       boxstyle="round,pad=0.002",
                                       facecolor=BLUE, edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.add_patch(FancyBboxPatch((0.44, 0.142), 0.09, 0.026,
                                       boxstyle="round,pad=0.003",
                                       facecolor="#374151", edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.text(0.485, 0.155, "Clear", fontsize=7.5, color="white",
                  fontweight="bold", ha="center", va="center",
                  transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.03, 0.125), 0.565, 0.026,
                                       boxstyle="round,pad=0.003",
                                       facecolor=BLUE, edgecolor="none",
                                       transform=ax_right.transAxes))
    ax_right.text(0.3125, 0.138, "Predict Drawing", fontsize=8.5, color="white",
                  fontweight="bold", ha="center", va="center",
                  transform=ax_right.transAxes)

    # Prediction details – WITH result
    ax_right.add_patch(FancyBboxPatch((0.62, 0.12), 0.37, 0.80,
                                       boxstyle="round,pad=0.005",
                                       facecolor=CARD, edgecolor=LINE,
                                       linewidth=0.8, transform=ax_right.transAxes))
    ax_right.text(0.625, 0.905, "Prediction Details", fontsize=9,
                  fontweight="bold", color=DARK, transform=ax_right.transAxes)
    ax_right.text(0.635, 0.870,
                  "Drawing: A (0)\nConfidence: 88.21%\n[original]",
                  fontsize=9.5, color=BLUE, fontweight="bold",
                  transform=ax_right.transAxes)

    ax_right.text(0.635, 0.79, "Processed 28×28 input", fontsize=8, color=MED,
                  transform=ax_right.transAxes)

    # 28x28 preview – render a tiny pixelated "A"
    preview_ax = fig.add_axes([0.635 * 0.72 + 0.005, 0.60 * 0.94 + 0.024,
                                0.10, 0.135])
    pix = np.ones((28, 28))
    for r in range(28):
        for c in range(28):
            if (abs(c - 14) + abs(r - 5) < 12) and (
                r > 5 and (c < 10 or c > 18)):
                pix[r, c] = 0
            if r == 14 and 9 <= c <= 19:
                pix[r, c] = 0
    preview_ax.imshow(pix, cmap="Blues_r", vmin=0, vmax=1)
    preview_ax.axis("off")

    ax_right.text(0.635, 0.575, "Top predictions", fontsize=8, color=MED,
                  transform=ax_right.transAxes)
    ax_right.add_patch(FancyBboxPatch((0.635, 0.545), 0.345, 0.026,
                                       boxstyle="square,pad=0",
                                       facecolor="#f3f4f6", edgecolor=LINE,
                                       linewidth=0.5, transform=ax_right.transAxes))
    ax_right.text(0.64, 0.558, "Class", fontsize=7.5, color=DARK,
                  fontweight="bold", transform=ax_right.transAxes)
    ax_right.text(0.93, 0.558, "Probability", fontsize=7.5, color=DARK,
                  fontweight="bold", ha="right", transform=ax_right.transAxes)
    top_preds = [("1. A", "0.8821"), ("2. H", "0.0431"), ("3. R", "0.0288"),
                 ("4. K", "0.0194"), ("5. Λ", "0.0087")]
    for i, (cls, prob) in enumerate(top_preds):
        bg = "#eff6ff" if i == 0 else "white"
        y_row = 0.515 - i * 0.042
        ax_right.add_patch(FancyBboxPatch((0.635, y_row), 0.345, 0.038,
                                           boxstyle="square,pad=0",
                                           facecolor=bg, edgecolor=LINE,
                                           linewidth=0.3,
                                           transform=ax_right.transAxes))
        ax_right.text(0.64, y_row + 0.019, cls, fontsize=8,
                      color=BLUE if i == 0 else DARK,
                      fontweight="bold" if i == 0 else "normal",
                      transform=ax_right.transAxes)
        ax_right.text(0.93, y_row + 0.019, prob, fontsize=8,
                      color=BLUE if i == 0 else DARK, ha="right",
                      transform=ax_right.transAxes)

    fig.savefig(str(OUT / "gui_prediction.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved: outputs/gui_prediction.png")


# ═══════════════════════════════════════════════════════════════════
# Mockup 3 – Training Metrics tab with curves
# ═══════════════════════════════════════════════════════════════════
def make_training_metrics():
    fig = _fig()
    gs = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                  top=0.97, bottom=0.03, wspace=0.012, width_ratios=[1, 2.1])

    ax_left = fig.add_subplot(gs[0])
    _panel(ax_left)
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
    ax_left.axis("off")

    ax_left.add_patch(FancyBboxPatch((0, 0.965), 1, 0.035, boxstyle="square,pad=0",
                                      facecolor="#1e293b", edgecolor="none",
                                      transform=ax_left.transAxes))
    ax_left.text(0.5, 0.982, "ANN Character Recognition Studio",
                 fontsize=10, color="white", fontweight="bold",
                 ha="center", va="center", transform=ax_left.transAxes)
    _section_title(ax_left, "  Dataset", 0.9)
    _entry_row(ax_left, "NPZ file", "character_fonts (with handwritten …", 0.872)
    _section_title(ax_left, "  Hyperparameters", 0.83)
    hp = [("Hidden layers", "256,128"), ("Learning rate", "0.01"),
          ("Epochs", "20"), ("Batch size", "256"),
          ("Max samples", "100000"), ("Validation size", "0.1"),
          ("Early stopping", "5"), ("Activation", "relu")]
    for i, (lbl, val) in enumerate(hp):
        _entry_row(ax_left, lbl, val, 0.8 - i * 0.05)
    for j, (lbl, col) in enumerate([("Train Model", BLUE),
                                     ("Load Model", "#374151"),
                                     ("Predict Image", "#374151")]):
        ax_left.add_patch(FancyBboxPatch((0.03 + j * 0.323, 0.38), 0.3, 0.048,
                                          boxstyle="round,pad=0.005",
                                          facecolor=col, edgecolor="none",
                                          transform=ax_left.transAxes))
        ax_left.text(0.03 + j * 0.323 + 0.15, 0.404, lbl, fontsize=8,
                     color="white", fontweight="bold",
                     ha="center", va="center", transform=ax_left.transAxes)

    ax_left.text(0.03, 0.365, "Training progress", fontsize=8, color=MED,
                 transform=ax_left.transAxes)
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor="#e5e7eb", edgecolor=LINE,
                                      linewidth=0.5, transform=ax_left.transAxes))
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor=PROG, edgecolor="none",
                                      transform=ax_left.transAxes))
    _section_title(ax_left, "  Status", 0.32)
    log_lines = [
        "Train=86400  Val=9600  Test=24000",
        "Epoch 1/20 | loss=2.4301 | acc=0.1832",
        "Epoch 5/20 | loss=1.2104 | acc=0.6371",
        "Epoch 10/20| loss=0.9854 | acc=0.7122",
        "Epoch 15/20| loss=0.9011 | acc=0.7680",
        "Training complete. Test accuracy=0.7815",
        "Best validation epoch: 15",
        "Model saved to models/gui_mlp_model.npz",
    ]
    for i, line in enumerate(log_lines):
        ax_left.text(0.03, 0.295 - i * 0.032, line, fontsize=7.0,
                     color=DARK if i < 5 else "#16a34a",
                     family="monospace", transform=ax_left.transAxes)

    # ── Right panel: Training Metrics tab ───────────────────────────
    ax_right = fig.add_subplot(gs[1])
    _panel(ax_right, facecolor=BG)
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    ax_right.axis("off")

    for ti, (tab, selected) in enumerate([("Draw and Predict", False),
                                            ("Training Metrics", True),
                                            ("Probability View", False)]):
        tc = CARD if selected else BG
        ax_right.add_patch(FancyBboxPatch((0.01 + ti * 0.25, 0.935), 0.24, 0.04,
                                           boxstyle="round,pad=0.003",
                                           facecolor=tc, edgecolor=LINE,
                                           linewidth=0.8,
                                           transform=ax_right.transAxes))
        ax_right.text(0.01 + ti * 0.25 + 0.12, 0.956, tab, fontsize=8.5,
                      color=BLUE if selected else MED,
                      fontweight="bold" if selected else "normal",
                      ha="center", va="center", transform=ax_right.transAxes)

    # The main content: two matplotlib sub-axes for loss and accuracy
    epochs = np.arange(1, 21)
    losses     = 2.5 * np.exp(-0.22 * epochs) + 0.35 + np.random.default_rng(1).normal(0, 0.03, 20)
    val_losses = 2.5 * np.exp(-0.18 * epochs) + 0.55 + np.random.default_rng(2).normal(0, 0.04, 20)
    accs       = 0.10 + 0.69 * (1 - np.exp(-0.22 * epochs)) + np.random.default_rng(3).normal(0, 0.01, 20)
    val_accs   = 0.08 + 0.63 * (1 - np.exp(-0.18 * epochs)) + np.random.default_rng(4).normal(0, 0.012, 20)

    ax_loss = fig.add_axes([0.355, 0.12, 0.30, 0.78])
    ax_loss.set_facecolor(CARD)
    ax_loss.plot(epochs, losses,     color="#2563eb", linewidth=2.2, label="train")
    ax_loss.plot(epochs, val_losses, color="#f97316", linewidth=2.2, label="val", linestyle="--")
    ax_loss.set_title("Training Loss", fontsize=11, color=DARK, pad=6)
    ax_loss.set_xlabel("Epoch", fontsize=9, color=MED)
    ax_loss.set_ylabel("Loss",  fontsize=9, color=MED)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.2)
    ax_loss.tick_params(labelsize=8, colors=MED)
    for sp in ax_loss.spines.values():
        sp.set_edgecolor(LINE)

    ax_acc = fig.add_axes([0.695, 0.12, 0.30, 0.78])
    ax_acc.set_facecolor(CARD)
    ax_acc.plot(epochs, accs,     color="#16a34a", linewidth=2.2, label="train")
    ax_acc.plot(epochs, val_accs, color="#b45309", linewidth=2.2, label="val", linestyle="--")
    ax_acc.set_title("Training Accuracy", fontsize=11, color=DARK, pad=6)
    ax_acc.set_xlabel("Epoch", fontsize=9, color=MED)
    ax_acc.set_ylabel("Accuracy",  fontsize=9, color=MED)
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(alpha=0.2)
    ax_acc.tick_params(labelsize=8, colors=MED)
    for sp in ax_acc.spines.values():
        sp.set_edgecolor(LINE)

    fig.savefig(str(OUT / "gui_training_metrics.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved: outputs/gui_training_metrics.png")


# ═══════════════════════════════════════════════════════════════════
# Mockup 4 – Probability View tab
# ═══════════════════════════════════════════════════════════════════
def make_probability():
    fig = _fig()
    gs = GridSpec(1, 2, figure=fig, left=0.005, right=0.995,
                  top=0.97, bottom=0.03, wspace=0.012, width_ratios=[1, 2.1])

    ax_left = fig.add_subplot(gs[0])
    _panel(ax_left)
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
    ax_left.axis("off")
    ax_left.add_patch(FancyBboxPatch((0, 0.965), 1, 0.035, boxstyle="square,pad=0",
                                      facecolor="#1e293b", edgecolor="none",
                                      transform=ax_left.transAxes))
    ax_left.text(0.5, 0.982, "ANN Character Recognition Studio",
                 fontsize=10, color="white", fontweight="bold",
                 ha="center", va="center", transform=ax_left.transAxes)
    _section_title(ax_left, "  Dataset", 0.9)
    _entry_row(ax_left, "NPZ file", "character_fonts (with handwritten …", 0.872)
    _section_title(ax_left, "  Hyperparameters", 0.83)
    hp = [("Hidden layers", "256,128"), ("Learning rate", "0.01"),
          ("Epochs", "20"), ("Batch size", "256"),
          ("Max samples", "100000"), ("Validation size", "0.1"),
          ("Early stopping", "5"), ("Activation", "relu")]
    for i, (lbl, val) in enumerate(hp):
        _entry_row(ax_left, lbl, val, 0.8 - i * 0.05)
    for j, (lbl, col) in enumerate([("Train Model", BLUE),
                                     ("Load Model", "#374151"),
                                     ("Predict Image", "#374151")]):
        ax_left.add_patch(FancyBboxPatch((0.03 + j * 0.323, 0.38), 0.3, 0.048,
                                          boxstyle="round,pad=0.005",
                                          facecolor=col, edgecolor="none",
                                          transform=ax_left.transAxes))
        ax_left.text(0.03 + j * 0.323 + 0.15, 0.404, lbl, fontsize=8,
                     color="white", fontweight="bold",
                     ha="center", va="center", transform=ax_left.transAxes)
    ax_left.text(0.03, 0.365, "Training progress", fontsize=8, color=MED,
                 transform=ax_left.transAxes)
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor="#e5e7eb", edgecolor=LINE,
                                      linewidth=0.5, transform=ax_left.transAxes))
    ax_left.add_patch(FancyBboxPatch((0.03, 0.345), 0.94, 0.018,
                                      boxstyle="round,pad=0.003",
                                      facecolor=PROG, edgecolor="none",
                                      transform=ax_left.transAxes))
    _section_title(ax_left, "  Status", 0.32)
    log_lines = [
        "Model loaded from models/mlp_az.npz",
        "Prediction for Drawing: A (0)",
        "  confidence=0.8821 | mode=original",
        "  (orig=0.8821, inv=0.3104)",
    ]
    for i, line in enumerate(log_lines):
        ax_left.text(0.03, 0.295 - i * 0.032, line, fontsize=7.2,
                     color=DARK if i == 0 else "#1d4ed8",
                     family="monospace", transform=ax_left.transAxes)

    # ── Right panel: Probability View tab ───────────────────────────
    ax_right = fig.add_subplot(gs[1])
    _panel(ax_right, facecolor=BG)
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    ax_right.axis("off")
    for ti, (tab, selected) in enumerate([("Draw and Predict", False),
                                            ("Training Metrics", False),
                                            ("Probability View", True)]):
        tc = CARD if selected else BG
        ax_right.add_patch(FancyBboxPatch((0.01 + ti * 0.25, 0.935), 0.24, 0.04,
                                           boxstyle="round,pad=0.003",
                                           facecolor=tc, edgecolor=LINE,
                                           linewidth=0.8,
                                           transform=ax_right.transAxes))
        ax_right.text(0.01 + ti * 0.25 + 0.12, 0.956, tab, fontsize=8.5,
                      color=BLUE if selected else MED,
                      fontweight="bold" if selected else "normal",
                      ha="center", va="center", transform=ax_right.transAxes)

    # Probability bar chart as actual matplotlib axes
    top_labels = ["A", "H", "R", "K", "N", "M", "Λ", "X", "V", "Υ", "E", "T"]
    top_probs  = [0.882, 0.043, 0.029, 0.019, 0.008, 0.006,
                  0.004, 0.003, 0.002, 0.002, 0.001, 0.001]

    ax_prob = fig.add_axes([0.355, 0.12, 0.635, 0.78])
    ax_prob.set_facecolor(CARD)
    colors = [TEAL if i > 0 else "#1d4ed8" for i in range(len(top_labels))]
    bars = ax_prob.barh(np.arange(len(top_labels)), top_probs,
                        color=colors, height=0.65)
    ax_prob.set_yticks(np.arange(len(top_labels)))
    ax_prob.set_yticklabels(top_labels, fontsize=10)
    ax_prob.invert_yaxis()
    ax_prob.set_title("Top Class Probabilities", fontsize=12, color=DARK, pad=8)
    ax_prob.set_xlabel("Probability", fontsize=10, color=MED)
    ax_prob.set_xlim(0, 1.05)
    ax_prob.grid(axis="x", linestyle="--", alpha=0.3)
    ax_prob.tick_params(colors=MED, labelsize=9)
    for sp in ax_prob.spines.values():
        sp.set_edgecolor(LINE)
    for bar, prob in zip(bars, top_probs):
        ax_prob.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=9, color=DARK)

    fig.savefig(str(OUT / "gui_probability.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved: outputs/gui_probability.png")


# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_main()
    make_prediction()
    make_training_metrics()
    make_probability()
    print("\nAll 4 GUI mockup images generated successfully!")
