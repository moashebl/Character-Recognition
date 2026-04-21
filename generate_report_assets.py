"""Generate report assets (figures) into the outputs/ folder.

This script is intentionally self-contained and uses only project dependencies.
It creates:
- Logic gate plots (AND/OR/XOR) using this repo's MLP implementation
- Neural-network explanatory diagrams (architecture + training flow)

Run:
    c:/python313/python.exe generate_report_assets.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ann.mlp import MLPClassifier


OUT_DIR = Path("outputs")


@dataclass(frozen=True)
class GateSpec:
    name: str
    y: np.ndarray


def _train_gate_model(x: np.ndarray, y: np.ndarray, seed: int) -> MLPClassifier:
    # Binary classification is represented as 2-class softmax (labels 0/1).
    model = MLPClassifier(
        layer_sizes=[2, 6, 2],
        learning_rate=0.15,
        epochs=2500,
        batch_size=4,
        hidden_activation="sigmoid",
        seed=seed,
        class_names=["0", "1"],
    )
    model.fit(x, y, verbose=False)
    return model


def _plot_gate(name: str, x: np.ndarray, y: np.ndarray, out_path: Path, seed: int) -> None:
    model = _train_gate_model(x, y, seed=seed)

    grid_x0 = np.linspace(-0.25, 1.25, 240)
    grid_x1 = np.linspace(-0.25, 1.25, 240)
    xx, yy = np.meshgrid(grid_x0, grid_x1)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = plt.figure(figsize=(10, 4.3), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.4, 1.0))

    ax = fig.add_subplot(gs[0, 0])
    cf = ax.contourf(xx, yy, proba, levels=20, cmap="viridis", alpha=0.9)
    fig.colorbar(cf, ax=ax, shrink=0.95, label="P(output=1)")

    colors = np.where(y == 1, "#ef4444", "#2563eb")
    ax.scatter(x[:, 0], x[:, 1], c=colors, s=120, edgecolors="black", linewidths=0.8, zorder=3)

    for (x0, x1), label in zip(x, y):
        ax.text(float(x0) + 0.03, float(x1) + 0.03, str(int(label)), fontsize=10, weight="bold")

    ax.set_title(f"Logic Gate: {name} (MLP output surface)")
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linestyle="--")

    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")
    ax_tbl.set_title("Truth Table", pad=10)

    # Build a small table-like text block (keeps dependencies minimal).
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    header = "A  B  Out"
    lines = [header, "-" * len(header)]
    for i, (a, b) in enumerate(rows):
        lines.append(f"{a:<2} {b:<2} {int(y[i]):<3}")

    ax_tbl.text(0.05, 0.85, "\n".join(lines), family="monospace", fontsize=12, va="top")

    # Show the model's final predictions for the 4 inputs.
    preds = model.predict(x)
    ok = (preds == y).all()
    ax_tbl.text(0.05, 0.35, f"MLP predictions: {preds.tolist()}\nCorrect: {bool(ok)}", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_logic_gate_images() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    gates = [
        GateSpec("AND", np.array([0, 0, 0, 1], dtype=np.int64)),
        GateSpec("OR", np.array([0, 1, 1, 1], dtype=np.int64)),
        GateSpec("XOR", np.array([0, 1, 1, 0], dtype=np.int64)),
    ]

    outputs: list[Path] = []
    for idx, gate in enumerate(gates, start=1):
        out_path = OUT_DIR / f"logic_gate_{gate.name.lower()}.png"
        _plot_gate(gate.name, x, gate.y, out_path, seed=100 + idx)
        outputs.append(out_path)

    return outputs


def _draw_layer(ax: plt.Axes, x: float, y0: float, n_dots: int, label: str) -> None:
    ys = np.linspace(y0, y0 + 1.0, n_dots)
    for y in ys:
        circ = plt.Circle((x, y), 0.04, color="#111827", fill=False, linewidth=1.2)
        ax.add_patch(circ)
    ax.text(x, y0 - 0.12, label, ha="center", va="top", fontsize=10)


def generate_neural_network_diagrams() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    # 1) Architecture diagram
    fig, ax = plt.subplots(figsize=(10.5, 3.7), dpi=150)
    ax.axis("off")

    x_positions = [0.1, 0.38, 0.62, 0.9]
    y0 = 0.2

    _draw_layer(ax, x_positions[0], y0, n_dots=7, label="Input\n(28×28 = 784)")
    _draw_layer(ax, x_positions[1], y0, n_dots=6, label="Hidden 1\n(256)")
    _draw_layer(ax, x_positions[2], y0, n_dots=6, label="Hidden 2\n(128)")
    _draw_layer(ax, x_positions[3], y0, n_dots=7, label="Output\n(26 classes)")

    # Connect layers with simple arrows.
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.06, 0.7),
            xytext=(x0 + 0.06, 0.7),
            arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#111827"),
        )

    ax.text(0.5, 1.06, "MLP Architecture (Conceptual)", ha="center", va="bottom", fontsize=14, weight="bold")

    arch_path = OUT_DIR / "neural_network_architecture.png"
    fig.tight_layout()
    fig.savefig(arch_path)
    plt.close(fig)
    out_paths.append(arch_path)

    # 2) Training / backprop flow diagram
    fig, ax = plt.subplots(figsize=(11.5, 3.2), dpi=150)
    ax.axis("off")

    boxes = [
        "Input image\n(28×28 grayscale)",
        "Preprocess\nnormalize + flatten", 
        "Forward pass\n(Wx + b + activation)",
        "Softmax\nprobabilities", 
        "Loss\n(cross-entropy)",
        "Backprop\n(gradients)",
        "Update\n(W := W − η∇W)",
    ]

    xs = np.linspace(0.04, 0.96, len(boxes))
    y = 0.55

    for i, (x, text) in enumerate(zip(xs, boxes)):
        ax.add_patch(
            plt.Rectangle(
                (x - 0.06, y - 0.16),
                0.12,
                0.32,
                fill=False,
                linewidth=1.5,
                edgecolor="#111827",
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.075, y),
                xytext=(x + 0.075, y),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#111827"),
            )

    ax.text(0.5, 0.95, "Training Loop (One Iteration)", ha="center", va="center", fontsize=14, weight="bold")

    flow_path = OUT_DIR / "neural_network_training_flow.png"
    fig.tight_layout()
    fig.savefig(flow_path)
    plt.close(fig)
    out_paths.append(flow_path)

    return out_paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.extend(generate_logic_gate_images())
    generated.extend(generate_neural_network_diagrams())

    print("Generated assets:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
