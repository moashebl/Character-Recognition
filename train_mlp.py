from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ann.data import get_npz_class_names, load_from_npz, make_labels_contiguous, train_test_split
from src.ann.mlp import MLPClassifier
from src.ann.utils import classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP for character recognition (letters, lowercase, symbols)")
    parser.add_argument("--dataset", type=str, default="character_fonts (with handwritten data).npz")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--target-size", type=int, default=28)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--save-model", type=str, default="models/mlp_az.npz")
    parser.add_argument("--plot-path", type=str, default="outputs/mlp_training_curve.png")
    parser.add_argument("--confusion-path", type=str, default="outputs/mlp_confusion_matrix.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x, y = load_from_npz(args.dataset, target_size=args.target_size, flatten=True, normalize=True)
    class_names = get_npz_class_names(args.dataset)
    y, class_names = make_labels_contiguous(y, class_names)

    if args.max_samples > 0 and len(y) > args.max_samples:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(y), size=args.max_samples, replace=False)
        x = x[sample_idx]
        y = y[sample_idx]

    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, seed=42, stratify=True)

    if args.val_size > 0.0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=args.val_size,
            seed=43,
            stratify=True,
        )
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val, y_val = None, None

    input_dim = x_train.shape[1]
    output_dim = len(class_names)
    layer_sizes = [input_dim, *args.hidden, output_dim]

    model = MLPClassifier(
        layer_sizes=layer_sizes,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_activation=args.activation,
        seed=42,
        class_names=class_names,
    )

    history = model.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        early_stopping_patience=args.early_stopping_patience,
        verbose=True,
    )
    metrics = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    print("\nTest metrics:")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    if len(history.val_losses) > 0:
        print(f"best_epoch: {history.best_epoch}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, num_classes=output_dim))

    save_model_path = Path(args.save_model)
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_model_path))
    print(f"\nSaved model to: {save_model_path}")

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.losses, label="train")
    if len(history.val_losses) > 0:
        plt.plot(history.val_losses, label="val")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.accuracies, label="train")
    if len(history.val_accuracies) > 0:
        plt.plot(history.val_accuracies, label="val")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curve to: {plot_path}")

    cm = confusion_matrix(y_test, y_pred, num_classes=output_dim)
    confusion_path = Path(args.confusion_path)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("MLP Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    tick_labels = class_names
    plt.xticks(range(output_dim), tick_labels, rotation=90)
    plt.yticks(range(output_dim), tick_labels)
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=150)
    print(f"Saved confusion matrix to: {confusion_path}")


if __name__ == "__main__":
    main()
