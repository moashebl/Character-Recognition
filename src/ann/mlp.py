from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .utils import one_hot_encode


@dataclass
class MLPHistory:
    losses: list[float]
    accuracies: list[float]
    val_losses: list[float]
    val_accuracies: list[float]
    best_epoch: int


class MLPClassifier:
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        epochs: int = 20,
        batch_size: int = 128,
        hidden_activation: str = "relu",
        seed: int = 42,
        class_names: list[str] | None = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output dimensions")

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.class_names = class_names

        self.rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if hidden_activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            w = self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def _hidden_activation(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x)) # This is the Sigmoid Function
        if self.hidden_activation == "relu":
            return np.maximum(0.0, x) # this is the Relu linear function
        raise ValueError("hidden_activation must be 'relu' or 'sigmoid'")

    def _hidden_activation_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-x))
            return sig * (1.0 - sig)
        if self.hidden_activation == "relu":
            return (x > 0.0).astype(np.float32)
        raise ValueError("hidden_activation must be 'relu' or 'sigmoid'")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_stable)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _cross_entropy(y_true_one_hot: np.ndarray, y_pred_probs: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred_probs, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1)))

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        pre_activations = []

        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._hidden_activation(z)
            pre_activations.append(z)
            activations.append(a)

        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        y_hat = self._softmax(z_out)
        pre_activations.append(z_out)
        activations.append(y_hat)

        return activations, pre_activations

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y_true_one_hot: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = y_true_one_hot.shape[0]

        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y_true_one_hot
        grad_w[-1] = np.dot(activations[-2].T, delta) / m
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        for layer in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[layer + 1].T) * self._hidden_activation_derivative(pre_activations[layer])
            grad_w[layer] = np.dot(activations[layer].T, delta) / m
            grad_b[layer] = np.sum(delta, axis=0, keepdims=True) / m

        return grad_w, grad_b

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_patience: int | None = None,
        verbose: bool = True,
        epoch_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> MLPHistory:
        history = MLPHistory(losses=[], accuracies=[], val_losses=[], val_accuracies=[], best_epoch=1)
        y_one_hot = one_hot_encode(y, self.layer_sizes[-1])
        best_val_loss = float("inf")
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            indices = np.arange(x.shape[0])
            self.rng.shuffle(indices)

            x_shuffled = x[indices]
            y_one_hot_shuffled = y_one_hot[indices]
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for start in range(0, x.shape[0], self.batch_size):
                end = min(start + self.batch_size, x.shape[0])
                x_batch = x_shuffled[start:end]
                y_batch = y_one_hot_shuffled[start:end]

                activations, pre_activations = self.forward(x_batch)
                grad_w, grad_b = self._backward(activations, pre_activations, y_batch)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grad_w[i]
                    self.biases[i] -= self.learning_rate * grad_b[i]

                batch_probs = activations[-1]
                batch_size = x_batch.shape[0]
                epoch_loss_sum += self._cross_entropy(y_batch, batch_probs) * batch_size
                batch_preds = np.argmax(batch_probs, axis=1)
                batch_true = np.argmax(y_batch, axis=1)
                epoch_correct += int(np.sum(batch_preds == batch_true))
                epoch_samples += batch_size

            loss = epoch_loss_sum / max(1, epoch_samples)
            acc = float(epoch_correct / max(1, epoch_samples))

            history.losses.append(loss)
            history.accuracies.append(acc)

            epoch_metrics: dict[str, float] = {
                "loss": float(loss),
                "accuracy": float(acc),
            }

            if x_val is not None and y_val is not None:
                val_metrics = self.evaluate(x_val, y_val)
                history.val_losses.append(val_metrics["loss"])
                history.val_accuracies.append(val_metrics["accuracy"])
                epoch_metrics["val_loss"] = float(val_metrics["loss"])
                epoch_metrics["val_accuracy"] = float(val_metrics["accuracy"])

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    history.best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"loss: {loss:.4f} - accuracy: {acc:.4f} - "
                        f"val_loss: {val_metrics['loss']:.4f} - val_accuracy: {val_metrics['accuracy']:.4f}"
                    )

                if epoch_callback is not None:
                    epoch_callback(epoch + 1, self.epochs, epoch_metrics)

                if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Best validation loss was at epoch {history.best_epoch}."
                        )
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            elif verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")

            if x_val is None or y_val is None:
                if epoch_callback is not None:
                    epoch_callback(epoch + 1, self.epochs, epoch_metrics)

        if x_val is not None and y_val is not None:
            self.weights = best_weights
            self.biases = best_biases

        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(x)
        return activations[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        probs = self.predict_proba(x)
        y_one_hot = one_hot_encode(y, self.layer_sizes[-1])
        loss = self._cross_entropy(y_one_hot, probs)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        return {"loss": loss, "accuracy": acc}

    def save(self, path: str) -> None:
        data = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int64),
            "learning_rate": np.float32(self.learning_rate),
            "epochs": np.int64(self.epochs),
            "batch_size": np.int64(self.batch_size),
            "hidden_activation": np.array([self.hidden_activation]),
            "num_layers": np.int64(len(self.weights)),
        }

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"w_{i}"] = w
            data[f"b_{i}"] = b

        if self.class_names is not None:
            data["class_names"] = np.array(self.class_names)

        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "MLPClassifier":
        data = np.load(path, allow_pickle=False)
        layer_sizes = data["layer_sizes"].astype(np.int64).tolist()
        hidden_activation = str(data["hidden_activation"][0])
        model = cls(
            layer_sizes=layer_sizes,
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            batch_size=int(data["batch_size"]),
            hidden_activation=hidden_activation,
            class_names=[str(v) for v in data["class_names"].tolist()] if "class_names" in data.files else None,
        )

        num_layers = int(data["num_layers"])
        model.weights = [data[f"w_{i}"].astype(np.float32) for i in range(num_layers)]
        model.biases = [data[f"b_{i}"].astype(np.float32) for i in range(num_layers)]
        return model
