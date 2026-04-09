from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageOps, ImageTk

from src.ann.data import get_npz_class_names, load_from_npz, make_labels_contiguous, train_test_split
from src.ann.mlp import MLPClassifier


class ANNGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ANN Character Recognition Studio")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.model: MLPClassifier | None = None
        self.class_names: list[str] | None = None
        self.target_size = 28
        self.draw_canvas_size = 320
        self.brush_size = tk.IntVar(value=14)
        self._last_draw_x: int | None = None
        self._last_draw_y: int | None = None
        self.draw_image = Image.new("L", (self.draw_canvas_size, self.draw_canvas_size), color=255)
        self.draw_handle = ImageDraw.Draw(self.draw_image)
        self.dataset_path = tk.StringVar(value="character_fonts (with handwritten data).npz")
        self.hidden_layers = tk.StringVar(value="256,128")
        self.learning_rate = tk.StringVar(value="0.01")
        self.epochs = tk.StringVar(value="10")
        self.batch_size = tk.StringVar(value="256")
        self.max_samples = tk.StringVar(value="40000")
        self.val_size = tk.StringVar(value="0.1")
        self.early_stopping = tk.StringVar(value="5")
        self.activation = tk.StringVar(value="relu")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.prediction_var = tk.StringVar(value="No prediction yet. Train or load a model to start.")
        self.preview_image_tk: ImageTk.PhotoImage | None = None
        self.expected_dark_background: bool | None = None

        self.training_thread: threading.Thread | None = None
        self.ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.train_history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self._ui_poll_after_id: str | None = None

        self._build_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._ui_poll_after_id = self.after(80, self._process_ui_events)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.configure(bg="#f4f7fb")
        style.configure("Root.TFrame", background="#f4f7fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#ffffff", foreground="#1f2937", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", foreground="#ffffff", background="#1d4ed8", padding=6)
        style.map("Accent.TButton", background=[("active", "#1e40af")])

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, style="Card.TFrame", padding=12)
        right = ttk.Frame(paned, style="Card.TFrame", padding=12)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        dataset_box = ttk.LabelFrame(left, text="Dataset", padding=10)
        dataset_box.pack(fill=tk.X)
        dataset_box.columnconfigure(1, weight=1)

        ttk.Label(dataset_box, text="NPZ file").grid(row=0, column=0, sticky="w")
        ttk.Entry(dataset_box, textvariable=self.dataset_path).grid(row=0, column=1, sticky="we", padx=(8, 6))
        ttk.Button(dataset_box, text="Browse", command=self._browse_dataset).grid(row=0, column=2)

        hp_box = ttk.LabelFrame(left, text="Hyperparameters", padding=10)
        hp_box.pack(fill=tk.X, pady=(10, 0))
        hp_box.columnconfigure(1, weight=1)

        form_fields = [
            ("Hidden layers", self.hidden_layers),
            ("Learning rate", self.learning_rate),
            ("Epochs", self.epochs),
            ("Batch size", self.batch_size),
            ("Max samples", self.max_samples),
            ("Validation size", self.val_size),
            ("Early stopping", self.early_stopping),
        ]

        for row, (label, variable) in enumerate(form_fields):
            ttk.Label(hp_box, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(hp_box, textvariable=variable).grid(row=row, column=1, sticky="we", padx=(8, 0), pady=3)

        row = len(form_fields)
        ttk.Label(hp_box, text="Activation").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(hp_box, textvariable=self.activation, values=["relu", "sigmoid"], state="readonly").grid(
            row=row, column=1, sticky="we", padx=(8, 0), pady=3
        )

        actions = ttk.Frame(left, style="Card.TFrame")
        actions.pack(fill=tk.X, pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)

        self.train_button = ttk.Button(actions, text="Train Model", style="Accent.TButton", command=self._train_clicked)
        self.train_button.grid(row=0, column=0, sticky="we", padx=(0, 6))

        self.load_button = ttk.Button(actions, text="Load Model", command=self._load_model)
        self.load_button.grid(row=0, column=1, sticky="we", padx=3)

        self.predict_image_button = ttk.Button(actions, text="Predict Image", command=self._predict_image)
        self.predict_image_button.grid(row=0, column=2, sticky="we", padx=(6, 0))

        ttk.Label(left, text="Training progress").pack(anchor="w", pady=(10, 3))
        self.progress = ttk.Progressbar(left, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress.pack(fill=tk.X)

        status_box = ttk.LabelFrame(left, text="Status", padding=8)
        status_box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.status_text = ScrolledText(status_box, height=20, wrap=tk.WORD, font=("Consolas", 10))
        self.status_text.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        draw_tab = ttk.Frame(notebook, padding=8)
        metrics_tab = ttk.Frame(notebook, padding=8)
        probs_tab = ttk.Frame(notebook, padding=8)

        notebook.add(draw_tab, text="Draw and Predict")
        notebook.add(metrics_tab, text="Training Metrics")
        notebook.add(probs_tab, text="Probability View")

        self._build_draw_tab(draw_tab)
        self._build_metrics_tab(metrics_tab)
        self._build_probability_tab(probs_tab)

    def _build_draw_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        draw_box = ttk.LabelFrame(parent, text="Drawing Pad", padding=8)
        draw_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.draw_canvas = tk.Canvas(
            draw_box,
            width=self.draw_canvas_size,
            height=self.draw_canvas_size,
            bg="white",
            highlightthickness=1,
            highlightbackground="#9ca3af",
            cursor="crosshair",
        )
        self.draw_canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.draw_canvas.bind("<ButtonPress-1>", self._on_draw_start)
        self.draw_canvas.bind("<B1-Motion>", self._on_draw_move)
        self.draw_canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        draw_box.rowconfigure(0, weight=1)
        draw_box.columnconfigure(1, weight=1)

        ttk.Label(draw_box, text="Brush size").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(draw_box, from_=4, to=28, variable=self.brush_size, orient=tk.HORIZONTAL).grid(
            row=1, column=1, sticky="we", padx=8, pady=(8, 0)
        )
        ttk.Button(draw_box, text="Clear", command=self._clear_drawing).grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Button(draw_box, text="Predict Drawing", style="Accent.TButton", command=self._predict_drawing).grid(
            row=2, column=0, columnspan=3, sticky="we", pady=(8, 0)
        )

        pred_box = ttk.LabelFrame(parent, text="Prediction Details", padding=8)
        pred_box.grid(row=0, column=1, sticky="nsew")
        pred_box.columnconfigure(0, weight=1)

        ttk.Label(pred_box, textvariable=self.prediction_var, style="Title.TLabel", wraplength=260, justify="left").grid(
            row=0, column=0, sticky="we"
        )
        ttk.Label(pred_box, text="Processed 28x28 input").grid(row=1, column=0, sticky="w", pady=(12, 4))

        self.preview_label = ttk.Label(pred_box)
        self.preview_label.grid(row=2, column=0, sticky="w")

        ttk.Label(pred_box, text="Top predictions").grid(row=3, column=0, sticky="w", pady=(12, 4))
        self.top_tree = ttk.Treeview(pred_box, columns=("class", "prob"), show="headings", height=10)
        self.top_tree.heading("class", text="Class")
        self.top_tree.heading("prob", text="Probability")
        self.top_tree.column("class", width=120, anchor="w")
        self.top_tree.column("prob", width=100, anchor="e")
        self.top_tree.grid(row=4, column=0, sticky="nsew")
        pred_box.rowconfigure(4, weight=1)

    def _build_metrics_tab(self, parent: ttk.Frame) -> None:
        self.metrics_fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(9.5, 3.7), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=parent)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._refresh_training_plot()

    def _build_probability_tab(self, parent: ttk.Frame) -> None:
        self.prob_fig, self.ax_prob = plt.subplots(figsize=(9.5, 4.8), dpi=100)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=parent)
        self.prob_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_probability_plot()

    def _browse_dataset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if path:
            self.dataset_path.set(path)

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(initialdir="models", filetypes=[("NPZ model", "*.npz")])
        if not path:
            return

        try:
            model = MLPClassifier.load(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.model = model
        self.class_names = model.class_names
        self.expected_dark_background = self._estimate_dataset_dark_background()
        self.prediction_var.set("Model loaded. Draw or upload an image to predict.")
        self._append_status(f"Loaded model from {Path(path)}")
        if self.expected_dark_background is not None:
            mode = "dark background / light character" if self.expected_dark_background else "light background / dark character"
            self._append_status(f"Estimated dataset polarity: {mode}")

    def _append_status(self, message: str) -> None:
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)

        line_count = int(self.status_text.index("end-1c").split(".")[0])
        if line_count > 450:
            self.status_text.delete("1.0", f"{line_count - 350}.0")

    def _queue_event(self, event: str, payload: object) -> None:
        self.ui_events.put((event, payload))

    def _process_ui_events(self) -> None:
        if not self.winfo_exists():
            return

        processed = 0
        while processed < 80:
            try:
                event, payload = self.ui_events.get_nowait()
            except queue.Empty:
                break

            self._handle_ui_event(event, payload)
            processed += 1

        if self.winfo_exists():
            self._ui_poll_after_id = self.after(80, self._process_ui_events)

    def _on_close(self) -> None:
        if self._ui_poll_after_id is not None:
            try:
                self.after_cancel(self._ui_poll_after_id)
            except tk.TclError:
                pass
            self._ui_poll_after_id = None

        try:
            plt.close(self.metrics_fig)
            plt.close(self.prob_fig)
        except Exception:
            pass

        self.destroy()

    def _handle_ui_event(self, event: str, payload: object) -> None:
        if event == "status":
            self._append_status(str(payload))
            return

        if event == "dataset_polarity" and isinstance(payload, bool):
            self.expected_dark_background = payload
            mode = "dark background / light character" if payload else "light background / dark character"
            self._append_status(f"Detected dataset polarity: {mode}")
            return

        if event == "train_start":
            total_epochs = int(payload) if isinstance(payload, int) else 1
            self.progress.configure(maximum=max(1, total_epochs))
            self.progress_var.set(0.0)
            self._reset_training_history()
            self._refresh_training_plot()
            self._append_status(f"Training started ({total_epochs} epochs)...")
            return

        if event == "epoch" and isinstance(payload, dict):
            epoch = int(payload.get("epoch", 1))
            total = int(payload.get("total", max(1, epoch)))

            self.progress.configure(maximum=max(1, total))
            self.progress_var.set(float(epoch))

            self.train_history["loss"].append(float(payload.get("loss", 0.0)))
            self.train_history["accuracy"].append(float(payload.get("accuracy", 0.0)))
            if "val_loss" in payload:
                self.train_history["val_loss"].append(float(payload["val_loss"]))
            if "val_accuracy" in payload:
                self.train_history["val_accuracy"].append(float(payload["val_accuracy"]))

            self._refresh_training_plot()

            status = f"Epoch {epoch}/{total} | loss={float(payload.get('loss', 0.0)):.4f} | acc={float(payload.get('accuracy', 0.0)):.4f}"
            if "val_loss" in payload and "val_accuracy" in payload:
                status += f" | val_loss={float(payload['val_loss']):.4f} | val_acc={float(payload['val_accuracy']):.4f}"
            self._append_status(status)
            return

        if event == "train_complete" and isinstance(payload, dict):
            model = payload.get("model")
            class_names = payload.get("class_names")
            metrics = payload.get("metrics")
            model_path = payload.get("model_path")
            best_epoch = payload.get("best_epoch")
            epochs_ran = int(payload.get("epochs_ran", self.progress["maximum"]))

            if isinstance(model, MLPClassifier):
                self.model = model
            self.class_names = class_names if isinstance(class_names, list) else self.class_names

            self.progress_var.set(float(epochs_ran))
            self.train_button.configure(state=tk.NORMAL)
            self.prediction_var.set("Model ready. Draw or upload an image to predict.")

            test_loss = float("nan")
            test_acc = float("nan")
            if isinstance(metrics, dict):
                if "loss" in metrics:
                    test_loss = float(metrics["loss"])
                if "accuracy" in metrics:
                    test_acc = float(metrics["accuracy"])

            self._append_status(f"Training complete. Test loss={test_loss:.4f}, accuracy={test_acc:.4f}")
            if isinstance(best_epoch, int):
                self._append_status(f"Best validation epoch: {best_epoch}")
            if model_path is not None:
                self._append_status(f"Model saved to {model_path}")
            return

        if event == "train_error":
            self.train_button.configure(state=tk.NORMAL)
            self.progress_var.set(0.0)
            error_message = str(payload)
            self._append_status(f"Training error: {error_message}")
            messagebox.showerror("Training error", error_message)
            return

    def _parse_hidden_layers(self) -> list[int]:
        parts = [part.strip() for part in self.hidden_layers.get().split(",") if part.strip()]
        if not parts:
            raise ValueError("Hidden layers cannot be empty. Example: 256,128")

        hidden = [int(part) for part in parts]
        if any(value <= 0 for value in hidden):
            raise ValueError("Hidden layer sizes must be positive integers.")
        return hidden

    def _collect_train_config(self) -> dict[str, object]:
        dataset = Path(self.dataset_path.get().strip())
        if not dataset.exists():
            raise ValueError(f"Dataset not found: {dataset}")

        hidden = self._parse_hidden_layers()
        learning_rate = float(self.learning_rate.get())
        epochs = int(self.epochs.get())
        batch_size = int(self.batch_size.get())
        max_samples = int(self.max_samples.get())
        val_size = float(self.val_size.get())
        early_stopping = int(self.early_stopping.get())
        activation = self.activation.get().strip().lower()

        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be > 0.")
        if epochs <= 0:
            raise ValueError("Epochs must be > 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0.")
        if max_samples < 0:
            raise ValueError("Max samples must be >= 0.")
        if not 0.0 <= val_size < 0.5:
            raise ValueError("Validation size must be in [0.0, 0.5).")
        if early_stopping <= 0:
            raise ValueError("Early stopping must be > 0.")
        if activation not in {"relu", "sigmoid"}:
            raise ValueError("Activation must be relu or sigmoid.")

        return {
            "dataset": dataset,
            "hidden": hidden,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "val_size": val_size,
            "early_stopping": early_stopping,
            "activation": activation,
        }

    def _estimate_dataset_dark_background(self) -> bool | None:
        dataset = Path(self.dataset_path.get().strip())
        if not dataset.exists():
            return None

        try:
            with np.load(dataset, allow_pickle=False) as data:
                if "images" not in data.files:
                    return None
                images = data["images"]
                if images.size == 0:
                    return None

                sample_count = min(512, int(images.shape[0]))
                mean_intensity = float(np.mean(images[:sample_count]))
                return mean_intensity < 127.0
        except Exception:
            return None

    def _train_clicked(self) -> None:
        if self.training_thread is not None and self.training_thread.is_alive():
            messagebox.showinfo("Training in progress", "Please wait for the current run to finish.")
            return

        try:
            cfg = self._collect_train_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.train_button.configure(state=tk.DISABLED)
        self._queue_event("status", f"Preparing training job with dataset: {Path(cfg['dataset']).name}")

        self.training_thread = threading.Thread(target=self._train_worker, args=(cfg,), daemon=True)
        self.training_thread.start()

    def _train_worker(self, cfg: dict[str, object]) -> None:
        try:
            dataset = Path(cfg["dataset"])
            hidden = list(cfg["hidden"])
            learning_rate = float(cfg["learning_rate"])
            epochs = int(cfg["epochs"])
            batch_size = int(cfg["batch_size"])
            max_samples = int(cfg["max_samples"])
            val_size = float(cfg["val_size"])
            early_stopping = int(cfg["early_stopping"])
            activation = str(cfg["activation"])

            self._queue_event("status", "Loading dataset...")
            x, y = load_from_npz(dataset, target_size=self.target_size, flatten=True, normalize=True)
            class_names = get_npz_class_names(dataset)
            y, class_names = make_labels_contiguous(y, class_names)

            dark_background = bool(float(np.mean(x[: min(len(x), 2000)])) < 0.5)
            self._queue_event("dataset_polarity", dark_background)

            if max_samples > 0 and len(y) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y), size=max_samples, replace=False)
                x = x[idx]
                y = y[idx]

            self._queue_event("status", f"Loaded {len(y)} samples across {len(class_names)} classes.")

            x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, seed=42, stratify=True)

            if val_size > 0.0:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_full,
                    y_train_full,
                    test_size=val_size,
                    seed=43,
                    stratify=True,
                )
            else:
                x_train, y_train = x_train_full, y_train_full
                x_val, y_val = None, None

            self._queue_event(
                "status",
                f"Train={len(y_train)} Validation={0 if y_val is None else len(y_val)} Test={len(y_test)}",
            )

            model = MLPClassifier(
                layer_sizes=[x_train.shape[1], *hidden, len(class_names)],
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                hidden_activation=activation,
                seed=42,
                class_names=class_names,
            )

            self._queue_event("train_start", epochs)

            def on_epoch(epoch: int, total: int, metrics: dict[str, float]) -> None:
                payload: dict[str, float | int] = {"epoch": epoch, "total": total}
                payload.update(metrics)
                self._queue_event("epoch", payload)

            history = model.fit(
                x_train,
                y_train,
                x_val=x_val,
                y_val=y_val,
                early_stopping_patience=early_stopping,
                verbose=False,
                epoch_callback=on_epoch,
            )
            metrics = model.evaluate(x_test, y_test)

            out_path = Path("models/gui_mlp_model.npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(out_path))

            self._queue_event(
                "train_complete",
                {
                    "model": model,
                    "class_names": class_names,
                    "metrics": metrics,
                    "model_path": str(out_path),
                    "best_epoch": history.best_epoch if len(history.val_losses) > 0 else None,
                    "epochs_ran": len(history.losses),
                },
            )
        except Exception as exc:
            self._queue_event("train_error", str(exc))

    def _predict_image(self) -> None:
        if self.model is None:
            messagebox.showwarning("Model missing", "Train or load a model first.")
            return

        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not image_path:
            return

        try:
            image = Image.open(image_path).convert("L")
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))
            return

        prepared = self._prepare_image_for_model(image)
        self._run_prediction(prepared, source=Path(image_path).name)

    def _prepare_image_for_model(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)

        dark_ratio = float(np.mean(arr < 128))
        bright_ratio = 1.0 - dark_ratio
        foreground_is_dark = dark_ratio < bright_ratio

        if foreground_is_dark:
            mask = arr < 245
            background_color = 255
        else:
            mask = arr > 10
            background_color = 0

        if np.any(mask):
            ys, xs = np.where(mask)
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1

            cropped = gray.crop((x0, y0, x1, y1))
            side = max(cropped.width, cropped.height) + 12
            square = Image.new("L", (side, side), color=background_color)
            square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
            gray = square

        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
        return gray

    def _run_prediction(self, prepared: Image.Image, source: str) -> None:
        if self.model is None:
            return

        x = np.asarray(prepared, dtype=np.float32).reshape(1, -1) / 255.0
        probs_normal = self.model.predict_proba(x)[0]

        prepared_inverted = ImageOps.invert(prepared)
        x_inverted = np.asarray(prepared_inverted, dtype=np.float32).reshape(1, -1) / 255.0
        probs_inverted = self.model.predict_proba(x_inverted)[0]

        conf_normal = float(np.max(probs_normal))
        conf_inverted = float(np.max(probs_inverted))

        if source == "Drawing" and self.expected_dark_background is not None:
            prefer_inverted = self.expected_dark_background
        else:
            prefer_inverted = conf_inverted > conf_normal + 0.03
            if source == "Drawing" and conf_inverted > conf_normal + 0.01:
                prefer_inverted = True

        if prefer_inverted and conf_normal > conf_inverted + 0.20:
            prefer_inverted = False
        if (not prefer_inverted) and conf_inverted > conf_normal + 0.20:
            prefer_inverted = True

        probs = probs_inverted if prefer_inverted else probs_normal
        used_input = prepared_inverted if prefer_inverted else prepared
        polarity_mode = "inverted" if prefer_inverted else "original"

        pred = int(np.argmax(probs))
        label = self._label_for_prediction(pred)
        confidence = float(probs[pred])
        self.prediction_var.set(f"{source}: {label} ({pred}) with confidence {confidence:.2%} [{polarity_mode}]")

        self._append_status(
            f"Prediction for {source}: {label} ({pred}) | confidence={confidence:.4f} | mode={polarity_mode} "
            f"(orig={conf_normal:.4f}, inv={conf_inverted:.4f})"
        )
        self._update_top_predictions(probs)
        self._update_probability_plot(probs)
        self._update_preview(used_input)

    def _update_top_predictions(self, probs: np.ndarray) -> None:
        for item in self.top_tree.get_children():
            self.top_tree.delete(item)

        top_k = min(8, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            label = self._label_for_prediction(int(idx))
            self.top_tree.insert("", tk.END, values=(f"{rank}. {label}", f"{float(probs[idx]):.4f}"))

    def _update_preview(self, prepared: Image.Image) -> None:
        upscaled = prepared.resize((196, 196), Image.Resampling.NEAREST)
        tint = ImageOps.colorize(upscaled, black="#111111", white="#ecf2f8")
        photo = ImageTk.PhotoImage(tint)
        self.preview_image_tk = photo
        self.preview_label.configure(image=photo)

    def _draw_empty_probability_plot(self) -> None:
        self.ax_prob.clear()
        self.ax_prob.set_title("Class Probabilities")
        self.ax_prob.text(
            0.5,
            0.5,
            "Predict an image or drawing to visualize probabilities.",
            ha="center",
            va="center",
            transform=self.ax_prob.transAxes,
            fontsize=11,
            color="#6b7280",
        )
        self.ax_prob.set_xticks([])
        self.ax_prob.set_yticks([])
        self.prob_fig.tight_layout()
        self.prob_canvas.draw_idle()

    def _update_probability_plot(self, probs: np.ndarray) -> None:
        self.ax_prob.clear()

        top_n = min(12, len(probs))
        top_indices = np.argsort(probs)[-top_n:][::-1]
        labels = [self._label_for_prediction(int(idx)) for idx in top_indices]
        values = [float(probs[int(idx)]) for idx in top_indices]
        y_axis = np.arange(top_n)

        bars = self.ax_prob.barh(y_axis, values, color="#2a9d8f")
        self.ax_prob.set_yticks(y_axis)
        self.ax_prob.set_yticklabels(labels)
        self.ax_prob.invert_yaxis()
        self.ax_prob.set_title("Top Class Probabilities")
        self.ax_prob.set_xlabel("Probability")
        self.ax_prob.grid(axis="x", linestyle="--", alpha=0.35)

        max_value = max(values) if values else 1.0
        self.ax_prob.set_xlim(0.0, max(1.0, max_value * 1.15))

        for bar, value in zip(bars, values):
            self.ax_prob.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

        self.prob_fig.tight_layout()
        self.prob_canvas.draw_idle()

    def _reset_training_history(self) -> None:
        for key in self.train_history:
            self.train_history[key] = []

    def _refresh_training_plot(self) -> None:
        self.ax_loss.clear()
        self.ax_acc.clear()

        train_loss = self.train_history["loss"]
        train_acc = self.train_history["accuracy"]
        val_loss = self.train_history["val_loss"]
        val_acc = self.train_history["val_accuracy"]

        if not train_loss:
            self.ax_loss.set_title("Training Loss")
            self.ax_loss.text(
                0.5,
                0.5,
                "Train the model to view loss curves",
                transform=self.ax_loss.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )

            self.ax_acc.set_title("Training Accuracy")
            self.ax_acc.text(
                0.5,
                0.5,
                "Accuracy curve appears during training",
                transform=self.ax_acc.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )
        else:
            epochs = np.arange(1, len(train_loss) + 1)
            self.ax_loss.plot(epochs, train_loss, color="#2563eb", linewidth=2.0, label="train")
            if len(val_loss) == len(train_loss):
                self.ax_loss.plot(epochs, val_loss, color="#f97316", linewidth=2.0, label="val")
            self.ax_loss.set_title("Training Loss")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.grid(alpha=0.25)
            self.ax_loss.legend()

            self.ax_acc.plot(epochs, train_acc, color="#16a34a", linewidth=2.0, label="train")
            if len(val_acc) == len(train_acc):
                self.ax_acc.plot(epochs, val_acc, color="#b45309", linewidth=2.0, label="val")
            self.ax_acc.set_title("Training Accuracy")
            self.ax_acc.set_xlabel("Epoch")
            self.ax_acc.set_ylabel("Accuracy")
            self.ax_acc.set_ylim(0.0, 1.0)
            self.ax_acc.grid(alpha=0.25)
            self.ax_acc.legend()

        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw_idle()

    def _label_for_prediction(self, pred: int) -> str:
        if self.model is not None and self.model.class_names is not None:
            return self.model.class_names[pred]
        if self.class_names is not None:
            return self.class_names[pred]
        if self.model is not None and self.model.layer_sizes[-1] == 26:
            return chr(pred + ord("A"))
        return str(pred)

    def _clear_drawing(self) -> None:
        self.draw_canvas.delete("all")
        self.draw_image = Image.new("L", (self.draw_canvas_size, self.draw_canvas_size), color=255)
        self.draw_handle = ImageDraw.Draw(self.draw_image)
        self.prediction_var.set("Canvas cleared. Draw a character to predict.")
        self._last_draw_x = None
        self._last_draw_y = None

    def _on_draw_start(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        self._last_draw_x = x
        self._last_draw_y = y

        width = int(self.brush_size.get())
        r = max(1, width // 2)
        self.draw_canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw_handle.ellipse((x - r, y - r, x + r, y + r), fill=0)

    def _on_draw_move(self, event: tk.Event) -> None:
        if self._last_draw_x is None or self._last_draw_y is None:
            self._on_draw_start(event)
            return

        x0, y0 = self._last_draw_x, self._last_draw_y
        x1, y1 = int(event.x), int(event.y)
        width = int(self.brush_size.get())

        self.draw_canvas.create_line(x0, y0, x1, y1, fill="black", width=width, capstyle=tk.ROUND, smooth=True)
        self.draw_handle.line((x0, y0, x1, y1), fill=0, width=width)

        self._last_draw_x = x1
        self._last_draw_y = y1

    def _on_draw_end(self, _event: tk.Event) -> None:
        self._last_draw_x = None
        self._last_draw_y = None

    def _predict_drawing(self) -> None:
        if self.model is None:
            messagebox.showwarning("Model missing", "Train or load a model first.")
            return

        prepared = self._prepare_image_for_model(self.draw_image)
        self._run_prediction(prepared, source="Drawing")


if __name__ == "__main__":
    app = ANNGui()
    app.mainloop()
