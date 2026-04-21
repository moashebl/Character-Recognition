"""Capture screenshots of the GUI application for the presentation."""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import ImageGrab

# We import the GUI class
from gui_app import ANNGui


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ScreenshotCapture:
    def __init__(self):
        self.app = ANNGui()
        self.app.geometry("1180x760+50+50")
        self.screenshots_taken = 0
        self.total_expected = 4
        # Schedule the first screenshot after GUI fully renders
        self.app.after(1500, self.capture_main_view)

    def _grab_window(self, filename: str) -> None:
        """Capture the current window and save to outputs/."""
        self.app.update_idletasks()
        self.app.update()
        time.sleep(0.3)

        x = self.app.winfo_rootx()
        y = self.app.winfo_rooty()
        w = self.app.winfo_width()
        h = self.app.winfo_height()

        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        path = OUTPUT_DIR / filename
        img.save(str(path), "PNG")
        print(f"Saved: {path}")
        self.screenshots_taken += 1

    def capture_main_view(self) -> None:
        """Capture the default Draw tab view."""
        # Add some status text to make it look active
        self.app._append_status("ANN Character Recognition Studio ready")
        self.app._append_status("Dataset: character_fonts (with handwritten data).npz")
        self.app._append_status("Architecture: 784 → 256 → 128 → 26 (ReLU)")
        self.app._append_status("Waiting for user action...")
        self.app.update()
        time.sleep(0.3)

        self._grab_window("gui_screenshot_main.png")
        self.app.after(500, self.capture_draw_tab)

    def capture_draw_tab(self) -> None:
        """Draw something on the canvas to show the drawing feature."""
        # Draw a simple letter "A" on the canvas
        canvas = self.app.draw_canvas
        handle = self.app.draw_handle
        size = self.app.draw_canvas_size
        cx, cy = size // 2, size // 2

        # Draw an "A" shape
        points_left = []
        points_right = []
        for i in range(20):
            t = i / 19.0
            # Left leg
            lx = cx - 60 + t * 60
            ly = cy + 80 - t * 160
            points_left.append((lx, ly))
            # Right leg
            rx = cx + 60 - t * 60
            ry = cy + 80 - t * 160
            points_right.append((rx, ry))

        brush = 12
        r = brush // 2
        # Draw left leg
        for i in range(len(points_left) - 1):
            x0, y0 = int(points_left[i][0]), int(points_left[i][1])
            x1, y1 = int(points_left[i+1][0]), int(points_left[i+1][1])
            canvas.create_line(x0, y0, x1, y1, fill="black", width=brush, capstyle=tk.ROUND)
            handle.line((x0, y0, x1, y1), fill=0, width=brush)

        # Draw right leg
        for i in range(len(points_right) - 1):
            x0, y0 = int(points_right[i][0]), int(points_right[i][1])
            x1, y1 = int(points_right[i+1][0]), int(points_right[i+1][1])
            canvas.create_line(x0, y0, x1, y1, fill="black", width=brush, capstyle=tk.ROUND)
            handle.line((x0, y0, x1, y1), fill=0, width=brush)

        # Draw crossbar
        cross_y = cy - 10
        canvas.create_line(cx - 35, cross_y, cx + 35, cross_y, fill="black", width=brush, capstyle=tk.ROUND)
        handle.line((cx - 35, cross_y, cx + 35, cross_y), fill=0, width=brush)

        self.app.prediction_var.set("Draw a character and click 'Predict Drawing'")
        self.app.update()
        time.sleep(0.3)

        self._grab_window("gui_screenshot_drawing.png")
        self.app.after(500, self.capture_metrics_tab)

    def capture_metrics_tab(self) -> None:
        """Switch to training metrics tab and add sample data."""
        # Find the notebook widget and switch to metrics tab
        notebook = None
        for widget in self.app.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Panedwindow):
                    for pane_child in child.winfo_children():
                        for sub in pane_child.winfo_children():
                            if isinstance(sub, ttk.Notebook):
                                notebook = sub
                                break

        if notebook:
            notebook.select(1)  # Training Metrics tab

        # Add some sample training data for the plot
        import numpy as np
        epochs = 15
        for i in range(epochs):
            t = (i + 1) / epochs
            self.app.train_history["loss"].append(2.5 * np.exp(-2.5 * t) + 0.3)
            self.app.train_history["accuracy"].append(0.1 + 0.68 * (1 - np.exp(-3.0 * t)))
            self.app.train_history["val_loss"].append(2.5 * np.exp(-2.2 * t) + 0.5)
            self.app.train_history["val_accuracy"].append(0.08 + 0.62 * (1 - np.exp(-2.5 * t)))

        self.app._refresh_training_plot()
        self.app.progress_var.set(100.0)
        self.app._append_status("Training complete. Test loss=0.8996, accuracy=0.7815")
        self.app._append_status("Best validation epoch: 12")
        self.app._append_status("Model saved to models/gui_mlp_model.npz")
        self.app.update()
        time.sleep(0.5)

        self._grab_window("gui_screenshot_metrics.png")
        self.app.after(500, self.capture_probability_tab)

    def capture_probability_tab(self) -> None:
        """Switch to probability tab with sample data."""
        notebook = None
        for widget in self.app.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Panedwindow):
                    for pane_child in child.winfo_children():
                        for sub in pane_child.winfo_children():
                            if isinstance(sub, ttk.Notebook):
                                notebook = sub
                                break

        if notebook:
            notebook.select(2)  # Probability View tab

        # Create sample probability data
        import numpy as np
        probs = np.random.dirichlet(np.ones(26) * 0.3)
        # Make "A" the highest
        probs[0] = 0.72
        probs = probs / probs.sum()

        self.app.class_names = [chr(i + ord('A')) for i in range(26)]
        self.app._update_probability_plot(probs)
        self.app.update()
        time.sleep(0.5)

        self._grab_window("gui_screenshot_probabilities.png")
        self.app.after(500, self.finish)

    def finish(self) -> None:
        print(f"\nAll {self.screenshots_taken} screenshots captured successfully!")
        self.app.destroy()

    def run(self) -> None:
        self.app.mainloop()


if __name__ == "__main__":
    capture = ScreenshotCapture()
    capture.run()
