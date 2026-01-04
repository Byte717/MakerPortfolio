import os, glob, json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass
class LabeledImg:
    img: np.ndarray
    topLeftPt: Tuple[int, int]
    deltaX: int
    deltaY: int


def crop(mat, topLeft: Tuple[int, int], w: int, h: int):
    return mat[topLeft[1]:topLeft[1] + h, topLeft[0]:topLeft[0] + w]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LabelGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RGB Box Labeler")

        self.folder: Optional[str] = None
        self.paths: List[str] = []
        self.idx: int = 0

        self.labels_path: Optional[str] = None
        self.labels: Dict[str, dict] = {}  # key: imgPath, value: {imgPath, topLeftPt, deltaX, deltaY}

        self.img_bgr: Optional[np.ndarray] = None
        self.img_rgb: Optional[np.ndarray] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        self.scale: float = 1.0
        self.canvas_w = 960
        self.canvas_h = 540

        self.dragging = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0
        self.current_box: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1) in ORIGINAL image coords

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        self.btn_open = tk.Button(top, text="Open Folder", command=self.open_folder)
        self.btn_open.pack(side="left")

        self.btn_prev = tk.Button(top, text="Prev", command=self.prev_img, state="disabled")
        self.btn_prev.pack(side="left", padx=(8, 0))

        self.btn_next = tk.Button(top, text="Next", command=self.next_img, state="disabled")
        self.btn_next.pack(side="left", padx=(8, 0))

        self.btn_clear = tk.Button(top, text="Clear Box", command=self.clear_box, state="disabled")
        self.btn_clear.pack(side="left", padx=(16, 0))

        self.btn_save = tk.Button(top, text="Save Label", command=self.save_label, state="disabled")
        self.btn_save.pack(side="left", padx=(8, 0))

        self.status = tk.Label(top, text="Open a folder to start.")
        self.status.pack(side="right")

        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(padx=8, pady=(0, 8))

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        self.info = tk.Label(bottom, text="")
        self.info.pack(side="left")

        self.tip = tk.Label(
            bottom,
            text="Drag to draw. Keys: S=save, N=next, B=prev, U=clear, Esc=quit",
        )
        self.tip.pack(side="right")

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.root.bind("s", lambda e: self.save_label())
        self.root.bind("n", lambda e: self.next_img())
        self.root.bind("b", lambda e: self.prev_img())
        self.root.bind("u", lambda e: self.clear_box())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(folder, e)))
        paths = sorted(paths)

        if not paths:
            messagebox.showerror("No images", "No images found in that folder.")
            return

        self.folder = folder
        self.paths = paths
        self.idx = 0

        self.labels_path = os.path.join(folder, "labels.jsonl")
        self._load_labels()

        self.btn_prev.config(state="normal")
        self.btn_next.config(state="normal")
        self.btn_clear.config(state="normal")
        self.btn_save.config(state="normal")

        self.load_current()

    def _load_labels(self):
        self.labels = {}
        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.labels[obj["imgPath"]] = obj

    def _write_labels(self):
        if not self.labels_path:
            return
        with open(self.labels_path, "w", encoding="utf-8") as f:
            for p in sorted(self.labels.keys()):
                f.write(json.dumps(self.labels[p]) + "\n")

    def load_current(self):
        if not self.paths:
            return

        path = self.paths[self.idx]
        img = cv2.imread(path)
        if img is None:
            messagebox.showwarning("Bad image", f"Could not read:\n{path}")
            return

        self.img_bgr = img
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.current_box = None
        if path in self.labels:
            tlx, tly = self.labels[path]["topLeftPt"]
            dx = self.labels[path]["deltaX"]
            dy = self.labels[path]["deltaY"]
            self.current_box = (tlx, tly, tlx + dx, tly + dy)

        self._render()

    def _render(self):
        if self.img_rgb is None:
            return

        H, W = self.img_rgb.shape[:2]
        scale = min(self.canvas_w / W, self.canvas_h / H)
        self.scale = scale

        disp_w, disp_h = int(W * scale), int(H * scale)
        pil = Image.fromarray(self.img_rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)

        self.tk_img = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        ox = (self.canvas_w - disp_w) // 2
        oy = (self.canvas_h - disp_h) // 2

        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_img)

        if self.dragging:
            x0, y0 = self._img_to_canvas(self.x0, self.y0, ox, oy)
            x1, y1 = self._img_to_canvas(self.x1, self.y1, ox, oy)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="lime", width=2)

        if self.current_box is not None and not self.dragging:
            x0i, y0i, x1i, y1i = self.current_box
            x0, y0 = self._img_to_canvas(x0i, y0i, ox, oy)
            x1, y1 = self._img_to_canvas(x1i, y1i, ox, oy)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="lime", width=2)

        path = self.paths[self.idx]
        labeled = "LABELED" if path in self.labels else "UNLABELED"
        self.status.config(text=f"{self.idx+1}/{len(self.paths)}  {labeled}")

        if self.current_box is not None:
            x0, y0, x1, y1 = self.current_box
            dx, dy = x1 - x0, y1 - y0
            self.info.config(text=f"topLeftPt=({x0},{y0})  deltaX={dx}  deltaY={dy}")
        else:
            self.info.config(text="No box")

    def _canvas_to_img(self, cx, cy, ox, oy):
        if self.img_rgb is None:
            return 0, 0
        H, W = self.img_rgb.shape[:2]
        x = int((cx - ox) / self.scale)
        y = int((cy - oy) / self.scale)
        x = clamp(x, 0, W - 1)
        y = clamp(y, 0, H - 1)
        return x, y

    def _img_to_canvas(self, x, y, ox, oy):
        cx = int(x * self.scale) + ox
        cy = int(y * self.scale) + oy
        return cx, cy

    def _image_offsets(self):
        if self.img_rgb is None:
            return 0, 0, 0, 0
        H, W = self.img_rgb.shape[:2]
        disp_w, disp_h = int(W * self.scale), int(H * self.scale)
        ox = (self.canvas_w - disp_w) // 2
        oy = (self.canvas_h - disp_h) // 2
        return ox, oy, disp_w, disp_h

    def on_mouse_down(self, event):
        if self.img_rgb is None:
            return
        ox, oy, dw, dh = self._image_offsets()
        if not (ox <= event.x <= ox + dw and oy <= event.y <= oy + dh):
            return

        self.dragging = True
        self.current_box = None
        self.x0, self.y0 = self._canvas_to_img(event.x, event.y, ox, oy)
        self.x1, self.y1 = self.x0, self.y0
        self._render()

    def on_mouse_move(self, event):
        if not self.dragging or self.img_rgb is None:
            return
        ox, oy, dw, dh = self._image_offsets()
        self.x1, self.y1 = self._canvas_to_img(event.x, event.y, ox, oy)
        self._render()

    def on_mouse_up(self, event):
        if not self.dragging or self.img_rgb is None:
            return
        ox, oy, dw, dh = self._image_offsets()
        x1, y1 = self._canvas_to_img(event.x, event.y, ox, oy)

        x0, y0 = self.x0, self.y0
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)

        self.dragging = False

        if abs(x1 - x0) >= 2 and abs(y1 - y0) >= 2:
            self.current_box = (x0, y0, x1, y1)
        else:
            self.current_box = None

        self._render()

    def clear_box(self):
        self.current_box = None
        self._render()

    def save_label(self):
        if not self.paths:
            return
        path = self.paths[self.idx]
        if self.current_box is None:
            messagebox.showinfo("No box", "Draw a box first.")
            return

        x0, y0, x1, y1 = self.current_box
        dx, dy = int(x1 - x0), int(y1 - y0)

        self.labels[path] = {
            "imgPath": path,
            "topLeftPt": [int(x0), int(y0)],
            "deltaX": dx,
            "deltaY": dy,
        }
        self._write_labels()
        self._render()

    def next_img(self):
        if not self.paths:
            return
        self.current_box = None
        self.idx = min(len(self.paths) - 1, self.idx + 1)
        self.load_current()

    def prev_img(self):
        if not self.paths:
            return
        self.current_box = None
        self.idx = max(0, self.idx - 1)
        self.load_current()
if __name__ == "__main__":
    root = tk.Tk()
    app = LabelGUI(root)
    root.mainloop()