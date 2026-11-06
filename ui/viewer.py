"""
ImageViewer: PACS-style canvas viewer.
  wheel = zoom about cursor | left-drag = pan | click = set nodule seed | right-click = clear seed
  double-click = fit | overlays: seed crosshair, segmentation contour, scale bar, corner annotations
Image coordinates are pixel indices (x = column, y = row); pixel i spans [i, i+1) in continuous space.
"""
import tkinter as tk
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from PIL import Image, ImageTk

from ui.theme import T, Fonts

NICE_MM = (1, 2, 5, 10, 20, 50, 100)


class ImageViewer(tk.Canvas):
    def __init__(self, parent,
                 on_seed: Optional[Callable[[int, int], None]] = None,
                 on_clear_seed: Optional[Callable[[], None]] = None,
                 on_hover: Optional[Callable[[Optional[Tuple[int, int]]], None]] = None):
        super().__init__(parent, bg=T.VIEW_BG, highlightthickness=0, cursor="crosshair")
        self.on_seed, self.on_clear_seed, self.on_hover = on_seed, on_clear_seed, on_hover
        self._pil: Optional[Image.Image] = None
        self._photo = None
        self._zoom, self._fit, self._ox, self._oy = 1.0, 1.0, 0.0, 0.0
        self._spatial = True
        self._seed: Optional[Tuple[int, int]] = None
        self._contours: List[np.ndarray] = []
        self._label: Optional[str] = None
        self._spacing: Optional[Tuple[float, float]] = None
        self._info: Sequence[str] = ()
        self._caption: Optional[str] = None
        self._drag = None
        self._moved = False

        self.bind("<Configure>", lambda e: self._on_resize())
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag_move)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Button-3>", self._right)
        self.bind("<Button-2>", self._right)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self.on_hover and self.on_hover(None))

    # ---------------------------------------------------------------- public API
    def set_image(self, bgr: Optional[np.ndarray], spatial: bool = True, caption: Optional[str] = None,
                  keep_view: bool = False):
        had = self._pil is not None
        if bgr is None:
            self._pil = None
        else:
            img = bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            self._pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._spatial, self._caption = spatial, caption
        if not (keep_view and had and self._pil is not None):
            self.fit(redraw=False)
        self.redraw()

    def set_spacing(self, spacing: Optional[Tuple[float, float]]):
        self._spacing = spacing

    def set_info(self, lines: Sequence[str]):
        self._info = tuple(lines)
        self._draw_overlays()

    def set_seed(self, seed: Optional[Tuple[int, int]]):
        self._seed = seed
        self._draw_overlays()

    def set_overlay(self, mask: Optional[np.ndarray], label: Optional[str] = None):
        self._contours = []
        if mask is not None and int(mask.sum()) > 0:
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self._contours = [c.reshape(-1, 2).astype(np.float32) for c in cnts if len(c) >= 3]
        self._label = label
        self._draw_overlays()

    def fit(self, redraw: bool = True):
        if self._pil is None:
            return
        cw, ch = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        w, h = self._pil.size
        self._fit = min((cw - 24) / w, (ch - 24) / h)
        self._zoom = self._fit
        self._ox, self._oy = (cw - w * self._zoom) / 2, (ch - h * self._zoom) / 2
        if redraw:
            self.redraw()

    # ---------------------------------------------------------------- geometry
    def _to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._zoom + self._ox, y * self._zoom + self._oy

    def _to_image(self, cx: float, cy: float) -> Tuple[int, int]:
        return int(np.floor((cx - self._ox) / self._zoom)), int(np.floor((cy - self._oy) / self._zoom))

    def _inside(self, ix: int, iy: int) -> bool:
        return self._pil is not None and 0 <= ix < self._pil.width and 0 <= iy < self._pil.height

    # ---------------------------------------------------------------- events
    def _on_resize(self):
        if self._pil is None:
            self.redraw()
            return
        old_fit = self._fit
        at_fit = abs(self._zoom - old_fit) < 1e-6
        if at_fit:
            self.fit(redraw=False)
        self.redraw()

    def _wheel(self, e):
        self._zoom_at(e.x, e.y, 1.15 if e.delta > 0 else 1 / 1.15)

    def _zoom_at(self, cx, cy, factor):
        if self._pil is None:
            return
        nz = min(max(self._zoom * factor, self._fit * 0.5), 24.0)
        f = nz / self._zoom
        self._ox, self._oy = cx - (cx - self._ox) * f, cy - (cy - self._oy) * f
        self._zoom = nz
        self.redraw()

    def _press(self, e):
        self._drag, self._moved = (e.x, e.y, self._ox, self._oy), False
        self.focus_set()

    def _drag_move(self, e):
        if self._drag is None or self._pil is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        if abs(dx) + abs(dy) > 4:
            self._moved = True
        if self._moved:
            self._ox, self._oy = self._drag[2] + dx, self._drag[3] + dy
            self.redraw()
            self.configure(cursor="fleur")

    def _release(self, e):
        self.configure(cursor="crosshair")
        if self._drag is not None and not self._moved and self._spatial and self._pil is not None:
            ix, iy = self._to_image(e.x, e.y)
            if self._inside(ix, iy) and self.on_seed:
                self.on_seed(ix, iy)
        self._drag = None

    def _right(self, _):
        if self.on_clear_seed:
            self.on_clear_seed()

    def _motion(self, e):
        if self.on_hover is None:
            return
        if self._pil is None or not self._spatial:
            self.on_hover(None)
            return
        ix, iy = self._to_image(e.x, e.y)
        self.on_hover((ix, iy) if self._inside(ix, iy) else None)

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        self.delete("all")
        cw, ch = self.winfo_width(), self.winfo_height()
        if self._pil is None:
            self.create_text(cw / 2, ch / 2 - 14, text="No scan loaded", font=Fonts.title, fill=T.VIEW_FG)
            self.create_text(cw / 2, ch / 2 + 14, text="Open a CT slice with Ctrl+O", font=Fonts.body, fill=T.VIEW_MUTED)
            return
        W, H = self._pil.size
        x0, y0 = max(0, int(np.floor(-self._ox / self._zoom))), max(0, int(np.floor(-self._oy / self._zoom)))
        x1 = min(W, int(np.ceil((cw - self._ox) / self._zoom)))
        y1 = min(H, int(np.ceil((ch - self._oy) / self._zoom)))
        if x1 > x0 and y1 > y0:
            crop = self._pil.crop((x0, y0, x1, y1))
            size = (max(1, round((x1 - x0) * self._zoom)), max(1, round((y1 - y0) * self._zoom)))
            resample = Image.NEAREST if self._zoom >= 4 else Image.BILINEAR
            self._photo = ImageTk.PhotoImage(crop.resize(size, resample))
            px, py = self._to_canvas(x0, y0)
            self.create_image(round(px), round(py), image=self._photo, anchor="nw", tags="img")
        self._draw_overlays()

    def _draw_overlays(self):
        self.delete("ov")
        if self._pil is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        z = self._zoom

        if self._spatial:
            for c in self._contours:
                pts = [v for p in c for v in self._to_canvas(p[0] + 0.5, p[1] + 0.5)]
                self.create_line(*pts, *pts[:2], fill=T.OVERLAY, width=2, tags="ov", joinstyle="round")
            if self._contours and self._label:
                allp = np.vstack(self._contours)
                tx, ty = self._to_canvas(float(allp[:, 0].mean()) + 0.5, float(allp[:, 1].min()) - 3)
                self.create_text(tx + 1, ty - 8 + 1, text=self._label, font=Fonts.small_b, fill="#000000", tags="ov")
                self.create_text(tx, ty - 8, text=self._label, font=Fonts.small_b, fill=T.OVERLAY, tags="ov")
            if self._seed is not None:
                sx, sy = self._to_canvas(self._seed[0] + 0.5, self._seed[1] + 0.5)
                r, g = 6, 3
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    self.create_line(sx + dx * g, sy + dy * g, sx + dx * (r + 7), sy + dy * (r + 7),
                                     fill=T.SEED, width=2, tags="ov")
                self.create_oval(sx - r, sy - r, sx + r, sy + r, outline=T.SEED, width=2, tags="ov")

            if self._spacing:          # scale bar
                mm_px = (self._spacing[0] + self._spacing[1]) / 2.0
                for mm in NICE_MM:
                    length = mm / mm_px * z
                    if length >= 70:
                        break
                bx, by = 18, ch - 22
                self.create_line(bx, by, bx + length, by, fill=T.VIEW_FG, width=2, tags="ov")
                for x in (bx, bx + length):
                    self.create_line(x, by - 4, x, by + 4, fill=T.VIEW_FG, width=2, tags="ov")
                self.create_text(bx + length / 2, by - 10, text=f"{mm} mm", font=Fonts.micro_b, fill=T.VIEW_FG, tags="ov")

        # corner annotations
        for i, line in enumerate(self._info):
            self.create_text(14, 14 + i * 15, text=line, anchor="nw", font=Fonts.mono, fill=T.VIEW_FG, tags="ov")
        right = f"{z * 100:.0f}%" if self._spatial else (self._caption or "")
        self.create_text(cw - 14, 14, text=right, anchor="ne", font=Fonts.mono, fill=T.VIEW_FG, tags="ov")
        if not self._spatial and self._caption is None:
            pass
        self.create_text(cw - 14, ch - 14, text="RESEARCH USE ONLY", anchor="se", font=Fonts.micro_b,
                         fill=T.VIEW_MUTED, tags="ov")
