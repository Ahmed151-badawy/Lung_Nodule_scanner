"""Small reusable widgets: FlatButton, Segmented, Card, Chip, KV rows, BarGauge, Collapsible, Tooltip, ScrollFrame."""
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Sequence, Tuple

from ui.theme import T, Fonts


class FlatButton(tk.Label):
    """Flat label-button with hover/pressed/disabled states (full colour control on every OS)."""
    STYLES = {
        "primary": dict(bg=T.ACCENT, fg="#FFFFFF", hover=T.ACCENT_HOVER, border=T.ACCENT),
        "secondary": dict(bg=T.SURFACE, fg=T.TEXT, hover=T.SURFACE_ALT, border=T.BORDER),
        "ghost": dict(bg=T.SURFACE, fg=T.MUTED, hover=T.SURFACE_ALT, border=T.SURFACE),
    }

    def __init__(self, parent, text: str, command: Optional[Callable] = None, variant: str = "secondary",
                 font=None, padx: int = 14, pady: int = 8, **kw):
        s = self.STYLES[variant]
        super().__init__(parent, text=text, font=font or Fonts.body_b, bg=s["bg"], fg=s["fg"], padx=padx, pady=pady,
                         cursor="hand2", highlightthickness=1, highlightbackground=s["border"],
                         highlightcolor=T.ACCENT, takefocus=1, **kw)
        self._s, self._command, self._enabled = s, command, True
        self.bind("<Enter>", lambda e: self._enabled and super(FlatButton, self).configure(bg=s["hover"]))
        self.bind("<Leave>", lambda e: self._enabled and super(FlatButton, self).configure(bg=s["bg"]))
        self.bind("<ButtonRelease-1>", self._click)
        self.bind("<Return>", self._click)
        self.bind("<space>", self._click)

    def _click(self, event=None):
        if not self._enabled or not self._command:
            return
        if event is not None and event.type == tk.EventType.ButtonRelease:
            if not (0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()):
                return
        self._command()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            super().configure(bg=self._s["bg"], fg=self._s["fg"], cursor="hand2", highlightbackground=self._s["border"])
        else:
            super().configure(bg=T.SURFACE_ALT, fg=T.FAINT, cursor="arrow", highlightbackground=T.BORDER)


class Segmented(tk.Frame):
    """Two-or-more option segmented control."""

    def __init__(self, parent, options: Sequence[Tuple[str, str]], command: Callable[[str], None], bg=T.SURFACE):
        super().__init__(parent, bg=T.BORDER, padx=1, pady=1)
        self._buttons, self._command, self._bg = {}, command, bg
        for i, (key, label) in enumerate(options):
            b = tk.Label(self, text=label, font=Fonts.small_b, padx=14, pady=5, cursor="hand2")
            b.grid(row=0, column=i, padx=(0 if i == 0 else 1, 0))
            b.bind("<Button-1>", lambda e, k=key: self._command(k))
            self._buttons[key] = b
        self.select(options[0][0])

    def select(self, key: str):
        for k, b in self._buttons.items():
            on = (k == key)
            b.configure(bg=T.ACCENT if on else self._bg, fg="#FFFFFF" if on else T.MUTED)

    def set_enabled(self, key: str, enabled: bool):
        self._buttons[key].configure(fg=T.MUTED if enabled else T.FAINT, cursor="hand2" if enabled else "arrow")


class Chip(tk.Label):
    def __init__(self, parent, text: str, kind: str = "neutral", bg: Optional[str] = None):
        soft, strong = T.KINDS[kind]
        super().__init__(parent, text=text, font=Fonts.micro_b, bg=soft, fg=strong, padx=8, pady=2)

    def update_chip(self, text: str, kind: str):
        soft, strong = T.KINDS[kind]
        self.configure(text=text, bg=soft, fg=strong)


class Card(tk.Frame):
    """White card with hairline border, optional title row (with a right-aligned slot) and padded body."""

    def __init__(self, parent, title: Optional[str] = None, accent: Optional[str] = None):
        super().__init__(parent, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        if accent:
            tk.Frame(self, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
        inner = tk.Frame(self, bg=T.SURFACE)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14, pady=12)
        self.header = tk.Frame(inner, bg=T.SURFACE)
        if title:
            self.header.pack(fill=tk.X, pady=(0, 8))
            tk.Label(self.header, text=title.upper(), font=Fonts.micro_b, bg=T.SURFACE, fg=T.MUTED).pack(side=tk.LEFT)
        self.body = tk.Frame(inner, bg=T.SURFACE)
        self.body.pack(fill=tk.BOTH, expand=True)


def kv_rows(parent, rows: List[Tuple[str, str]], bg=T.SURFACE, value_fonts=None):
    """Label (muted, left) / value (bold, right) rows."""
    f = tk.Frame(parent, bg=bg)
    f.pack(fill=tk.X)
    f.columnconfigure(1, weight=1)
    for i, (k, v) in enumerate(rows):
        tk.Label(f, text=k, font=Fonts.small, bg=bg, fg=T.MUTED, anchor="w").grid(row=i, column=0, sticky="w", pady=2)
        tk.Label(f, text=v, font=Fonts.small_b, bg=bg, fg=T.TEXT, anchor="e", justify="right").grid(
            row=i, column=1, sticky="e", pady=2, padx=(12, 0))
    return f


class BarGauge(tk.Canvas):
    """Horizontal 0-100 gauge with optional tick labels (e.g. risk cut-offs)."""

    def __init__(self, parent, value: float, color: str, ticks: Sequence[Tuple[float, str]] = (), height: int = 34):
        super().__init__(parent, height=height, bg=T.SURFACE, highlightthickness=0)
        self.value, self.color, self.ticks = max(0.0, min(100.0, value)), color, ticks
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 10)
        y0, y1 = 4, 12
        self.create_rectangle(0, y0, w, y1, fill=T.DIVIDER, outline="")
        self.create_rectangle(0, y0, w * self.value / 100.0, y1, fill=self.color, outline="")
        for v, label in self.ticks:
            x = w * v / 100.0
            self.create_line(x, y0 - 2, x, y1 + 2, fill=T.FAINT)
            self.create_text(min(max(x, 12), w - 12), y1 + 11, text=label, font=Fonts.micro, fill=T.MUTED)


class Collapsible(tk.Frame):
    def __init__(self, parent, title: str, expanded: bool = False):
        super().__init__(parent, bg=T.SURFACE)
        self._title, self._open = title, expanded
        self.head = tk.Label(self, font=Fonts.micro_b, bg=T.SURFACE, fg=T.MUTED, anchor="w", cursor="hand2")
        self.head.pack(fill=tk.X)
        self.head.bind("<Button-1>", lambda e: self.toggle())
        self.body = tk.Frame(self, bg=T.SURFACE)
        self._sync()

    def toggle(self):
        self._open = not self._open
        self._sync()

    def _sync(self):
        self.head.configure(text=f"{'▾' if self._open else '▸'}  {self._title.upper()}")
        if self._open:
            self.body.pack(fill=tk.X, pady=(8, 0))
        else:
            self.body.pack_forget()


class Tooltip:
    def __init__(self, widget, text: str, delay: int = 450):
        self.widget, self.text, self.delay, self._tip, self._job = widget, text, delay, None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _):
        self._job = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip or not self.text:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        x, y = self.widget.winfo_rootx() + 12, self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, font=Fonts.small, bg=T.NAV_BG, fg=T.NAV_FG, padx=8, pady=4,
                 justify="left", wraplength=320).pack()

    def _hide(self, _=None):
        if self._job:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame; content width follows the available width."""

    def __init__(self, parent, bg=T.APP_BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.inner.bind("<Configure>", lambda e: self._update_bar())
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)
        self._update_bar()

    def _update_bar(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.after_idle(self._toggle_bar)

    def _toggle_bar(self):
        try:
            need = self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1
            if need and not self.bar.winfo_ismapped():
                self.bar.pack(side=tk.RIGHT, fill=tk.Y, before=self.canvas)
            elif not need and self.bar.winfo_ismapped():
                self.bar.pack_forget()
                self.canvas.yview_moveto(0)
        except tk.TclError:
            pass

    def _bind_wheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _unbind_wheel(self, _):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(ev)

    def _wheel(self, e):
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-e.delta / 120) * 2 if e.delta else 0, "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.canvas.yview_moveto(0)
