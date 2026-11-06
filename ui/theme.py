"""Design tokens + ttk styling. One accent colour, neutral slate surfaces, dark image viewer."""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


class T:
    # surfaces
    APP_BG = "#F3F5F8"
    SURFACE = "#FFFFFF"
    SURFACE_ALT = "#F8FAFC"
    BORDER = "#E2E8F0"
    DIVIDER = "#EDF1F5"
    # top bar
    NAV_BG = "#0F172A"
    NAV_FG = "#F1F5F9"
    NAV_MUTED = "#94A3B8"
    # text
    TEXT = "#0F172A"
    MUTED = "#64748B"
    FAINT = "#94A3B8"
    # accent
    ACCENT = "#0E7490"
    ACCENT_HOVER = "#0B5E75"
    ACCENT_SOFT = "#E0F2F7"
    # semantic
    OK, OK_SOFT = "#15803D", "#DCFCE7"
    WARN, WARN_SOFT = "#B45309", "#FEF3C7"
    DANGER, DANGER_SOFT = "#B91C1C", "#FEE2E2"
    INFO, INFO_SOFT = "#1D4ED8", "#DBEAFE"
    # viewer
    VIEW_BG = "#0A0E13"
    VIEW_FG = "#CBD5E1"
    VIEW_MUTED = "#64748B"
    OVERLAY = "#22D3EE"
    SEED = "#FACC15"

    KINDS = {  # chip colours: (soft background, strong foreground)
        "ok": (OK_SOFT, OK), "warn": (WARN_SOFT, WARN), "danger": (DANGER_SOFT, DANGER),
        "info": (INFO_SOFT, INFO), "accent": (ACCENT_SOFT, ACCENT), "neutral": ("#EEF2F6", MUTED),
    }


_SANS = ["Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Cantarell", "DejaVu Sans", "Arial"]
_MONO = ["Cascadia Mono", "Consolas", "SF Mono", "Menlo", "JetBrains Mono", "DejaVu Sans Mono", "Courier New"]


class Fonts:
    """Named fonts, created after the Tk root exists. First installed family from each list wins."""
    ready = False

    @classmethod
    def init(cls, root: tk.Misc) -> None:
        if cls.ready:
            return
        fams = set(tkfont.families(root))
        sans = next((f for f in _SANS if f in fams), "TkDefaultFont")
        mono = next((f for f in _MONO if f in fams), "TkFixedFont")
        mk = lambda fam, size, weight="normal": tkfont.Font(root=root, family=fam, size=size, weight=weight)
        cls.family = sans
        cls.title = mk(sans, 13, "bold")
        cls.h2 = mk(sans, 10, "bold")
        cls.body = mk(sans, 10)
        cls.body_b = mk(sans, 10, "bold")
        cls.small = mk(sans, 9)
        cls.small_b = mk(sans, 9, "bold")
        cls.micro = mk(sans, 8)
        cls.micro_b = mk(sans, 8, "bold")
        cls.num_xl = mk(sans, 28, "bold")
        cls.num_l = mk(sans, 20, "bold")
        cls.mono = mk(mono, 9)
        cls.ready = True


def setup_ttk(root: tk.Misc) -> None:
    """Style the ttk widgets we use (entries, combobox, checkbutton, scrollbar, progressbar)."""
    Fonts.init(root)
    st = ttk.Style(root)
    st.theme_use("clam")
    field = dict(fieldbackground=T.SURFACE, background=T.SURFACE, foreground=T.TEXT,
                 bordercolor=T.BORDER, lightcolor=T.BORDER, darkcolor=T.BORDER, padding=(8, 5))
    st.configure("TEntry", **field)
    st.map("TEntry", bordercolor=[("focus", T.ACCENT)], lightcolor=[("focus", T.ACCENT)], darkcolor=[("focus", T.ACCENT)])
    st.configure("TSpinbox", arrowsize=12, **field)
    st.map("TSpinbox", bordercolor=[("focus", T.ACCENT)], lightcolor=[("focus", T.ACCENT)], darkcolor=[("focus", T.ACCENT)])
    st.configure("TCombobox", arrowsize=14, arrowcolor=T.MUTED, selectbackground=T.SURFACE, selectforeground=T.TEXT, **field)
    st.map("TCombobox", fieldbackground=[("readonly", T.SURFACE)], bordercolor=[("focus", T.ACCENT)],
           lightcolor=[("focus", T.ACCENT)], darkcolor=[("focus", T.ACCENT)])
    root.option_add("*TCombobox*Listbox.font", Fonts.body)
    root.option_add("*TCombobox*Listbox.selectBackground", T.ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    st.configure("TCheckbutton", background=T.SURFACE, foreground=T.TEXT, focuscolor=T.SURFACE)
    st.map("TCheckbutton", background=[("active", T.SURFACE)], indicatorcolor=[("selected", T.ACCENT), ("!selected", T.SURFACE)])
    st.configure("Vertical.TScrollbar", background=T.BORDER, troughcolor=T.SURFACE, bordercolor=T.SURFACE,
                 lightcolor=T.BORDER, darkcolor=T.BORDER, arrowsize=0, gripcount=0, width=8)
    st.map("Vertical.TScrollbar", background=[("active", T.FAINT)])
    st.configure("Accent.Horizontal.TProgressbar", troughcolor=T.NAV_BG, background=T.OVERLAY,
                 bordercolor=T.NAV_BG, lightcolor=T.OVERLAY, darkcolor=T.OVERLAY, thickness=4)
