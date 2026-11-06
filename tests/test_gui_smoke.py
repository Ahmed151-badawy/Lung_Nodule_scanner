"""GUI smoke test: builds the real window and exercises seed / pan / zoom / validation.
Skipped automatically when there is no display (CI) or Tk is unavailable."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import tkinter as tk
    root = tk.Tk(); root.withdraw(); root.destroy()
    HAVE_DISPLAY = True
except Exception:
    HAVE_DISPLAY = False


@unittest.skipUnless(HAVE_DISPLAY, "needs a display")
class TestGuiSmoke(unittest.TestCase):
    def setUp(self):
        import numpy as np, cv2, tkinter as tk
        import gui
        from utils.image_utils import apply_lung_window
        self.root = tk.Tk()
        self.root.geometry("1280x800+0+0")
        self.app = gui.NoduleWorkstation(self.root)
        hu = np.full((512, 512), -800.0, np.float32)
        cv2.circle(hu, (160, 230), 12, 60.0, -1)
        self.app.current_image = cv2.cvtColor(apply_lung_window(hu), cv2.COLOR_GRAY2BGR)
        self.app.current_image_path = "x.dcm"
        self.app.image_metadata = {"pixel_spacing": (0.75, 0.75), "hu_slice": hu}
        self.app.viewer.set_spacing((0.75, 0.75))
        self.app.viewer.set_image(self.app.current_image)
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_click_sets_seed_and_drag_does_not(self):
        v = self.app.viewer
        cx, cy = v._to_canvas(160.5, 230.5)
        v.event_generate("<ButtonPress-1>", x=round(cx), y=round(cy))
        v.event_generate("<ButtonRelease-1>", x=round(cx), y=round(cy))
        self.root.update()
        self.assertEqual(self.app.seed, (160, 230))
        v.event_generate("<ButtonPress-1>", x=300, y=300)
        v.event_generate("<B1-Motion>", x=350, y=340)
        v.event_generate("<ButtonRelease-1>", x=350, y=340)
        self.root.update()
        self.assertEqual(self.app.seed, (160, 230))      # a pan must not move the seed
        self.app._clear_seed()
        self.assertIsNone(self.app.seed)

    def test_brock_inputs_validation(self):
        a = self.app
        self.assertEqual(a._read_risk_inputs(), {})
        a.var_age.set("64"); a.var_sex.set("Female"); a.var_lobe.set("Upper")
        self.assertEqual(a._read_risk_inputs()["age"], 64)
        a.var_age.set("7")
        self.assertEqual(a._read_risk_inputs(), {})        # implausible age -> Brock not computed

    def test_saliency_blocked_before_analysis(self):
        self.app._set_view_mode("saliency")
        self.assertEqual(self.app.view_mode, "ct")


if __name__ == "__main__":
    unittest.main()
