"""Render architecture_nomod.html (no module numbering) to ../figures/architecture_nomod.png and vector/architecture_nomod.pdf."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
CHROMIUM = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium")
launch = {"args": ["--no-sandbox"]}
if Path(CHROMIUM).exists():
    launch["executable_path"] = CHROMIUM
with sync_playwright() as p:
    b = p.chromium.launch(**launch)
    page = b.new_page(viewport={"width": 1600, "height": 720}, device_scale_factor=4)
    page.goto("file://" + str(HERE / "architecture_nomod.html"))
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(800)
    page.screenshot(path=str(FIG / "architecture_nomod.png"))
    page.pdf(path=str(FIG / "vector" / "architecture_nomod.pdf"), width="1600px", height="720px", print_background=True, page_ranges="1")
    b.close()
print("done")
