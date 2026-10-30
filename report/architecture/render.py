"""Render the architecture figure and the DFA figure at very high resolution, plus vector (PDF/SVG) copies.

Setup (once):  cd report/architecture && npm install && pip install playwright
Run:           python render.py

Writes ../figures/architecture.png (4x device scale = 6400 px wide), ../figures/dfa_min.png (3x), and
../figures/vector/architecture.pdf, dfa_min.svg, dfa_min.pdf. Chromium comes from the Playwright install.
"""
import os
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
VEC = FIG / "vector"
VEC.mkdir(parents=True, exist_ok=True)
SCALE = 4        # architecture: 6400 px wide, far beyond print resolution at page width
DFA_SCALE = 3
CHROMIUM = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium")
launch = {"args": ["--no-sandbox"]}
if Path(CHROMIUM).exists():
    launch["executable_path"] = CHROMIUM

# DFA: Graphviz (WebAssembly) turns the committed .dot into SVG
dot = HERE.parent.parent / "results" / "module2" / "dfa_min.dot"
subprocess.run(["node", str(HERE / "dot2svg.mjs"), str(dot), str(VEC / "dfa_min.svg")], check=True, cwd=HERE)
svg = (VEC / "dfa_min.svg").read_text(encoding="utf-8")
dfa_page = HERE / "_dfa.html"
dfa_page.write_text(f"<!doctype html><html><body style='margin:0;background:#fff'>{svg}</body></html>", encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(**launch)

    # architecture: fixed 1600 x 720 canvas
    page = browser.new_page(viewport={"width": 1600, "height": 720}, device_scale_factor=SCALE)
    page.goto("file://" + str(HERE / "architecture.html"))
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(800)
    page.screenshot(path=str(FIG / "architecture.png"))
    page.pdf(path=str(VEC / "architecture.pdf"), width="1600px", height="720px", print_background=True, page_ranges="1")

    # DFA: size the viewport to the drawing
    page = browser.new_page(viewport={"width": 800, "height": 400}, device_scale_factor=DFA_SCALE)
    page.goto("file://" + str(dfa_page))
    box = page.evaluate("(() => { const s = document.querySelector('svg').getBoundingClientRect(); return [s.width, s.height]; })()")
    w, h = int(box[0]) + 1, int(box[1]) + 1
    page.set_viewport_size({"width": w, "height": h})
    page.screenshot(path=str(FIG / "dfa_min.png"), clip={"x": 0, "y": 0, "width": w, "height": h})
    page.pdf(path=str(VEC / "dfa_min.pdf"), width=f"{w}px", height=f"{h}px", print_background=True, page_ranges="1")
    browser.close()
dfa_page.unlink()
print("done")
