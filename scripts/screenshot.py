"""Dev helper: screenshot dashboard pages with a local Chrome.  python scripts/screenshot.py out_dir [page ...]"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"
out = Path(sys.argv[1])
pages = sys.argv[2:] or ["Operator cab"]
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 1150}, device_scale_factor=1)
    pg.goto("http://localhost:8501/", wait_until="networkidle")
    pg.wait_for_selector("text=Operator Copilot", timeout=30000)
    for name in pages:
        pg.get_by_text(name, exact=True).first.click()
        time.sleep(6)
        pg.screenshot(path=str(out / (name.replace(" ", "_").replace("&", "and") + ".png")), full_page=True)
        print("saved", name)
    b.close()
