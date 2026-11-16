"""README screenshots from a demo instance (synthetic audio, simulated model).

    python users.py add demo                                  # with USERS_FILE pointing at a scratch file
    DEMO_MODE=true DEMO_DELAY=3 PARALLEL_CHUNKS=1 PORT=5050 python app.py &
    python scripts/screenshots.py http://127.0.0.1:5050 demo <password>
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
URL, USER, PASSWORD = (sys.argv[1:4] + [None] * 3)[:3]
URL = URL or "http://127.0.0.1:5050"

HISTORY = [  # (title, minutes) — fictional recordings, oldest first
    ("Formation sécurité chantier", 12),
    ("Entretien client — cahier des charges", 22),
    ("Point équipe du lundi", 6),
]
MAIN = ("Réunion de chantier — semaine 11", 40)


def make_audio(path: Path, minutes: float):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=220:duration={int(minutes * 60)}", "-af", "volume=0.2",
                    "-c:a", "aac", "-b:a", "24k", str(path)], check=True)


def upload(request, path: Path, language=""):
    r = request.post(URL + "/jobs", multipart={
        "audio": {"name": path.name, "mimeType": "audio/mp4", "buffer": path.read_bytes()}, "language": language})
    assert r.status == 202, r.text()
    return r.json()["job_id"]


def wait_done(request, job_id):
    while request.get(f"{URL}/jobs/{job_id}").json()["state"] not in ("done", "failed"):
        time.sleep(0.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    creds = {"username": USER, "password": PASSWORD} if USER else None
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        tmp = Path(tmp)
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1360, "height": 860}, device_scale_factor=2,
                                  http_credentials=creds, color_scheme="light", locale="fr-FR")
        api = ctx.request

        # 1. History: a few finished jobs, then the main one
        for title, minutes in HISTORY:
            f = tmp / f"{title}.m4a"
            make_audio(f, minutes)
            wait_done(api, upload(api, f, "Français"))
        main_file = tmp / f"{MAIN[0]}.m4a"
        make_audio(main_file, MAIN[1])
        main_job = upload(api, main_file, "Français et anglais")
        wait_done(api, main_job)

        page = ctx.new_page()
        page.goto(f"{URL}/#job={main_job}")
        page.wait_for_selector("#tcard:not([hidden])")
        page.wait_for_timeout(500)
        page.screenshot(path=OUT / "transcript.png")

        # 2. Find in transcript
        page.fill("#tsearch", "vendredi")
        page.wait_for_timeout(900)
        page.screenshot(path=OUT / "search.png")

        # 3. Upload form with a file chosen
        page.click("#newBtn")
        upfile = tmp / "Visite de site — zone nord.m4a"
        make_audio(upfile, 35)
        page.set_input_files("#file", str(upfile))
        page.fill("#lang", "Hassaniya et français")
        page.wait_for_timeout(300)
        page.screenshot(path=OUT / "upload.png")

        # 4. Job in progress (3 parts, one at a time)
        page.click("#go")
        page.wait_for_function("document.getElementById('jtext').textContent.startsWith('Part 2 of 3')", timeout=120000)
        page.wait_for_timeout(400)
        page.screenshot(path=OUT / "progress.png")

        # 5. Dark mode, same transcript
        dark = browser.new_context(viewport={"width": 1360, "height": 860}, device_scale_factor=2,
                                   http_credentials=creds, color_scheme="dark")
        dpage = dark.new_page()
        dpage.goto(f"{URL}/#job={main_job}")
        dpage.wait_for_selector("#tcard:not([hidden])")
        dpage.wait_for_timeout(500)
        dpage.screenshot(path=OUT / "dark.png")

        # 6. Mobile
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3,
                                     is_mobile=True, http_credentials=creds)
        mpage = mobile.new_page()
        mpage.goto(f"{URL}/#job={main_job}")
        mpage.wait_for_selector("#tcard:not([hidden])")
        mpage.wait_for_timeout(500)
        mpage.screenshot(path=OUT / "mobile.png")
        mpage.click("#menuBtn")
        mpage.wait_for_timeout(400)
        mpage.screenshot(path=OUT / "mobile_history.png")

        page.wait_for_selector("#tcard:not([hidden])", timeout=120000)  # let the last job finish
        browser.close()
    print("Saved:", *sorted(x.name for x in OUT.iterdir()), sep="\n  ")


if __name__ == "__main__":
    main()
