#!/usr/bin/env python3
"""Keep the Osteon dashboard + permanent public tunnel alive.

Supervises two processes and restarts whichever dies:
  1. the Flask app on 127.0.0.1:5001
  2. an ngrok tunnel pinned to a reserved STATIC domain (so the public URL never rotates)

The static domain comes from the OSTEON_NGROK_DOMAIN env var (e.g. "osteon.ngrok-free.app").
The ngrok authtoken is read from ngrok's own config (~/Library/Application Support/ngrok),
so no secret lives in this repo. The current public URL is written to webapp/PUBLIC_URL.txt.

Run detached:

    OSTEON_NGROK_DOMAIN=<your-domain> \
      nohup .venv/bin/python scripts/keepalive.py > /tmp/osteon_keepalive.log 2>&1 &
"""
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
NGROK = "/opt/homebrew/bin/ngrok"
PORT = 5001
DOMAIN = os.environ.get("OSTEON_NGROK_DOMAIN", "").strip()
PUBLIC_URL = f"https://{DOMAIN}" if DOMAIN else None
URL_FILE = ROOT / "webapp" / "PUBLIC_URL.txt"
TUN_LOG = Path("/tmp/osteon_tunnel.log")
APP_LOG = Path("/tmp/osteon_web.log")

ENV = {**os.environ, "OSTEON_BLENDER": "/Applications/Blender.app/Contents/MacOS/Blender"}
app_proc = None
tun_proc = None


def app_up():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_app():
    global app_proc
    print("[keepalive] starting Flask app", flush=True)
    app_proc = subprocess.Popen([PY, "webapp/app.py"], cwd=ROOT, env=ENV,
                                stdout=open(APP_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(30):
        if app_up():
            return
        time.sleep(1)


def start_tunnel():
    global tun_proc
    print(f"[keepalive] starting ngrok tunnel -> {PUBLIC_URL}", flush=True)
    TUN_LOG.write_text("")
    tun_proc = subprocess.Popen(
        [NGROK, "http", str(PORT), f"--url={PUBLIC_URL}", "--log=stdout"],
        stdout=open(TUN_LOG, "w"), stderr=subprocess.STDOUT,
    )
    # Pinned domain is deterministic; just confirm the agent accepted it.
    for _ in range(40):
        if public_ok(PUBLIC_URL):
            URL_FILE.write_text(PUBLIC_URL + "\n")
            print(f"[keepalive] PUBLIC URL: {PUBLIC_URL}", flush=True)
            return PUBLIC_URL
        time.sleep(1)
    print("[keepalive] WARNING: tunnel did not come up in time", flush=True)
    return PUBLIC_URL


def public_ok(url):
    if not url:
        return False
    try:
        req = urllib.request.Request(url + "/", method="GET",
                                     headers={"ngrok-skip-browser-warning": "1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    if not DOMAIN:
        raise SystemExit("OSTEON_NGROK_DOMAIN is required (your reserved ngrok static domain).")
    start_app()
    url = start_tunnel()
    URL_FILE.write_text(url + "\n")
    next_public_check = time.time() + 60
    fails = 0
    while True:
        if not app_up():
            print("[keepalive] app down — restarting", flush=True)
            try:
                app_proc and app_proc.kill()
            except Exception:
                pass
            start_app()
        # tunnel process dead -> restart (same static URL)
        if tun_proc is None or tun_proc.poll() is not None:
            print("[keepalive] tunnel process exited — restarting", flush=True)
            start_tunnel()
            next_public_check = time.time() + 60
            fails = 0
        else:
            now = time.time()
            if now >= next_public_check:
                next_public_check = now + 60
                if public_ok(PUBLIC_URL):
                    fails = 0
                else:
                    fails += 1
                    print(f"[keepalive] public URL check failed ({fails}/3)", flush=True)
                    if fails >= 3:
                        print("[keepalive] recycling tunnel", flush=True)
                        try:
                            tun_proc.kill()
                        except Exception:
                            pass
                        start_tunnel()
                        next_public_check = time.time() + 60
                        fails = 0
        time.sleep(15)


if __name__ == "__main__":
    main()
