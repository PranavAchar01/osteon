#!/usr/bin/env python3
"""Keep the Osteon dashboard + public tunnel alive.

Supervises two processes and restarts whichever dies:
  1. the Flask app on 127.0.0.1:5001
  2. a cloudflared quick tunnel exposing it publicly

The current public URL is written to webapp/PUBLIC_URL.txt (and printed) so it is always
discoverable even if cloudflared reconnects with a new hostname. Run detached:

    nohup .venv/bin/python scripts/keepalive.py > /tmp/osteon_keepalive.log 2>&1 &
"""
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
CLOUDFLARED = "/opt/homebrew/bin/cloudflared"
PORT = 5001
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


def tunnel_url():
    try:
        m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", TUN_LOG.read_text())
        return m[-1] if m else None
    except Exception:
        return None


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
    print("[keepalive] starting cloudflared tunnel", flush=True)
    TUN_LOG.write_text("")
    tun_proc = subprocess.Popen([CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
                                stdout=open(TUN_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(40):
        u = tunnel_url()
        if u:
            URL_FILE.write_text(u + "\n")
            print(f"[keepalive] PUBLIC URL: {u}", flush=True)
            return u
        time.sleep(1)
    return None


def public_ok(url):
    if not url:
        return False
    try:
        req = urllib.request.Request(url + "/", method="GET")
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    start_app()
    url = start_tunnel()
    next_public_check = time.time() + 120   # grace: let the hostname propagate before checking
    fails = 0
    while True:
        if not app_up():
            print("[keepalive] app down — restarting", flush=True)
            try:
                app_proc and app_proc.kill()
            except Exception:
                pass
            start_app()
        # tunnel process dead -> restart (new URL)
        if tun_proc is None or tun_proc.poll() is not None:
            print("[keepalive] tunnel process exited — restarting", flush=True)
            url = start_tunnel()
            next_public_check = time.time() + 120
            fails = 0
        else:
            now = time.time()
            if now >= next_public_check:
                next_public_check = now + 60
                url = url or tunnel_url()
                if public_ok(url):
                    fails = 0
                else:
                    fails += 1
                    print(f"[keepalive] public URL check failed ({fails}/3)", flush=True)
                    if fails >= 3:   # only recycle after ~3 min of real downtime
                        print("[keepalive] recycling tunnel", flush=True)
                        try:
                            tun_proc.kill()
                        except Exception:
                            pass
                        url = start_tunnel()
                        next_public_check = time.time() + 120
                        fails = 0
        time.sleep(15)


if __name__ == "__main__":
    main()
