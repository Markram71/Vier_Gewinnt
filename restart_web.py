#!/usr/bin/env python3
"""Restart the web server, loading the latest trained model.

Use this to bounce the server and pick up new checkpoints while training is running.

Usage:
    python restart_web.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_APP = PROJECT_ROOT / "web" / "app.py"


def main():
    # Kill any existing web server
    try:
        result = subprocess.run(
            ["pgrep", "-f", "web/app.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split()
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                    print(f"Stopped existing server (PID {pid})")
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
            time.sleep(1)
    except FileNotFoundError:
        # pgrep not available (e.g. Windows)
        pass

    os.chdir(PROJECT_ROOT)
    print("Starting web server...")
    print("Open http://localhost:5000")
    subprocess.run([sys.executable, str(WEB_APP)], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
