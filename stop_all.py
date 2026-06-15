"""Stop all background deriv-bot instances started by run_all.py.

Hard-stop (Windows TerminateProcess), so up to ~99 buffered ticks and any buffered signals are
lost. But on the next run_all.py each instance BACKFILLS the gap from the tick archive, and
`python backfill_signals.py <symbol>` recovers any missed signals.
"""
import subprocess
from pathlib import Path

PIDFILE = Path(__file__).resolve().parent / "logs" / "pids.txt"


def main() -> None:
    if not PIDFILE.exists():
        print("No logs/pids.txt - nothing to stop.")
        return
    for ln in PIDFILE.read_text().splitlines():
        if "=" not in ln:
            continue
        sym, pid = ln.split("=", 1)
        # taskkill /T kills the whole tree (in case the venv launcher spawned a worker child).
        r = subprocess.run(["taskkill", "/PID", pid.strip(), "/T", "/F"],
                           capture_output=True, text=True)
        print(f"stopped {sym} (pid {pid})" if r.returncode == 0 else f"{sym} (pid {pid}): {r.stderr.strip() or r.stdout.strip()}")
    PIDFILE.unlink()
    print("Done. On next start, each instance backfills the downtime gap automatically.")


if __name__ == "__main__":
    main()
