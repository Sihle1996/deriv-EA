"""Launch each deriv-bot symbol as a DETACHED background process (Windows), logging to
logs/<symbol>.log. The processes survive closing the terminal. Stop with: python stop_all.py

Robust to the PowerShell `Start-Process` "Item has already been added: COMSPEC" bug — Python's
os.environ is case-insensitive, so it collapses the duplicate COMSPEC/ComSpec key.

IMPORTANT: run with NO foreground `python main.py` active for these symbols — two processes writing
the same data/<symbol>/ folder corrupt the archive. (This guard only catches prior run_all.py runs.)

Edit SYMBOLS to change which assets run.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYMBOLS = [
    "stpRNG", "1HZ50V",                   # synthetics (CSPRNG, 24/7) — edge impossible by construction
    "frxUSDJPY", "frxXAUUSD", "OTC_NDX",  # REAL markets (USD/JPY, Gold/USD, US Tech 100) — edge POSSIBLE
]                                         # real markets have closing hours: weekend/overnight gaps are
                                          # EXPECTED, not failures (see CLAUDE.md). The control vs treatment
                                          # for "does the pattern thrive where order flow exists?"
LOGDIR = ROOT / "logs"
PIDFILE = LOGDIR / "pids.txt"
PY = ROOT / ".venv" / "Scripts" / "pythonw.exe"   # windowless: no console window pops up


def _alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True)
    return str(pid) in r.stdout


def main() -> None:
    LOGDIR.mkdir(exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"venv python not found at {PY} - create/activate the venv first.")

    # Guard: refuse to start if a previous launch is still alive (prevents duplicate writers).
    if PIDFILE.exists():
        still = [ln for ln in PIDFILE.read_text().splitlines()
                 if "=" in ln and _alive(int(ln.split("=")[1]))]
        if still:
            raise SystemExit(f"Already running: {', '.join(still)}. Run: python stop_all.py")

    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    lines = []
    for sym in SYMBOLS:
        env = dict(os.environ)            # case-insensitive on Windows -> no COMSPEC dup
        env["DERIV_SYMBOL"] = sym
        logf = open(LOGDIR / f"{sym}.log", "a", encoding="utf-8")  # child inherits this handle
        p = subprocess.Popen([str(PY), "main.py"], cwd=str(ROOT), env=env,
                             stdout=logf, stderr=subprocess.STDOUT,
                             creationflags=flags, close_fds=True)
        lines.append(f"{sym}={p.pid}")
        print(f"started {sym} (pid {p.pid}) -> logs/{sym}.log")
    PIDFILE.write_text("\n".join(lines) + "\n")
    print("\nWatch a log (PowerShell):  Get-Content logs\\stpRNG.log -Wait -Tail 20")
    print("Stop all:                  python stop_all.py")


if __name__ == "__main__":
    main()
