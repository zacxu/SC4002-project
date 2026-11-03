"""Holds off idle sleep while training logs are active, then exits by itself.

Uses the per-process Windows request SetThreadExecutionState, the same mechanism media players use.
It changes no power settings, and the request disappears when this process ends. A closed lid or a
manual Sleep still sleeps. Stops when no log under results/ has changed for IDLE_MIN minutes, or after
MAX_HOURS at the latest.
"""
import ctypes
import sys
import time
from pathlib import Path

IDLE_MIN = 30
MAX_HOURS = 14
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

results = Path(sys.argv[1])
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
start = time.time()
print("holding off idle sleep while training logs are active", flush=True)
while time.time() - start < MAX_HOURS * 3600:
    time.sleep(60)
    newest = max((p.stat().st_mtime for p in results.glob("*.log")), default=0)
    if time.time() - newest > IDLE_MIN * 60:
        print(f"no log activity for {IDLE_MIN} minutes, releasing", flush=True)
        break
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("released", flush=True)
