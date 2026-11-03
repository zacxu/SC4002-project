"""Protects the laptop during long unattended training: pause on heat or low memory.

The machine rebooted uncleanly during an earlier run (GPU shared with another process, little
free RAM), so every run checks free RAM, GPU temperature and free VRAM, and waits instead of
pushing on. If the condition does not clear, it stops cleanly: the best checkpoint is already on
disk and the ladder resumes where it stopped.
"""
import subprocess
import time

import psutil

MIN_RAM_MB = 900            # pause below this much available system RAM
MAX_TEMP_C = 86             # pause at or above this GPU temperature ...
RESUME_TEMP_C = 76          # ... and resume at or below this one
START_TEMP_C = 80           # a new run waits until the GPU has cooled to this
MIN_FREE_VRAM_MB = 1500     # needed before a run allocates its model
MAX_WAIT_S = 3600           # give up (cleanly) after this long
THROTTLE_S = 20             # do not query the hardware more often than this during training

_last_check = 0.0


class ResourceAbort(RuntimeError):
    """Raised when the machine stayed unsafe for MAX_WAIT_S."""


def _gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        temp, used, total = [int(x) for x in out.strip().splitlines()[0].split(",")]
        return temp, total - used
    except Exception:  # no GPU tool available: only the RAM check applies
        return None, None


def _reasons(need_vram, cooling):
    reasons = []
    avail = psutil.virtual_memory().available / 1e6
    if avail < MIN_RAM_MB:
        reasons.append(f"low RAM {avail:.0f} MB")
    temp, free_vram = _gpu()
    if temp is not None:
        limit = RESUME_TEMP_C if cooling else (START_TEMP_C if need_vram else MAX_TEMP_C)
        if (cooling and temp > limit) or (not cooling and temp >= limit):
            reasons.append(f"GPU {temp}C")
        if need_vram and free_vram < MIN_FREE_VRAM_MB:
            reasons.append(f"free VRAM {free_vram} MB")
    return reasons


def check(tag="", need_vram=False, force=False):
    """Blocks while the machine is unsafe. Returns the seconds spent waiting."""
    global _last_check
    now = time.time()
    if not force and not need_vram and now - _last_check < THROTTLE_S:
        return 0.0
    _last_check = now
    reasons = _reasons(need_vram, cooling=False)
    if not reasons:
        return 0.0
    start = time.time()
    print(f"[guard] {tag}: pausing ({', '.join(reasons)})", flush=True)
    cooling = False
    while reasons:
        if time.time() - start > MAX_WAIT_S:
            raise ResourceAbort(f"{tag}: still unsafe after {MAX_WAIT_S}s ({', '.join(reasons)})")
        cooling = cooling or any(r.startswith("GPU") for r in reasons)
        time.sleep(30)
        reasons = _reasons(need_vram, cooling=cooling)
    waited = time.time() - start
    print(f"[guard] {tag}: resuming after {waited:.0f}s", flush=True)
    _last_check = time.time()
    return waited


def peak_ram_mb():
    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 1e6)
