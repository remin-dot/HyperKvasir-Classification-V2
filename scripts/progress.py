"""Live download progress for the HyperKvasir archives.

    python scripts/progress.py            # refresh until everything completes
    python scripts/progress.py --once     # print one snapshot and exit

Rate and ETA are measured over a rolling window rather than since start, so a
stalled connection shows up within seconds instead of being hidden behind a
healthy average. That matters on this link: throughput has ranged from 2 KB/s
to 650 KB/s in a single session.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config  # noqa: E402

WINDOW_SECONDS = 30
BAR_WIDTH = 34


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _hms(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds > 400 * 3600:
        return "--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _bar(frac: float, width: int = BAR_WIDTH) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    # Eighth-block partial cell, so slow progress is still visibly moving.
    partial = ""
    if filled < width:
        eighth = int((frac * width - filled) * 8)
        partial = " ▏▎▍▌▋▊▉"[eighth] if eighth else " "
    return "█" * filled + partial + " " * max(0, width - filled - len(partial))


def targets(cfg: dict) -> list[tuple[str, Path, int]]:
    from scripts.download_dataset import COMPONENT_TARGETS

    out = []
    for name, spec in cfg["dataset"]["components"].items():
        if name == "labeled":
            continue                      # V.1 already has it
        zip_name = COMPONENT_TARGETS[name][0]
        out.append((name, cfg["paths"]["archives"] / zip_name,
                    int(spec["size_gb"] * 1024 ** 3)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one snapshot, no refresh")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    items = targets(cfg)
    history: dict[str, deque] = {name: deque() for name, _, _ in items}
    lines_drawn = 0

    if args.once:
        # A rate needs two samples. Without this, --once reports every transfer
        # as stalled no matter how fast it is actually moving.
        for name, path, _ in items:
            history[name].append((time.monotonic(),
                                  path.stat().st_size if path.exists() else 0))
        time.sleep(2.0)

    while True:
        now = time.monotonic()
        rows = []
        all_done = True

        for name, path, expected in items:
            size = path.stat().st_size if path.exists() else 0
            hist = history[name]
            hist.append((now, size))
            while len(hist) > 2 and now - hist[0][0] > WINDOW_SECONDS:
                hist.popleft()

            elapsed = now - hist[0][0]
            rate = (size - hist[0][1]) / elapsed if elapsed > 1.5 else 0.0
            frac = size / expected if expected else 0.0
            done = size >= expected * 0.95
            all_done &= done

            if done:
                status, eta = "complete", "     "
            elif rate > 1024:
                status, eta = f"{_human(rate)}/s", _hms((expected - size) / rate)
            elif size == 0:
                status, eta = "queued", "     "
            else:
                # Archives download one at a time, so a partial file with no
                # movement is usually just waiting its turn rather than broken.
                # "idle" covers both without crying wolf.
                status, eta = "idle", "     "

            rows.append(f"  {name:<10} |{_bar(frac)}| {frac * 100:5.1f}%  "
                        f"{_human(size):>9} / {_human(expected):<9} "
                        f"{status:>11}  {eta:>7}")

        width = shutil.get_terminal_size((100, 20)).columns
        header = "HyperKvasir V.2 — archive downloads"
        body = [header, "-" * min(width - 1, 92)] + rows + ["-" * min(width - 1, 92)]

        total_have = sum(p.stat().st_size if p.exists() else 0 for _, p, _ in items)
        total_want = sum(e for _, _, e in items)
        body.append(f"  {'TOTAL':<10} |{_bar(total_have / total_want)}| "
                    f"{total_have / total_want * 100:5.1f}%  "
                    f"{_human(total_have):>9} / {_human(total_want):<9}")

        if args.once:
            print("\n".join(body))
            return

        if lines_drawn:
            sys.stdout.write(f"\033[{lines_drawn}A")
        sys.stdout.write("\n".join(line.ljust(width - 1)[:width - 1] for line in body) + "\n")
        sys.stdout.flush()
        lines_drawn = len(body)

        if all_done:
            print("\nAll archives complete.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
