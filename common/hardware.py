"""Hardware detection. Never assumes a GPU exists -- it verifies, then adapts.

Produces hardware_info.txt, which doubles as the reproducibility record because
it also captures the resolved versions of every relevant package.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PACKAGES = [
    "torch", "torchvision", "ultralytics", "timm", "grad-cam",
    "scikit-learn", "numpy", "pandas", "matplotlib", "opencv-python",
    "Pillow", "PyYAML", "psutil",
]


def _nvidia_smi() -> dict | None:
    """Query nvidia-smi directly. Returns None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, total, free, driver, cc = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "name": name,
            "vram_total_mb": int(float(total)),
            "vram_free_mb": int(float(free)),
            "driver_version": driver,
            "compute_capability": cc,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["device_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def _package_versions() -> dict:
    """Resolved versions of the packages that affect results."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return {}
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def detect(data_root: Path | None = None) -> dict:
    """Full hardware + software inventory."""
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
        cpu_physical = psutil.cpu_count(logical=False)
    except ImportError:
        ram_gb, cpu_physical = None, None

    disk_target = Path(data_root) if data_root else Path.cwd()
    # Walk up to the first existing ancestor -- data_root may not exist yet.
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    usage = shutil.disk_usage(disk_target)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": cpu_physical,
        "cpu_threads": os.cpu_count(),
        "ram_gb": ram_gb,
        "disk_checked": str(disk_target),
        "disk_free_gb": round(usage.free / 1024**3, 1),
        "disk_total_gb": round(usage.total / 1024**3, 1),
        "gpu": _nvidia_smi(),
        "torch": _torch_info(),
        "packages": _package_versions(),
    }


def gpu_available(info: dict | None = None) -> bool:
    """True only if torch can actually use CUDA. An NVIDIA card with a CPU-only
    torch build is NOT usable, and that is the common Windows failure mode."""
    info = info or detect()
    return bool(info["torch"].get("cuda_available"))


def usable_vram_gb(info: dict | None = None) -> float:
    info = info or detect()
    if info["gpu"]:
        return info["gpu"]["vram_free_mb"] / 1024
    return 0.0


def pick_batch_size(configured: int, vram_gb: float, has_gpu: bool) -> int:
    """Scale the configured batch size to whatever VRAM is actually free.

    ponytail: a lookup table, not an auto-tuner. Ultralytics has batch=-1 autobatch
    if you ever need the fancy version.
    """
    if not has_gpu:
        return max(8, configured // 4)     # CPU: keep memory sane, speed is hopeless anyway
    if vram_gb >= 10:
        return configured
    if vram_gb >= 7:
        return configured
    if vram_gb >= 5:
        return max(8, configured // 2)     # your 4070 with a browser open lands here
    return max(4, configured // 4)


def setup_torch(seed: int = 42):
    """Enable the fast paths this GPU supports and return the device."""
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("\n" + "!" * 70)
        print("!! NO CUDA GPU AVAILABLE -- training on CPU.")
        print("!! Expect roughly 20-40x the times quoted in the README.")
        print("!! If you have an NVIDIA GPU, you likely installed the CPU-only torch build:")
        print("!!   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("!" * 70 + "\n")
    return device


def amp_dtype():
    """bf16 where supported (no loss scaling needed), else fp16, else None."""
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def format_report(info: dict) -> str:
    lines = ["=" * 70, "HARDWARE & ENVIRONMENT REPORT", "=" * 70,
             f"Generated : {info['timestamp']}", ""]

    lines += ["--- SYSTEM " + "-" * 59,
              f"Platform        : {info['platform']}",
              f"CPU             : {info['cpu']}",
              f"CPU cores       : {info['cpu_cores_physical']} physical / {info['cpu_threads']} logical",
              f"RAM             : {info['ram_gb']} GB",
              f"Disk ({info['disk_checked']}) : {info['disk_free_gb']} GB free of {info['disk_total_gb']} GB",
              f"Python          : {info['python']}",
              f"Interpreter     : {info['python_executable']}", ""]

    lines += ["--- GPU " + "-" * 62]
    if info["gpu"]:
        g = info["gpu"]
        lines += [f"Model           : {g['name']}",
                  f"VRAM            : {g['vram_total_mb']} MiB total / {g['vram_free_mb']} MiB free",
                  f"Driver          : {g['driver_version']}",
                  f"Compute cap.    : {g['compute_capability']}"]
    else:
        lines += ["No NVIDIA GPU detected by nvidia-smi."]
    lines += [""]

    lines += ["--- PYTORCH " + "-" * 58]
    t = info["torch"]
    if not t.get("installed"):
        lines += ["PyTorch is NOT installed. See README installation section."]
    else:
        lines += [f"torch           : {t['version']}",
                  f"CUDA available  : {t['cuda_available']}",
                  f"CUDA version    : {t['cuda_version']}",
                  f"cuDNN           : {t['cudnn_version']}",
                  f"Devices         : {t['device_count']}"]
        if t["cuda_available"]:
            lines += [f"Active device   : {t['device_name']} ({t['device_vram_gb']} GB)",
                      f"bfloat16        : {t['bf16_supported']}"]
    lines += [""]

    lines += ["--- VERDICT " + "-" * 58]
    if t.get("cuda_available"):
        lines += ["GPU training ENABLED. Mixed precision on, batch size auto-scaled to free VRAM."]
    elif info["gpu"]:
        lines += ["GPU present but PyTorch cannot use it -- almost certainly a CPU-only torch build.",
                  "Reinstall: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"]
    else:
        lines += ["No usable GPU. Pipeline will run on CPU (very slow but functional)."]
    lines += [""]

    lines += ["--- PACKAGE VERSIONS (reproducibility record) " + "-" * 24]
    for pkg, ver in info["packages"].items():
        lines.append(f"{pkg:<28}{ver}")
    lines += ["", "=" * 70]
    return "\n".join(lines)


def write_report(cfg: dict) -> dict:
    info = detect(cfg["paths"]["data_root"])
    out = cfg["paths"]["project_root"] / "hardware_info.txt"
    report = format_report(info)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"-> written to {out}")

    import json
    (cfg["paths"]["results"] / "hardware_info.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


if __name__ == "__main__":
    from common.config import ensure_dirs, load_config
    c = load_config()
    ensure_dirs(c)
    write_report(c)
