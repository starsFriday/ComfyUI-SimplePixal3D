#!/usr/bin/env python3
import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
COMFY_ROOT = NODE_DIR.parents[1]
PYTHON = sys.executable
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

PURE_REQUIREMENTS = [
    "pillow>=11.0.0",
    "imageio>=2.37.0",
    "imageio-ffmpeg>=0.6.0",
    "tqdm>=4.67.0",
    "easydict>=1.13",
    "opencv-python-headless>=4.8.0",
    "trimesh[easy]>=4.6.0",
    "transformers>=4.57.0",
    "zstandard>=0.23.0",
    "kornia>=0.8.0",
    "timm>=1.0.0",
    "diffusers>=0.36.0",
    "accelerate>=1.0.0",
    "plyfile>=1.1.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.34.0",
    "pymeshlab>=2023.12",
    "lpips>=0.1.4",
    "ninja>=1.11.0",
    "moderngl>=5.10.0",
    "scipy>=1.14.0",
    "matplotlib>=3.9.0",
    "gradio>=5.0.0",
    "comfy-3d-viewers>=0.2.44",
]

# Keep MoGe pinned to the revision already verified in this ComfyUI environment.
MOGE_SPEC = "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
UTILS3D_WHEEL = "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# Prebuilt Linux wheels for Python 3.12 + torch 2.8.0 + CUDA 12.8.
CUDA_WHEELS = [
    ("flex_gemm", "https://github.com/PozzettiAndrea/flexgemm-wheels/releases/download/cu128-torch280/flex_gemm-0.0.1+cu128torch28-cp312-cp312-linux_x86_64.whl"),
    ("cumesh", "https://github.com/PozzettiAndrea/cumesh-wheels/releases/download/cu128-torch280/cumesh-0.0.1+cu128torch28-cp312-cp312-linux_x86_64.whl"),
    ("nvdiffrast.torch", "https://github.com/PozzettiAndrea/nvdiffrast-full-wheels/releases/download/cu128-torch280/nvdiffrast-0.4.0+cu128torch28-cp312-cp312-linux_x86_64.whl"),
    ("nvdiffrec_render", "https://github.com/PozzettiAndrea/nvdiffrec_render-wheels/releases/download/cu128-torch280/nvdiffrec_render-0.0.1+cu128torch28-cp312-cp312-linux_x86_64.whl"),
    ("o_voxel", "https://github.com/PozzettiAndrea/ovoxel-wheels/releases/download/cu128-torch280/o_voxel-0.0.1+cu128torch28-cp312-cp312-linux_x86_64.whl"),
]

# NATTEN official wheel index documents torch==2.8.0+cu128 as:
# pip install natten==0.21.1+torch280cu128 -f https://whl.natten.org
NATTEN_SPEC = "natten==0.21.1+torch280cu128"
NATTEN_LINKS = "https://whl.natten.org"
FLASH_ATTN_SPEC = "flash-attn==2.7.3"

CHECK_IMPORTS = [
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("cv2", "opencv"),
    ("imageio", "imageio"),
    ("tqdm", "tqdm"),
    ("easydict", "easydict"),
    ("trimesh", "trimesh"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("accelerate", "accelerate"),
    ("safetensors.torch", "safetensors"),
    ("kornia", "kornia"),
    ("timm", "timm"),
    ("plyfile", "plyfile"),
    ("pymeshlab", "pymeshlab"),
    ("lpips", "lpips"),
    ("moge.model.v2", "moge"),
    ("utils3d", "utils3d"),
    ("natten", "NATTEN"),
    ("flash_attn", "flash-attn"),
    ("flex_gemm", "flex_gemm"),
    ("flex_gemm.ops.grid_sample", "flex_gemm grid_sample"),
    ("cumesh", "cumesh"),
    ("nvdiffrast.torch", "nvdiffrast"),
    ("nvdiffrec_render", "nvdiffrec_render"),
    ("o_voxel", "o_voxel"),
    ("o_voxel.postprocess", "o_voxel.postprocess"),
    ("comfy_api.latest", "ComfyUI typed API"),
]

MODEL_WARNINGS = [
    "models/pixal3d/pipeline.json",
    "models/pixal3d/moge-2-vitl-normal.pt",
    "models/pixal3d/camenduru_dinov3-vitl16-pretrain-lvd1689m",
    "models/pixal3d/briaai_RMBG-2.0",
]


def log(message: str) -> None:
    print(f"[SimplePixal3D install] {message}", flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def pip_install(args: list[str], env: dict[str, str] | None = None) -> None:
    run([PYTHON, "-m", "pip", "install", *args], env=env)


def import_ok(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def import_error(module_name: str) -> str:
    try:
        importlib.import_module(module_name)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def torch_info() -> tuple[object | None, str, str, bool]:
    try:
        import torch
    except Exception:
        return None, "missing", "missing", False
    return torch, torch.__version__, str(torch.version.cuda), bool(torch.cuda.is_available())


def wheel_stack_supported() -> tuple[bool, str]:
    torch, torch_version, cuda_version, _cuda_ok = torch_info()
    if torch is None:
        return False, "torch is not installed"
    if not sys.platform.startswith("linux"):
        return False, f"unsupported platform {sys.platform}; bundled wheels are Linux only"
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False, f"unsupported CPU arch {platform.machine()}; bundled wheels are x86_64 only"
    if sys.version_info[:2] != (3, 12):
        return False, f"unsupported Python {sys.version_info.major}.{sys.version_info.minor}; bundled wheels are cp312"
    if not torch_version.startswith("2.8.0"):
        return False, f"unsupported torch {torch_version}; bundled wheels require torch 2.8.0"
    if not cuda_version.startswith("12.8"):
        return False, f"unsupported torch CUDA {cuda_version}; bundled wheels require cu128"
    return True, "Linux x86_64 + Python 3.12 + torch 2.8.0 + CUDA 12.8"


def ensure_torch_present() -> None:
    torch, torch_version, cuda_version, cuda_ok = torch_info()
    if torch is None:
        raise SystemExit("torch is not installed. Install the matching ComfyUI torch build first; this installer will not install torch.")
    log(f"Python: {sys.version.split()[0]} at {PYTHON}")
    log(f"Torch: {torch_version}, torch CUDA: {cuda_version}, cuda_available={cuda_ok}")
    if not cuda_ok:
        log("WARNING: torch.cuda.is_available() is False. Install may finish, but Pixal3D inference requires CUDA.")


def install_if_missing(module_name: str, pip_args: list[str], force: bool = False, env: dict[str, str] | None = None) -> None:
    if not force and import_ok(module_name):
        log(f"OK: {module_name}")
        return
    pip_install(pip_args, env=env)


def natten_has_lib() -> bool:
    try:
        import natten
        return bool(getattr(natten, "HAS_LIBNATTEN", False))
    except Exception:
        return False


def install(args: argparse.Namespace) -> None:
    ensure_torch_present()

    log("Installing pure Python dependencies; torch will not be installed or upgraded by this script.")
    if args.upgrade_build_tools:
        pip_install(["--upgrade", "pip", "setuptools", "wheel"])
    else:
        log("Skipping pip/setuptools/wheel upgrade. Use --upgrade-build-tools if your build tools are too old.")
    pip_install(["--upgrade-strategy", "only-if-needed", *PURE_REQUIREMENTS])

    install_if_missing("utils3d", [UTILS3D_WHEEL], force=args.force)
    install_if_missing("moge.model.v2", [MOGE_SPEC], force=args.force)

    supported, reason = wheel_stack_supported()
    if args.skip_cuda_wheels:
        log("Skipping CUDA extension wheels because --skip-cuda-wheels was set.")
    elif not supported:
        log(f"Skipping bundled CUDA wheels: {reason}")
    else:
        log(f"Installing CUDA extension wheels for {reason}")
        for module_name, wheel_url in CUDA_WHEELS:
            install_if_missing(module_name, [wheel_url], force=args.force)

        if args.force or not natten_has_lib():
            pip_install([NATTEN_SPEC, "-f", NATTEN_LINKS])
        else:
            log("OK: natten with libnatten")

        install_if_missing("flash_attn", [FLASH_ATTN_SPEC, "--no-build-isolation"], force=args.force)


def check_environment() -> bool:
    ok = True
    ensure_torch_present()
    for module_name, label in CHECK_IMPORTS:
        error = import_error(module_name)
        if error:
            ok = False
            log(f"FAIL: {label} ({module_name}) -> {error}")
        else:
            log(f"OK: {label}")

    if import_ok("natten"):
        if natten_has_lib():
            log("OK: natten.HAS_LIBNATTEN=True")
        else:
            ok = False
            log("FAIL: natten.HAS_LIBNATTEN=False; install the torch/CUDA matched NATTEN wheel.")

    for rel in MODEL_WARNINGS:
        path = COMFY_ROOT / rel
        if path.exists():
            log(f"OK model path: {rel}")
        else:
            log(f"WARN missing model path: {rel}")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Install dependencies for ComfyUI-SimplePixal3D.")
    parser.add_argument("--check", action="store_true", help="Only check imports and model paths; do not install anything.")
    parser.add_argument("--force", action="store_true", help="Reinstall special packages/wheels even if their imports already work.")
    parser.add_argument("--skip-cuda-wheels", action="store_true", help="Skip CUDA extension wheels, NATTEN, and flash-attn.")
    parser.add_argument("--upgrade-build-tools", action="store_true", help="Also upgrade pip, setuptools, and wheel before installing dependencies.")
    args = parser.parse_args()

    log(f"Node dir: {NODE_DIR}")
    log(f"ComfyUI root: {COMFY_ROOT}")

    if args.check:
        return 0 if check_environment() else 1

    install(args)
    log("Running final dependency check...")
    if not check_environment():
        log("Install finished, but one or more checks failed. Read the FAIL lines above.")
        return 1

    log("Done. Restart ComfyUI before using the node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
