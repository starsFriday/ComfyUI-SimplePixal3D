import gc
import json
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_COMFY_ROOT = os.path.abspath(os.path.join(_NODE_DIR, "..", ".."))
_VENDORED_DIR = _NODE_DIR
if _VENDORED_DIR not in sys.path:
    sys.path.insert(0, _VENDORED_DIR)

os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.path.join(_NODE_DIR, "autotune_cache.json"))
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

try:
    import comfy.utils as comfy_utils
except Exception:
    comfy_utils = None

try:
    from comfy_api.latest import Types as comfy_types
except Exception:
    comfy_types = None


_PIPELINE_CACHE: Dict[Tuple[str, bool, str], Any] = {}


class _Pixal3DProgress:
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.active_name = None
        self.active_start = None
        self.pbar = comfy_utils.ProgressBar(total) if comfy_utils is not None else None

    def step(self, name: str) -> None:
        self._finish_active()
        self.active_name = name
        self.active_start = time.perf_counter()
        logging.info("[SimplePixal3D] Step %s/%s: %s", self.current + 1, self.total, name)
        if self.pbar is not None:
            self.pbar.update_absolute(self.current, self.total)

    def done(self) -> None:
        self._finish_active()
        if self.pbar is not None:
            self.pbar.update_absolute(self.total, self.total)
        logging.info("[SimplePixal3D] Done.")

    def fail(self) -> None:
        if self.active_name is not None:
            elapsed = time.perf_counter() - self.active_start
            logging.exception("[SimplePixal3D] Failed during %s after %.1fs", self.active_name, elapsed)

    def _finish_active(self) -> None:
        if self.active_name is None:
            return
        elapsed = time.perf_counter() - self.active_start
        self.current = min(self.current + 1, self.total)
        logging.info("[SimplePixal3D] Finished %s in %.1fs", self.active_name, elapsed)
        if self.pbar is not None:
            self.pbar.update_absolute(self.current, self.total)
        self.active_name = None
        self.active_start = None


IMAGE_COND_CONFIGS = {
    "ss": {
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


def _resolve_path(path: str) -> str:
    path = os.path.expanduser(path.strip())
    if os.path.isabs(path):
        resolved = os.path.abspath(path)
    else:
        resolved = os.path.abspath(os.path.join(_COMFY_ROOT, path))
    if not os.path.exists(resolved):
        lowered = resolved.replace(os.sep + "Pixal3D", os.sep + "pixal3d")
        if lowered != resolved and os.path.exists(lowered):
            return lowered
        uppered = resolved.replace(os.sep + "pixal3d", os.sep + "Pixal3D")
        if uppered != resolved and os.path.exists(uppered):
            return uppered
    return resolved


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text[:80] if text else "pixal3d"


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4:
        raise ValueError(f"Expected ComfyUI IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")
    img = image[0].detach().cpu().float().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    if img.shape[-1] == 4:
        return Image.fromarray(img, mode="RGBA")
    return Image.fromarray(img[..., :3], mode="RGB")


def _pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _get_device() -> str:
    if model_management is not None:
        return str(model_management.get_torch_device())
    return "cuda" if torch.cuda.is_available() else "cpu"


def _soft_empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model_management is not None:
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass


def _load_pipeline_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(model_path, "pipeline.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Pixal3D pipeline config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["args"]


def _local_helper_path(model_path: str, dirname: str) -> Optional[str]:
    candidates = [
        os.path.join(model_path, dirname),
        os.path.join(model_path, "TencentARC_Pixal3D", dirname),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None

def _load_safetensors_state_dict(model_file: str, skip_unsupported_keys=None):
    from safetensors.torch import load_file

    try:
        return load_file(model_file)
    except Exception:
        pass

    import struct

    skip_unsupported_keys = set(skip_unsupported_keys or [])
    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }
    with open(model_file, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))

    tensor_items = {k: v for k, v in header.items() if isinstance(v, dict) and "dtype" in v}
    unsupported = {k for k, v in tensor_items.items() if v["dtype"] not in dtype_map}
    if unsupported - skip_unsupported_keys:
        raise RuntimeError(f"Unsupported safetensors dtype in {model_file}: {unsupported}")
    if unsupported:
        logging.warning("Skipping unsupported safetensors tensors from %s: %s", model_file, sorted(unsupported))

    data_start = 8 + header_len
    state_dict = {}
    for key, meta in tensor_items.items():
        dtype = meta["dtype"]
        if dtype not in dtype_map:
            continue
        start, _end = meta["data_offsets"]
        shape = tuple(meta["shape"])
        arr = np.memmap(model_file, mode="r", dtype=dtype_map[dtype], offset=data_start + start, shape=shape)
        state_dict[key] = torch.from_numpy(np.array(arr, copy=True))
    return state_dict


def _load_pixal_model(ckpt_base: str):
    from pixal3d import models as pixal_models

    config_file = f"{ckpt_base}.json"
    model_file = f"{ckpt_base}.safetensors"
    if not os.path.isfile(config_file) or not os.path.isfile(model_file):
        raise FileNotFoundError(f"Missing Pixal3D checkpoint files for {ckpt_base}")

    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_cls = getattr(pixal_models, config["name"])
    model = model_cls(**config["args"])
    state_dict = _load_safetensors_state_dict(model_file, skip_unsupported_keys={"rope_phases"})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    unexpected = set(unexpected)
    allowed_missing = {"rope_phases"}
    if missing - allowed_missing:
        logging.warning("Pixal3D checkpoint %s missing keys: %s", model_file, sorted(missing - allowed_missing))
    if unexpected:
        logging.debug("Pixal3D checkpoint %s unexpected keys: %s", model_file, sorted(unexpected))
    return model


def _build_pipeline_from_local(model_path: str, low_vram: bool):
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline, rembg, samplers

    args = _load_pipeline_config(model_path)
    loaded_models = {}
    for key, rel_path in args["models"].items():
        if key not in Pixal3DImageTo3DPipeline.model_names_to_load:
            continue
        ckpt_base = os.path.join(model_path, rel_path)
        load_start = time.perf_counter()
        logging.info("[SimplePixal3D] Loading core model %s from %s", key, ckpt_base)
        loaded_models[key] = _load_pixal_model(ckpt_base)
        logging.info("[SimplePixal3D] Loaded core model %s in %.1fs", key, time.perf_counter() - load_start)

    pipeline = Pixal3DImageTo3DPipeline(loaded_models)
    pipeline._pretrained_args = args
    pipeline.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(
        **args["sparse_structure_sampler"]["args"]
    )
    pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipeline.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(
        **args["shape_slat_sampler"]["args"]
    )
    pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]
    pipeline.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(
        **args["tex_slat_sampler"]["args"]
    )
    pipeline.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]
    pipeline.shape_slat_normalization = args["shape_slat_normalization"]
    pipeline.tex_slat_normalization = args["tex_slat_normalization"]

    rembg_name = args.get("rembg_model", {}).get("args", {}).get("model_name", "briaai/RMBG-2.0")
    local_rembg = _local_helper_path(model_path, "briaai_RMBG-2.0")
    if local_rembg is not None:
        rembg_name = local_rembg
    pipeline.rembg_model = getattr(rembg, args["rembg_model"]["name"])(model_name=rembg_name)

    pipeline.low_vram = bool(low_vram)
    pipeline.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")
    pipeline.pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipeline._device = "cpu"
    return pipeline


class _LocalDinoV3ProjFeatureExtractor(torch.nn.Module):
    def __init__(self, naf_mode: str = "duplicate_lr", **kwargs):
        super().__init__()
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
            ProjGrid,
        )

        self._inner = DinoV3ProjFeatureExtractor(**kwargs)
        self._proj_grid_cls = ProjGrid
        self.naf_mode = naf_mode
        self._naf_failed = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("_inner")
            return getattr(inner, name)

    def __setattr__(self, name: str, value):
        modules = self.__dict__.get("_modules")
        inner = modules.get("_inner") if modules is not None else None
        if inner is not None and name in {"grid_resolution", "proj_grid"}:
            setattr(inner, name, value)
            return
        super().__setattr__(name, value)

    def to(self, *args, **kwargs):
        self._inner.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self):
        self._inner.cuda()
        return super().cuda()

    def cpu(self):
        self._inner.cpu()
        return super().cpu()

    def _try_load_naf(self) -> bool:
        if self.naf_mode == "duplicate_lr":
            return False
        if self._inner.naf_model is not None:
            return True
        try:
            self._inner._load_naf()
            return self._inner.naf_model is not None
        except Exception:
            if self.naf_mode == "strict_naf":
                raise
            if not self._naf_failed:
                logging.warning("Pixal3D NAF upsampler is unavailable; using duplicated DINO features.")
                self._naf_failed = True
            return False

    def forward(
        self,
        image,
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ):
        inner = self._inner
        if isinstance(image, torch.Tensor):
            if image.ndim != 4:
                raise ValueError("Image tensor should be batched [B,C,H,W]")
        elif isinstance(image, list):
            if not all(isinstance(i, Image.Image) for i in image):
                raise ValueError("Image list should contain PIL images.")
            image = [i.resize((inner.image_size, inner.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255.0 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            device = next(inner.model.parameters()).device
            image = torch.stack(image).to(device)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if camera_angle_x is None or distance is None or mesh_scale is None:
            raise ValueError("camera_angle_x, distance, and mesh_scale are required.")

        image_for_naf = image.clone() if inner.use_naf_upsample else None
        image = inner.transform(image)

        with torch.no_grad():
            z = inner.extract_features(image)
            z_clstoken = z[:, 0:1]
            num_reg = getattr(inner.model.config, "num_register_tokens", 4)
            z_regtokens = z[:, 1 : 1 + num_reg]
            z_patchtokens = z[:, 1 + num_reg :]
            batch_size = image.shape[0]
            z_patchtokens_spatial = z_patchtokens.reshape(
                batch_size,
                inner.patch_number,
                inner.patch_number,
                -1,
            )

            z_proj_lr = inner.proj_grid(
                z_patchtokens_spatial,
                camera_angle_x,
                distance,
                mesh_scale,
                transform_matrix,
            )

            if inner.use_naf_upsample:
                z_proj_hr = None
                if self._try_load_naf():
                    lr_features = z_patchtokens_spatial.permute(0, 3, 1, 2)
                    hr_features = inner.naf_model(image_for_naf, lr_features, inner.naf_target_size)
                    z_proj_hr = inner.proj_grid(
                        hr_features,
                        camera_angle_x,
                        distance,
                        mesh_scale,
                        transform_matrix,
                        BHWC=False,
                    )
                if z_proj_hr is None:
                    z_proj_hr = z_proj_lr
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr

            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)
        return z_global, z_proj


def _build_image_cond_model(config: Dict[str, Any], model_path: str, naf_mode: str):
    dino_model = _local_helper_path(model_path, "camenduru_dinov3-vitl16-pretrain-lvd1689m")
    if dino_model is None:
        dino_model = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    full_config = {**config, "model_name": dino_model}
    model = _LocalDinoV3ProjFeatureExtractor(naf_mode=naf_mode, **full_config)
    model.eval()
    return model


def _get_pipeline(model_path: str, low_vram: bool, naf_mode: str):
    cache_key = (model_path, bool(low_vram), naf_mode)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logging.info("Loading Simple Pixal3D pipeline from %s", model_path)
    pipeline = _build_pipeline_from_local(model_path, low_vram=low_vram)
    for attr, config_key in (
        ("image_cond_model_ss", "ss"),
        ("image_cond_model_shape_512", "shape_512"),
        ("image_cond_model_shape_1024", "shape_1024"),
        ("image_cond_model_tex_1024", "tex_1024"),
    ):
        load_start = time.perf_counter()
        logging.info("[SimplePixal3D] Loading image condition model %s", config_key)
        setattr(pipeline, attr, _build_image_cond_model(IMAGE_COND_CONFIGS[config_key], model_path, naf_mode))
        logging.info("[SimplePixal3D] Loaded image condition model %s in %.1fs", config_key, time.perf_counter() - load_start)

    device = _get_device()
    if low_vram:
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
    else:
        pipeline.low_vram = False
        pipeline.to(torch.device(device))
        for attr in (
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ):
            getattr(pipeline, attr).to(torch.device(device))

    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def _clear_pipeline_cache() -> None:
    for pipeline in _PIPELINE_CACHE.values():
        try:
            pipeline.cpu()
        except Exception:
            pass
    _PIPELINE_CACHE.clear()
    _soft_empty_cache()


def _compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    return float((focal_length * resolution / 32.0).item())


def _distance_from_fov(
    camera_angle_x: float,
    grid_point: torch.Tensor,
    target_point: torch.Tensor,
    mesh_scale: float,
    image_resolution: int,
) -> float:
    rotation_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw = gp[0].item(), gp[1].item()
    xt = float(target_point[0].item())
    f_pixels = _compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    return float(f_pixels * xw / x_ndc - yw)


def _manual_camera_params(
    fov_degrees: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    camera_angle_x = math.radians(float(fov_degrees))
    distance = _distance_from_fov(
        camera_angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )
    return {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": float(mesh_scale),
    }


def _auto_moge_camera_params(
    image: Image.Image,
    moge_model: str,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    from moge.model.v2 import MoGeModel

    device = _get_device()
    model_name = moge_model.strip() or "models/pixal3d/moge-2-vitl-normal.pt"
    local_model_name = _resolve_path(model_name)
    if os.path.exists(local_model_name):
        model_name = local_model_name
    elif model_name.startswith(("~", "/", ".")):
        model_name = local_model_name
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    width, height = image.size
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx = float(intrinsics[0, 0]) * width
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    distance = _distance_from_fov(
        camera_angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )
    model.cpu()
    del model
    _soft_empty_cache()
    return {
        "camera_angle_x": float(camera_angle_x),
        "distance": float(distance),
        "mesh_scale": float(mesh_scale),
    }


def _export_glb(
    pipeline,
    mesh,
    resolution: int,
    output_path: str,
    decimation_target: int,
    texture_size: int,
    remesh: bool,
    use_webp: bool,
) -> None:
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(decimation_target),
        texture_size=int(texture_size),
        remesh=bool(remesh),
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    rot = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rot)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=bool(use_webp))

def _pixal_mesh_to_comfy_mesh(mesh):
    if comfy_types is None:
        raise RuntimeError('ComfyUI MESH type is unavailable; please update ComfyUI.')

    vertices = mesh.vertices.detach().float().cpu()
    faces = mesh.faces.detach().long().cpu()

    rotation = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=vertices.dtype,
    )
    vertices = vertices @ rotation.T

    vertex_colors = None
    if hasattr(mesh, 'query_vertex_attrs'):
        try:
            attrs = mesh.query_vertex_attrs()
            base_color = getattr(mesh, 'layout', {}).get('base_color', slice(0, 3))
            vertex_colors = attrs[:, base_color].detach().float().clamp(0.0, 1.0).cpu()
            if vertex_colors.shape[-1] not in (3, 4):
                logging.warning('[SimplePixal3D] Ignoring unsupported vertex color shape: %s', tuple(vertex_colors.shape))
                vertex_colors = None
        except Exception:
            logging.exception('[SimplePixal3D] Failed to query vertex colors; exporting geometry only.')

    return comfy_types.MESH(
        vertices=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        vertex_colors=vertex_colors.unsqueeze(0) if vertex_colors is not None else None,
        unlit=vertex_colors is not None,
    )


class SimplePixal3DImageToGLB:
    CATEGORY = "Pixal3D/Simple"
    RETURN_TYPES = ("STRING", "IMAGE", "MESH")
    RETURN_NAMES = ("glb_path", "preprocessed_image", "mesh")
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "models/pixal3d"}),
                "resolution": (["1024", "1536"], {"default": "1024"}),
                "low_vram": ("BOOLEAN", {"default": True}),
                "camera_mode": (["manual_fov", "auto_moge"], {"default": "manual_fov"}),
                "manual_fov_degrees": ("FLOAT", {"default": 49.13, "min": 5.0, "max": 120.0, "step": 0.01}),
                "sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "ss_guidance": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "shape_guidance": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "tex_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "max_num_tokens": ("INT", {"default": 49152, "min": 4096, "max": 196608, "step": 1024}),
                "texture_size": ([1024, 2048, 4096], {"default": 2048}),
                "decimation_target": ("INT", {"default": 500000, "min": 10000, "max": 2000000, "step": 10000}),
                "remesh": ("BOOLEAN", {"default": True}),
                "use_webp_texture": ("BOOLEAN", {"default": True}),
                "naf_mode": (["duplicate_lr", "auto_fallback", "strict_naf"], {"default": "duplicate_lr"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "pixal3d"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "moge_model": ("STRING", {"default": "models/pixal3d/moge-2-vitl-normal.pt"}),
            },
        }

    def generate(
        self,
        image,
        model_path: str,
        seed: int,
        resolution: str,
        low_vram: bool,
        camera_mode: str,
        manual_fov_degrees: float,
        sampling_steps: int,
        ss_guidance: float,
        shape_guidance: float,
        tex_guidance: float,
        max_num_tokens: int,
        texture_size: int,
        decimation_target: int,
        remesh: bool,
        use_webp_texture: bool,
        naf_mode: str,
        keep_model_loaded: bool,
        filename_prefix: str,
        moge_model: str = "models/pixal3d/moge-2-vitl-normal.pt",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Pixal3D requires CUDA.")

        progress = _Pixal3DProgress(7)
        progress.step("prepare inputs")

        model_path_abs = _resolve_path(model_path)
        if not os.path.isdir(model_path_abs):
            raise FileNotFoundError(f"Pixal3D model directory not found: {model_path_abs}")

        output_dir = folder_paths.get_output_directory()
        basename = f"{_safe_name(filename_prefix)}_{int(time.time() * 1000)}_s{int(seed)}.glb"
        output_path = os.path.join(output_dir, basename)

        pil_image = ImageOps.exif_transpose(_image_tensor_to_pil(image))

        progress.step("load Pixal3D pipeline")
        pipeline = _get_pipeline(model_path_abs, bool(low_vram), naf_mode)

        progress.step("preprocess image / remove background")
        logging.info("[SimplePixal3D] Preprocessing image")
        preprocessed = pipeline.preprocess_image(pil_image)

        progress.step(f"estimate camera ({camera_mode})")
        if camera_mode == "auto_moge":
            camera_params = _auto_moge_camera_params(
                preprocessed,
                moge_model,
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=512,
            )
        else:
            camera_params = _manual_camera_params(
                manual_fov_degrees,
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=512,
            )

        progress.step(f"run Pixal3D sampler/decode ({resolution}_cascade, {sampling_steps} steps)")
        torch.manual_seed(int(seed))
        ss_sampler = {
            "steps": int(sampling_steps),
            "guidance_strength": float(ss_guidance),
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        }
        shape_sampler = {
            "steps": int(sampling_steps),
            "guidance_strength": float(shape_guidance),
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        }
        tex_sampler = {
            "steps": int(sampling_steps),
            "guidance_strength": float(tex_guidance),
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        }

        pipeline_type = f"{int(resolution)}_cascade"
        logging.info("Pixal3D generating mesh with pipeline_type=%s", pipeline_type)
        mesh_list, (_shape_slat, _tex_slat, actual_resolution) = pipeline.run(
            preprocessed,
            camera_params=camera_params,
            seed=int(seed),
            sparse_structure_sampler_params=ss_sampler,
            shape_slat_sampler_params=shape_sampler,
            tex_slat_sampler_params=tex_sampler,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=int(max_num_tokens),
        )

        progress.step("export GLB / prepare mesh")
        logging.info("[SimplePixal3D] Exporting GLB to %s", output_path)
        _export_glb(
            pipeline,
            mesh_list[0],
            int(actual_resolution),
            output_path,
            decimation_target=int(decimation_target),
            texture_size=int(texture_size),
            remesh=bool(remesh),
            use_webp=bool(use_webp_texture),
        )
        logging.info("[SimplePixal3D] Converting Pixal3D output to ComfyUI MESH")
        comfy_mesh = _pixal_mesh_to_comfy_mesh(mesh_list[0])

        progress.step("cleanup")
        del mesh_list, _shape_slat, _tex_slat
        if not keep_model_loaded:
            _clear_pipeline_cache()
        else:
            _soft_empty_cache()

        progress.done()
        return {
            "ui": {
                "3d": [
                    {
                        "filename": basename,
                        "subfolder": "",
                        "type": "output",
                    }
                ]
            },
            "result": (output_path, _pil_to_image_tensor(preprocessed), comfy_mesh),
        }


class SimplePixal3DClearCache:
    CATEGORY = "Pixal3D/Simple"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def clear(self):
        _clear_pipeline_cache()
        return ("Simple Pixal3D cache cleared.",)


NODE_CLASS_MAPPINGS = {
    "SimplePixal3DImageToGLB": SimplePixal3DImageToGLB,
    "SimplePixal3DClearCache": SimplePixal3DClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplePixal3DImageToGLB": "Simple Pixal3D Image to GLB + Mesh",
    "SimplePixal3DClearCache": "Simple Pixal3D Clear Cache",
}
