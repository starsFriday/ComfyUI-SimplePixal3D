# ComfyUI-SimplePixal3D

A simplified ComfyUI node for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D). It generates a 3D asset from a single image while keeping the workflow simple: one main node, local model loading, progress logging, GLB path output, and ComfyUI `MESH` output.

This node vendors the required Pixal3D Python package under the custom node directory and loads models from `models/pixal3d`. Runtime requirements are ComfyUI, the Python packages installed by `install.py`, CUDA extension packages, and the local model files.

## Preview

![ComfyUI-SimplePixal3D preview](https://github.com/user-attachments/assets/542f4496-1d60-436d-aed1-1f707c57058e)

## Node

Node name: `Simple Pixal3D Image to GLB + Mesh`

Category: `Pixal3D/Simple`

Outputs:

- `glb_path`: GLB file path exported by the Pixal3D/o_voxel exporter. By default it is saved directly under `output/`.
- `preprocessed_image`: Background-removed and cropped image used for inference.
- `mesh`: ComfyUI official `MESH` output. You can connect it to ComfyUI's official `Save 3D Model / SaveGLB` node.

The node also returns a ComfyUI 3D preview entry for the internally exported GLB, so the generated model can be previewed directly when your ComfyUI frontend has 3D preview support.

Note: `glb_path` uses Pixal3D's own PBR export path and usually preserves richer material output. The `mesh` output is intended for ComfyUI-native workflows; ComfyUI's official GLB saver stores geometry plus optional vertex colors/texture and is not identical to Pixal3D's full PBR texture baking.

## Install

Clone this repository into ComfyUI `custom_nodes`:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/starsFriday/ComfyUI-SimplePixal3D.git
```

Install dependencies with the Python interpreter used by your ComfyUI installation:

```bash
cd /path/to/ComfyUI
python custom_nodes/ComfyUI-SimplePixal3D/install.py
```

Check dependencies and model paths without installing anything:

```bash
python custom_nodes/ComfyUI-SimplePixal3D/install.py --check
```

Skip CUDA extension installation if you want to install them manually:

```bash
python custom_nodes/ComfyUI-SimplePixal3D/install.py --skip-cuda-wheels
```

The installer does not install or upgrade `torch`. Install a CUDA-enabled PyTorch build compatible with your ComfyUI first.

The bundled prebuilt CUDA wheels in `install.py` target:

- Linux x86_64
- Python 3.12
- torch 2.8.0
- CUDA 12.8

If your stack is different, the installer skips those fixed wheels and the final check will show which packages are missing. In that case, install matching builds manually.

By default the installer also avoids upgrading `pip`, `setuptools`, and `wheel`. If your build tools are too old, run:

```bash
python custom_nodes/ComfyUI-SimplePixal3D/install.py --upgrade-build-tools
```

## Model Directory

Default model root:

```text
models/pixal3d
```

The node also accepts old workflows that still point to `models/Pixal3D`; it will try to resolve that path to `models/pixal3d` if the lowercase directory exists.

Expected structure:

```text
models/pixal3d/
├── pipeline.json
├── ckpts/
│   ├── ss_flow_img_dit_1_3B_64_bf16.json
│   ├── ss_flow_img_dit_1_3B_64_bf16.safetensors
│   ├── ss_dec_conv3d_16l8_fp16.json
│   ├── ss_dec_conv3d_16l8_fp16.safetensors
│   ├── slat_flow_img2shape_dit_1_3B_512_bf16.json
│   ├── slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
│   ├── slat_flow_img2shape_dit_1_3B_1024_bf16.json
│   ├── slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors
│   ├── shape_dec_next_dc_f16c32_fp16.json
│   ├── shape_dec_next_dc_f16c32_fp16.safetensors
│   ├── slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json
│   ├── slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors
│   ├── tex_dec_next_dc_f16c32_fp16.json
│   └── tex_dec_next_dc_f16c32_fp16.safetensors
├── camenduru_dinov3-vitl16-pretrain-lvd1689m/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── briaai_RMBG-2.0/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── moge-2-vitl-normal.pt
```

## Download Models

Install Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
```

Download the Pixal3D model files:

```bash
cd /path/to/ComfyUI
mkdir -p models/pixal3d
huggingface-cli download TencentARC/Pixal3D --local-dir models/pixal3d
```

Download DINOv3 image feature model:

```bash
huggingface-cli download camenduru/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir models/pixal3d/camenduru_dinov3-vitl16-pretrain-lvd1689m
```

Download RMBG-2.0 background removal model:

```bash
huggingface-cli download briaai/RMBG-2.0 \
  --local-dir models/pixal3d/briaai_RMBG-2.0
```

Download MoGe-2 and rename it to the node's default local filename:

```bash
python - <<'PY'
from pathlib import Path
from shutil import copyfile
from huggingface_hub import hf_hub_download

root = Path('models/pixal3d')
root.mkdir(parents=True, exist_ok=True)
src = hf_hub_download('Ruicheng/moge-2-vitl-normal', filename='model.pt')
copyfile(src, root / 'moge-2-vitl-normal.pt')
print(root / 'moge-2-vitl-normal.pt')
PY
```

Optional `hfd` download example:

```bash
cd /path/to/ComfyUI
hfd TencentARC/Pixal3D --local-dir models/pixal3d --tool aria2c

hfd camenduru/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir models/pixal3d/camenduru_dinov3-vitl16-pretrain-lvd1689m \
  --tool aria2c

hfd briaai/RMBG-2.0 \
  --local-dir models/pixal3d/briaai_RMBG-2.0 \
  --tool aria2c
```

For MoGe, the Hugging Face file is named `model.pt`; place or rename it as:

```text
models/pixal3d/moge-2-vitl-normal.pt
```

## Recommended Defaults

Start with:

- `model_path`: `models/pixal3d`
- `resolution`: `1024`
- `low_vram`: `True`
- `camera_mode`: `manual_fov`
- `naf_mode`: `duplicate_lr`

To use MoGe camera estimation:

- Set `camera_mode` to `auto_moge`.
- Make sure `models/pixal3d/moge-2-vitl-normal.pt` exists.

If you have enough VRAM, try:

- `resolution`: `1536`
- higher `sampling_steps`
- higher `texture_size`

## Troubleshooting

Run the checker first:

```bash
cd /path/to/ComfyUI
python custom_nodes/ComfyUI-SimplePixal3D/install.py --check
```

Check that the node imports:

```bash
python -c "import sys; sys.path.insert(0, 'custom_nodes/ComfyUI-SimplePixal3D'); import nodes; print(nodes.NODE_DISPLAY_NAME_MAPPINGS)"
```

If CUDA extension packages are missing, rerun:

```bash
python custom_nodes/ComfyUI-SimplePixal3D/install.py
```

If the direct 3D preview is blank but the saved GLB opens correctly, try disabling `use_webp_texture`. Some browser-side viewers have incomplete support for GLB files that use WebP texture extensions.

## Notes

- CUDA is required. CPU inference is not supported.
- The Pixal3D model files are large, roughly 24 GB for the main repository before helper models.
- The first generation in a fresh ComfyUI session can be slow because the Pixal3D pipeline, helper models, and CUDA kernels need to load and initialize. With `keep_model_loaded=True`, second and later generations in the same session are much faster.
- Keep DINOv3, RMBG, and MoGe local if you want fully offline ComfyUI startup/inference.
- If your workflow only needs the official ComfyUI saver, connect the `mesh` output to `Save 3D Model`. If you want Pixal3D's own material export, use `glb_path`.

## Credits

- Pixal3D GitHub: https://github.com/TencentARC/Pixal3D
- Pixal3D Hugging Face: https://huggingface.co/TencentARC/Pixal3D
- DINOv3 mirror: https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- MoGe: https://github.com/microsoft/MoGe
- MoGe-2 ViT-L normal: https://huggingface.co/Ruicheng/moge-2-vitl-normal
