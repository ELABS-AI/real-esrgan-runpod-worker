"""
RunPod serverless handler for Real-ESRGAN — image upscaling.

Architecture:
  - Real-ESRGAN (Real-ESRGAN_x4plus / RealESRGAN_x2plus)
  - Multiple scale factors (2x, 3x, 4x)
  - Optional face enhancement (GFPGAN / RestoreFormer)
  - Tile-based processing for large images
  - Runs on any GPU with >=4GB VRAM

Environment (set by RunPod template):
  - RUNPOD_POD_ID       — auto
  - RUNPOD_AI_API_KEY   — auto
  - ESRGAN_SCALE        — default scale factor (default: 4)
  - ESRGAN_FACE_ENHANCE — default face enhancement (default: false)

Input schema (via RunPod serverless job):
  {
    "input": {
      "image_base64": "<base64-encoded image bytes>",  // REQUIRED — JPEG/PNG/WEBP
      "scale": 4,                                      // optional — upscale factor (2-4)
      "face_enhance": false,                           // optional — apply GFPGAN face enhancement
      "tile_size": 512                                 // optional — tile size for large images (0=no tiling)
    }
  }

Output:
  {
    "image_base64": "<base64-encoded PNG>",
    "scale": 4,
    "original_size": [1920, 1080],
    "wall_time_s": 2.1
  }
"""

import base64
import io
import os
import time
import traceback

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_SCALE = int(os.environ.get("ESRGAN_SCALE", "4"))
DEFAULT_FACE_ENHANCE = os.environ.get("ESRGAN_FACE_ENHANCE", "false").lower() == "true"

# ── Global models (loaded once, reused across jobs) ──────────────────────────
_upsampler = None
_face_enhancer = None


def load_models(scale: int, face_enhance: bool):
    """Load Real-ESRGAN upsampler (and optional face enhancer) once."""
    global _upsampler, _face_enhancer

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download import load_file_from_url
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer

    # Load upsampler
    if _upsampler is None:
        print(f"[Cold Start] Loading Real-ESRGAN upsampler (scale={scale})...", flush=True)
        t0 = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}", flush=True)

        # Determine model URL based on scale
        if scale <= 2:
            model_name = "RealESRGAN_x2plus"
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            netscale = 2
        else:
            model_name = "RealESRGAN_x4plus"
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x4plus.pth"
            netscale = 4

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)

        model_path = load_file_from_url(url=model_url, model_dir="/models/realesrgan", progress=True)

        _upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=512,  # Tile size for large images, can be overridden per job
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False,  # fp16 for speed
            device=device,
        )

        print(f"[Cold Start] Upsampler loaded in {time.time() - t0:.1f}s", flush=True)

    # Load face enhancer if requested
    if face_enhance and _face_enhancer is None:
        print("[Cold Start] Loading GFPGAN face enhancer...", flush=True)
        t0 = time.time()

        _face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            upscale=1,  # We already upscale first, face enhancer just refines
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=_upsampler,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        print(f"[Cold Start] Face enhancer loaded in {time.time() - t0:.1f}s", flush=True)

    return _upsampler, _face_enhancer


def decode_image(image_base64: str) -> Image.Image:
    """Decode base64 image bytes into a PIL Image."""
    image_bytes = base64.b64decode(image_base64)
    buf = io.BytesIO(image_bytes)
    return Image.open(buf).convert("RGB")


def image_to_b64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_inference(
    image_base64: str,
    scale: int = DEFAULT_SCALE,
    face_enhance: bool = DEFAULT_FACE_ENHANCE,
    tile_size: int = 512,
) -> dict:
    """
    Run Real-ESRGAN upscaling.
    Returns dict with image_b64, scale, original_size, wall_time_s.
    """
    t_start = time.time()

    # Decode input image
    print(f"[Inference] Decoding image ({len(image_base64)} base64 chars)...", flush=True)
    input_image = decode_image(image_base64)
    original_size = input_image.size  # (width, height)
    print(f"  Original size: {original_size[0]}x{original_size[1]}", flush=True)

    # Convert PIL to numpy (RGB)
    img_np = np.array(input_image)

    # Load models
    upsampler, face_enhancer = load_models(scale, face_enhance)

    # Apply upsampling
    print(f"[Inference] Upscaling {scale}x with tile_size={tile_size}...", flush=True)
    t_upscale = time.time()

    # Update tile size if provided
    if tile_size > 0:
        upsampler.tile = tile_size

    try:
        with torch.inference_mode():
            if face_enhance and face_enhancer is not None:
                # Face enhance path: GFPGAN handles both upscaling and face restoration
                _, _, output = face_enhancer.enhance(
                    img_np,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=0.5,
                )
            else:
                # Standard upscaling path
                output, _ = upsampler.enhance(img_np, outscale=scale)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear CUDA cache and try with smaller tiles
            torch.cuda.empty_cache()
            if tile_size > 256:
                print(f"[Retry] OOM detected, retrying with tile_size={tile_size // 2}", flush=True)
                upsampler.tile = tile_size // 2
                with torch.inference_mode():
                    output, _ = upsampler.enhance(img_np, outscale=scale)
            else:
                raise
        else:
            raise

    upscale_time = time.time() - t_upscale
    print(f"[Inference] Upscaling took {upscale_time:.1f}s", flush=True)

    # Convert numpy output back to PIL
    output_image = Image.fromarray(output)

    # Encode output
    output_b64 = image_to_b64(output_image, format="PNG")

    wall_time = time.time() - t_start
    output_size = output_image.size
    print(f"[Done] Output size: {output_size[0]}x{output_size[1]}, total time: {wall_time:.1f}s", flush=True)

    return {
        "image_b64": output_b64,
        "scale": scale,
        "original_size": list(original_size),
        "output_size": list(output_size),
        "wall_time_s": round(wall_time, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ═══════════════════════════════════════════════════════════════════════════════


def handler(job):
    """
    RunPod serverless handler: base64 image → upscaled base64 image.
    Called once per job. Models stay loaded across jobs (globals).
    """
    job_input = job.get("input", {})

    image_base64 = job_input.get("image_base64", "")
    if not image_base64:
        return {"error": "Missing required field: image_base64"}

    scale = int(job_input.get("scale", DEFAULT_SCALE))
    face_enhance = bool(job_input.get("face_enhance", DEFAULT_FACE_ENHANCE))
    tile_size = int(job_input.get("tile_size", 512))

    # Validate scale
    scale = max(1, min(4, scale))

    # Validate tile_size
    tile_size = max(0, min(2048, tile_size))

    try:
        result = run_inference(
            image_base64=image_base64,
            scale=scale,
            face_enhance=face_enhance,
            tile_size=tile_size,
        )
        return result

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": f"Real-ESRGAN upscaling failed: {str(exc)}",
            "traceback": traceback.format_exc(),
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
