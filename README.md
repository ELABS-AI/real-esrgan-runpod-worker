# elabs / Real-ESRGAN Upscale

[![Run on RunPod](https://runpod.io/badge/runpod-hub)](https://runpod.io/console/hub)

**Real-ESRGAN** upscaling with multiple scale factors (2x–4x). Supports image and video frames, face enhancement (GFPGAN), and tile-based processing for large images. Runs on any GPU with ≥4GB VRAM.

## Highlights

- **2x–4x upscaling** — high-quality, real-world image super-resolution
- **Face enhancement** — integrated GFPGAN for realistic face restoration
- **Tile-based processing** — handles large images without OOM errors
- **Automatic OOM recovery** — falls back to smaller tile sizes on out-of-memory
- **GPU efficient** — fp16 inference, runs on RTX 4090, L40S, A5000+

## API

### Input

```json
{
  "input": {
    "image_base64": "<base64-encoded JPEG/PNG/WEBP bytes>",
    "scale": 4,
    "face_enhance": false,
    "tile_size": 512
  }
}
```

### Output

```json
{
  "image_b64": "<base64-encoded PNG>",
  "scale": 4,
  "original_size": [1920, 1080],
  "output_size": [7680, 4320],
  "wall_time_s": 2.1
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_base64` | string | **required** | Base64-encoded image bytes (JPEG, PNG, WEBP) |
| `scale` | int | `4` | Upscale factor (1, 2, 3, 4) |
| `face_enhance` | bool | `false` | Apply GFPGAN face enhancement |
| `tile_size` | int | `512` | Tile size for large image processing (0 = no tiling, max 2048) |

## GPU Requirements

- **Recommended**: RTX 4090 / RTX 6000 Ada / L40S / A5000+
- **Minimum**: Any GPU with ≥4GB VRAM
- **CUDA**: 12.0+

## Benchmark

| GPU | Scale | Image Size | Face Enhance | Wall Time |
|---|---|---|---|---|
| RTX 4090 | 4x | 512×512 | No | ~0.5s |
| RTX 4090 | 4x | 1920×1080 | No | ~2.1s |
| RTX 4090 | 4x | 512×512 | Yes | ~1.2s |
| L40S | 4x | 1920×1080 | No | ~1.8s |
| A5000 | 4x | 1920×1080 | No | ~4.5s |

## License

Apache-2.0 — Real-ESRGAN (BSD-3-Clause), GFPGAN (Apache-2.0).
