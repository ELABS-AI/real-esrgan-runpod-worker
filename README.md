# elabs / Real-ESRGAN Upscale

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy-orange?logo=runpod)](https://console.runpod.io/hub)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)

**AI-powered image upscaling** using Real-ESRGAN. Upscale images 2x-4x with optional face enhancement (GFPGAN). Tile-based processing handles images of any size.

![Real-ESRGAN Upscale](https://pub-796a08821c1c483aaf5e274e0d03e350.r2.dev/hub-icons/real-esrgan.svg)

## Highlights

- 2x-4x upscaling -- crisp, detail-preserving enhancement
- Face enhancement -- optional GFPGAN face restoration
- Tile processing -- handles large images (4K+) without OOM
- Multiple models -- General, Anime, Photo variants
- Fast -- ~1-3s per 512x512 tile on RTX 4090

## Quick Start

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_base64": "<base64 PNG>", "scale": 4, "face_enhance": false}}'
```

## API

### Input

```json
{
  "input": {
    "image_base64": "<base64 PNG or JPG>",
    "scale": 4,
    "face_enhance": false,
    "tile_size": 512,
    "model": "general"
  }
}
```

### Output

```json
{
  "image_base64": "<base64 upscaled PNG>",
  "scale": 4,
  "original_size": [640, 480],
  "output_size": [2560, 1920],
  "wall_time_s": 2.1
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_base64` | string | required | Base64 PNG/JPG to upscale |
| `scale` | int | `4` | Upscale factor: 2 or 4 |
| `face_enhance` | bool | `false` | Apply GFPGAN face enhancement |
| `tile_size` | int | `512` | Tile size for large images (0 = no tiling) |
| `model` | string | `"general"` | "general", "anime", "photo" |

## GPU Requirements

- Minimum: >=4GB VRAM
- Recommended: RTX 4090, L40S, A5000 (>=12GB)
- CUDA: 12.4+

## License

BSD-3-Clause. Based on [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
