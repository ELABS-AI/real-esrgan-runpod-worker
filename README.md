# elabs / Real-ESRGAN Upscaler

Real-ESRGAN image upscaling at 2x, 4x, or 8x. Works on photos, illustrations, and anime-style images.

[![Docker Build](https://github.com/ELABS-AI/real-esrgan-runpod-worker/actions/workflows/build.yml/badge.svg)](https://github.com/ELABS-AI/real-esrgan-runpod-worker/actions/workflows/build.yml)

---

## Quick Start

Deploy this worker on [RunPod Serverless](https://www.runpod.io/serverless) using the **Deploy on RunPod** button in the Hub, or manually with the Docker image:

```
ghcr.io/elabs-ai/real-esrgan-runpod-worker:latest
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_SCALE` | `4` | Default upscale factor (2, 4, or 8) |
| `HF_HOME` | `/runpod-volume/models/huggingface` | HuggingFace cache directory |
| `HUGGINGFACE_HUB_CACHE` | `/runpod-volume/models/huggingface/hub` | HuggingFace hub cache |

> **Note:** `HF_HOME` and `HUGGINGFACE_HUB_CACHE` should point to a RunPod Network Volume mount path for model caching between runs.

---

## API Reference

### Input

```json
{"input": {"image_b64": "<base64 PNG/JPG>", "scale": 4}}
```

### Output

```json
{"image_b64": "<base64 PNG>", "original_size": [512, 512], "output_size": [2048, 2048], "wall_time_s": 2.1}
```

---

## Usage Examples

### Python (runpod SDK)

```python
import runpod
import base64

client = runpod.AsyncioEndpointClient("real-esrgan-runpod-worker")
result = await client.run({"input": {"image_b64": "<base64 PNG/JPG>", "scale": 4}})
print(result)
```

### cURL

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_b64": "<base64 PNG/JPG>", "scale": 4}}'

```

---

## GPU Requirements

RTX 3090+ (24GB VRAM) | ~1-5s per image | BSD-3 license

---

## License

Apache 2.0 — See [LICENSE](LICENSE)

---

## Built by [E-Labs AI](https://www.elabsai.com)

Part of the E-Labs AI Studio serverless model fleet. Visit [elabsai.com](https://www.elabsai.com) to use these models in a hosted UI.
