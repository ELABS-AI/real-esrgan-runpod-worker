FROM ghcr.io/andrewembry312-hub/elabs-server/real-esrgan-runpod:latest
LABEL maintainer="E-Labs AI Studio" description="Real-ESRGAN upscale RunPod worker handler"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY handler.py /workspace/handler.py
COPY requirements-runpod.txt /workspace/requirements-runpod.txt

CMD ["python", "-u", "handler.py"]
