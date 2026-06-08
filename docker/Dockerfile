# Dronomy GPS-denied localization — deployable image.
#
# Build context is the repo ROOT (see docker/docker-compose.yml `context: ..`),
# so all COPY paths are relative to the project root, not docker/.
#
# The 3.5 GB drone video and all generated data are NEVER baked in — they are
# mounted as volumes at runtime (see docker-compose.yml). Only code + deps live
# in the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries:
#  - libgl1 / libglib2.0-0 : required at import time by opencv-python (libGL, glib)
#  - ffmpeg                 : robust MP4 decoding for frame extraction
#  - git                    : occasionally needed by pip for VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CPU-only torch/torchvision FIRST, from the CPU wheel index, so we don't
# pull the multi-GB CUDA build (no local GPU). kornia (LoFTR) then resolves
# against this CPU torch.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package and its deps. Copy only the install inputs first so this
# layer is cached unless dependencies actually change.
COPY pyproject.toml README.md /workspace/
COPY src/ /workspace/src/
RUN pip install -e ".[deep,gee]"

# Application code + default config. Mounted over at runtime by compose, so an
# edit / git pull applies without a rebuild; baked here too so the image runs
# standalone (e.g. on a plain `docker run`).
COPY scripts/ /workspace/scripts/
COPY config/ /workspace/config/

# Default command verifies the whole stack imports — a quick "is the image OK?".
# Override to run any pipeline step, e.g.:
#   docker compose run --rm dronomy python scripts/01_extract_frames.py --probe
#   docker compose run --rm dronomy python scripts/02_fetch_reference.py
CMD ["python", "-c", "import cv2, ee, kornia, dronomy_loc; print('dronomy-loc container OK')"]
