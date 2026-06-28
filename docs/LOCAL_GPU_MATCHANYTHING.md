# Running MatchAnything (RoMA / eLoFTR) on a LOCAL CUDA GPU

`docker/RUNPOD.md` covers the cloud-GPU path. This is the **native, no-Docker**
recipe that runs the real MatchAnything backend directly on a local NVIDIA GPU —
verified end-to-end on Windows 11 + **RTX 3080 Ti (12 GB)**, no virtualization,
no Docker, no admin. RoMA peaks at ~4.6 GB VRAM here, so any ≥8 GB CUDA card fits.

The MatchAnything weights live only in zju3dv's `imcui` fork (a HuggingFace Space),
not on PyPI — so this installs that fork into an **isolated env** kept separate
from the main `.venv` (its pins, e.g. `numpy~=1.24`, would otherwise clash).

## Recipe

```bash
# 0) isolated env — Python 3.11 (the C++ deps pytlsd/pycolmap have cp311 wheels;
#    cp312 does NOT, so do not use 3.10/3.12 here)
py -3.11 -m venv .venv-ma
.venv-ma/Scripts/python -m pip install --upgrade pip

# 1) CUDA torch (matches the fork's torch==2.8.0 pin, but GPU build)
.venv-ma/Scripts/python -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
.venv-ma/Scripts/python -c "import torch; print('cuda', torch.cuda.is_available())"  # True

# 2) the MatchAnything fork (code only; weights come in step 4)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://huggingface.co/spaces/LittleFrog/MatchAnything ../ma
cd ../ma
../dronomy/.venv-ma/Scripts/python -m pip install -r requirements.txt spaces
../dronomy/.venv-ma/Scripts/python -m pip install -e . --no-deps

# 3) **THE FIX**: the fork leaves kornia unpinned -> latest (0.8.x) removed
#    kornia.utils.grid; MatchAnything's old code imports it. Pin 0.7.3, which
#    still has that module AND works with torch 2.8.
../dronomy/.venv-ma/Scripts/python -m pip install "kornia==0.7.3"

# 4) weights (445 MB RoMA + 62 MB eLoFTR) into the dir the loader expects
cd imcui/third_party/MatchAnything
../../../../dronomy/.venv-ma/Scripts/python -m gdown 12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d -O weights.zip
../../../../dronomy/.venv-ma/Scripts/python -c "import zipfile; zipfile.ZipFile('weights.zip').extractall('.')"
# -> weights/matchanything_roma.ckpt, weights/matchanything_eloftr.ckpt

# 5) make our package importable in this env
cd ../../../../dronomy
.venv-ma/Scripts/python -m pip install -e . --no-deps
```

## Run

```bash
export IMCUI_CONFIG="$(pwd)/../ma/config/config.yaml"          # the matcher zoo
export SSL_CERT_FILE="$(.venv-ma/Scripts/python -c 'import certifi;print(certifi.where())')"
#  ^ RoMA's DINOv2 + VGG backbones download via torch.hub on first use; on a fresh
#    Windows Python that TLS fetch fails with CERTIFICATE_VERIFY_FAILED without this.

# cascade: RoMA refines LoFTR's locked frames, then a small recovery grid where LoFTR missed
.venv-ma/Scripts/python scripts/bench_roma_cascade.py \
    --loftr-csv data/outputs/run_all/val_loftr.csv \
    --spread 20 --device cuda --max-recover 8 \
    --out data/outputs/val_roma_cuda.csv
```

## Result (real, this machine)

| | coverage | recall@5m | median err |
|---|---|---|---|
| RoMA **refine** (given LoFTR's prior) | 4/4 | **1.00** | **~2.0 m** (one frame 16.5 m → 1.3 m) |
| RoMA **cascade** (refine + blind recovery) | 60 % | **0.35** | 3.5 m (mean inflated by blind false-locks) |

Direct cross-frame match: **~3500 inliers**. Blind dense recovery still false-locks
on repetitive grass without the relative-margin gate (`search.lock_margin_ratio`);
RoMA's value is precision-given-a-prior plus recovering frames the sparse matchers miss.

## Gotchas
- **Python 3.11 only** for `.venv-ma` (pytlsd/pycolmap wheels stop at cp311).
- **`kornia==0.7.3`** — the unpinned latest breaks MatchAnything's `kornia.utils.grid` import.
- **`SSL_CERT_FILE`** must point at certifi or the backbone downloads fail on Windows.
- Keep this env **separate** from `.venv` — the fork pins `numpy~=1.24`, torch 2.8, etc.
