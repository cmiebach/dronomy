# Running RoMA on a RunPod CUDA GPU

RoMA is ~191 s/match emulated on Apple Silicon but **~4 s/match on a CUDA GPU**.
A RunPod GPU (RTX 4090 ≈ $0.34–0.69/h, billed per second) runs our whole RoMA /
auto-selection benchmark for **~1–2 €**. Two paths — pick one.

## Pick a GPU
RTX 4090 or RTX 3090 (24 GB) is ideal; even an RTX A4000 (16 GB) fits RoMA.
"Secure Cloud" is steadier; "Community Cloud" is cheaper.

---
## Path A — fastest start (PyTorch pod + setup script)  ← recommended for a one-off
1. Deploy a pod from the **"RunPod PyTorch 2.x (CUDA 12.1)"** template.
2. Open the web terminal and run:

```bash
cd /workspace
apt-get update && apt-get install -y git git-lfs wget unzip libgl1 libglib2.0-0
# MatchAnything fork (RoMA/eLoFTR weights + loader)
git clone --depth 1 https://huggingface.co/spaces/LittleFrog/MatchAnything ma
cd ma && pip install --upgrade pip wheel 'setuptools<81'
pip install -r requirements.txt spaces && pip install -e . --no-deps
cd imcui/third_party/MatchAnything && gdown 12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d \
  && unzip -q weights.zip && rm weights.zip && cd /workspace
# our code (private repo — paste a GitHub PAT for <TOKEN>)
git clone https://<TOKEN>@github.com/cmiebach/dronomy.git
cd dronomy && pip install -e . --no-deps && export PYTHONPATH=/workspace/dronomy/src
python -c "import torch; print('cuda', torch.cuda.is_available())"   # must print True
```

3. **Upload the data** (it is git-ignored, so not in the clone). From your Mac:
```bash
# small: reference tile + GPS track   (use runpodctl, printed in the pod's Connect tab)
runpodctl send data/reference data/gps_track.csv
# frames: either upload the MP4 to dronomy_video/ and extract on the pod,
# or pre-extract locally and upload data/frames/  (smaller).
```

4. **Run** the cascade on the GPU (LoFTR-located frames refined + missed recovered):
```bash
cd /workspace/dronomy
python scripts/bench_roma_cascade.py --spread 200 --max-recover 40 \
    --device cuda --loftr-csv data/outputs/val_loftr_12.csv \
    --out data/outputs/val_roma_cuda.csv
```
(With CUDA you can afford a big `--spread` and `--max-recover`; full-flight is feasible.)

5. **Download results** back to your Mac:
```bash
runpodctl receive <code-printed-by-the-pod>   # grabs data/outputs/val_roma_cuda.csv
```
Then locally: `python scripts/bench_combine.py` to fold RoMA into the table, and
re-run `scripts/08_vo_trajectory.py` with the RoMA-locked frames as extra anchors.

---
## Path B — robust, reproducible (build the CUDA image, push, deploy)
On your Mac (Docker Desktop):
```bash
cd <repo root>
docker build -f docker/Dockerfile.matchanything.cuda -t <dockerhubuser>/dronomy-ma-cuda .
docker push <dockerhubuser>/dronomy-ma-cuda            # ~11 GB, slow upload once
```
On RunPod: deploy a GPU pod **from this image** (`<dockerhubuser>/dronomy-ma-cuda`),
mount/upload `data/`, then run the same `bench_roma_cascade.py … --device cuda`.
The image already has torch+cu121, imcui, weights, and dronomy_loc baked in —
nothing to install on the pod.

---
## Cost
- Setup (Path A): ~10–20 min of GPU time.
- Bench: full-flight cascade ≈ 1–2 h on a 4090.
- **Total ≈ 1–2 € on a 4090; under ~5 € even generously.** Stop/delete the pod when done.
```
