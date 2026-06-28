# End-to-end run — real numbers (LoFTR on Apple-Silicon GPU/MPS)

## Per-frame localization (LoFTR, grid search)
- Frames scanned: **40**
- Locked (coverage): **6/40 = 15%**
- Accuracy on locked frames: median **2.6 m**, mean 8.7 m, best 1.8 m, worst 24.4 m

## Fusion filter on the real fix stream
- Fused positions for **all 40 frames** (gaps bridged)
- Fused error vs GT: median **60.5 m** (all frames)
- Outlier locks gated out: **0**

## Trajectory (visual odometry, LoFTR anchors)
- Frames in track: **686** (100% coverage)
- Raw ATE: 165.3 m
- **Shape-aligned RMSE (the precision metric): 137.2 m**
- Path-length ratio (1.0 = right size): 3.11
- Figure: `data/outputs/flightpath_real.svg`
