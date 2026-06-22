# Merge decision — diego vs caspar -> main

Two independent solutions to the same capstone were compared head-to-head
(2026-06-15) to decide what goes on `main`. The comparison was run as a blind,
adversarial review (fresh agents read both branches; three judge lenses —
architecture, results, pragmatic — then a synthesis). Decision rule (from the
team): **base = the branch with the better approach; port the better-metrics
pieces from the other.**

## Verdict

| | Approach | Metrics | Recommended base |
|---|---|---|---|
| **Result** | **diego** | **diego** | **diego** |

diego won **both**, so `main` is built on the `diego` base (`src/dronomy_loc/`)
and Caspar's genuinely-superior infrastructure is **ported onto it**.

### Why (evidence-cited)

- **Approach.** Real `Matcher`/`ReferenceProvider` ABCs + factories vs a
  504-line `if/elif` localizer monolith; a typed `GeoImage` geo-contract vs a
  bare bbox tuple split across modules (his `benchmark.py` even imports
  underscore-private helpers across module boundaries); sharded, sha1-verified,
  resumable ingestion + blur-aware frame selection vs plain uniform extraction.
  Decisively, diego's VO **anchors frame-to-frame chains on absolute
  satellite-locked frames** — the satellite+VO *fusion* Caspar's own
  `ACCURACY_LOG` lists as an unbuilt next step and elsewhere calls "not viable."
- **Metrics (Adrian's criterion: shape AND dimensions + coverage).** diego's
  `align_se2` is **rigid Umeyama with no scale**, so a wrong-size path cannot
  hide -> SE(2)-aligned ATE 27.6 m, path-length ratio 0.91, **100% coverage**
  (686/686 vs the ~6% per-frame ceiling), and the **only committed figure on
  either branch**. Caspar's scorer hard-codes `with_scale=True` (shape only,
  dimensions unverifiable), and his headline 71.5 m / 24.2 m numbers exist
  **only in a commit message** — not in his committed notebook.
- **Ties / honest caveats.** Absolute accuracy ties within noise (diego
  55.3 / 67.7 / 1.76 m vs caspar 56.8 / 70.0 / 1.73 m — diego's pipeline
  reproduces his bench). Both branches' VO CSVs are gitignored (not fully
  reproducible from the tree); diego at least commits the rendered figure.

## Ported from Caspar onto the diego base

| Item | Why |
|---|---|
| `docker/Dockerfile.matchanything` (real zju3dv weights) | diego's MatchAnything adapter referenced it but the file lived only on caspar — porting it makes the cross-modal path runnable (the lever for the 6% coverage ceiling) |
| MAGSAC++ homography (`cv2.USAC_MAGSAC` + RANSAC fallback) | more stable inlier counts near the low cross-modal floor; drop-in upgrade |
| `altitude.py` (telemetry-free AGL from footprint + FOV) | the brief's altitude bonus; diego had no altitude module |
| Dependency pins + dev extra; `STRUCTURE.md`, `CONTRIBUTING.md`, `.env.example` | packaging/process hygiene for a shared `main` |
| Caspar's VO data point + matchability narrative | second VO configuration + the quantified ~6% coverage story in the merged `ACCURACY_LOG` |

## Deferred (logged, not blocking the merge)

- Console-script entry points (our CLIs live in `scripts/`; needs a small CLI
  refactor into the package).
- `experiment.py` leaderboard (uses POSIX-only `fcntl`; needs a cross-platform
  lock before it runs on Windows).
- `benchmark.py` synthetic-GT harness (refactor off Caspar's private helpers
  onto `GeoImage`/`ReferenceProvider`).
- `match_confidence` geometric gate signals (harden the inlier-count lock gate).

These are real improvements; they carry integration risk and are scheduled as
follow-ups rather than rushed in ahead of the merge.
