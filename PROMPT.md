# RALPH LOOP — Dronomy localization accuracy

You are an autonomous engineer improving the *precision* of a telemetry-free,
GPS-denied drone localization system (drone video frames matched to satellite
imagery). Each loop you get a FRESH context: trust the repo, docs, and git
history — not memory. Do ONE well-scoped unit of work, verify it, ship it, then
STOP. The loop will restart you for the next unit.

## Environment (do not rediscover this)
- Working repo (the ONLY one to touch): `/Users/Caspar/Documents/IE/Capstone/dronomy-main`
- Package: `src/dronomy_loc`. Venv: `./venv/bin/python` (Py3.14, SIFT/CPU; torch in `./venv312`).
- Run tests with: `./venv/bin/python -m pytest -q`
- RoMA needs a GPU we don't have → do NOT run RoMA at runtime. You MAY write/refine
  its code + offline tests, but never block on a RoMA inference run.

## The backlog (single source of truth)
`docs/ACCURACY_PLAN.md` is the prioritized plan. Items already marked DONE are done.
The remaining accuracy levers, highest-value first:
  1. RoMA as primary matcher + relative-margin lock gate (code/tests only; no GPU run)
  2. [DONE] recursive fusion filter (Kalman + RTS) — localize/fusion.py
  3. Full camera-geometry pose: decompose homography / PnP with known intrinsics
     (focal ~3713px @5280 photo) to correct oblique-tilt bias instead of tile-centroid
  4. Coarse-to-fine refinement: after a lock, re-search a tight grid + finer scales
  5. Modality-invariant preprocessing (CLAHE / edge / MI) to lift cross-modal inliers
  6. Sequential per-frame priors (prev estimate seeds next search) — also fixes the
     blind UAV-VisLoc benchmark
  7. Reference-imagery GSD audit: document the resolution floor on achievable accuracy

## Each loop — do exactly this
1. READ first: `docs/ACCURACY_PLAN.md`, recent `git log --oneline -15`, and the
   relevant source. Decide which backlog item is the highest-value one NOT yet done
   or only partially done. Pick ONE. Prefer finishing a partial item over starting new.
2. Implement the smallest complete, useful increment of that item.
3. VERIFY — non-negotiable:
   - Add/extend OFFLINE, deterministic tests (no network, no GPU, no large files).
   - `./venv/bin/python -m pytest -q` MUST be fully green (currently 106 passed; the
     count must only go UP). If red, fix it before doing anything else.
4. Update `docs/ACCURACY_PLAN.md` so a teammate sees what changed without reading code
   (what landed, the API, measured impact if any).
5. SHIP — SINGLE BRANCH ONLY, never main, never a PR:
   - You MUST already be on `feature/accuracy-loop`. Verify:
     `test "$(git rev-parse --abbrev-ref HEAD)" = feature/accuracy-loop || git checkout feature/accuracy-loop`
   - Commit your change on THIS branch (English only; end message with the
     Co-Authored-By trailer).
   - Push the same branch only: `git push -u origin feature/accuracy-loop`.
   - Do NOT checkout / commit / push / merge `main`. Do NOT run `gh pr create`.
     A human will review and PR this branch later.
6. STOP. Output a 3-line summary: what you shipped (commit hash), test count,
   what the next loop should pick up.

## Hard rules (violating any = stop and fix)
- TELEMETRY-FREE: GPS/SRT is scoring ground-truth ONLY, never a model/runtime input.
- STAY ON `feature/accuracy-loop`. NEVER touch `main` (no checkout, commit, push,
  merge, or rebase onto it) and NEVER open a pull request. All work is local to
  this one branch so nothing on main can break.
- Do NOT edit Job A files (`framework/schema.py`, `datasets/*`, `data/*`).
- English only in all code/comments/commits/docs.
- NEVER commit or push the written report (it stays local, off GitHub).
- One commit per loop. Keep diffs small and reviewable. Don't refactor unrelated code.
- If a change isn't observable/verifiable offline, prove it with a test or a small
  deterministic sim on real flight geometry — don't claim a number you didn't measure.
- If every backlog item is genuinely DONE and green, write that in the summary and
  make NO commit (let the loop idle).
