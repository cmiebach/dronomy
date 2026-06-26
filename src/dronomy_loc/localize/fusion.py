"""Recursive fusion of intermittent absolute fixes into one smooth track.

The two estimators we have fail in *opposite* ways. Per-frame satellite locks
(``validate_frames``) carry no drift but are noisy and intermittent — many frames
do not lock at all, and the ones that do occasionally lock to the wrong place.
Visual odometry (``odometry``) is smooth and continuous but drifts without bound.
Fusing them gives the best of both: a constant-velocity motion model carries the
state across the gaps and *predicts* where the next fix should be, each accepted
fix pulls the state back onto the ground truth (killing drift), and a chi-square
**gate** rejects any fix that disagrees with the prediction — exactly the lock-to-
the-wrong-building outliers that wreck a raw per-frame track.

This is the standard loosely-coupled architecture for GPS-denied navigation, kept
deliberately offline and dependency-light (numpy only): a linear Kalman filter
over the state ``[east, north, v_east, v_north]`` in a local metre plane, plus a
Rauch–Tung–Striebel backward smoother. Because we score whole flights after the
fact, the *smoother* — which uses future fixes to correct past states — is the
estimate to report; the forward filter is what an online system would run live.

Telemetry-free contract is preserved: the only measurements are the system's own
visual fixes (and optionally VO-derived velocity). GPS never enters here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reference.geo import meters_per_degree_lat
from .trajectory import lonlat_to_local_m


@dataclass
class FusionConfig:
    """Tuning for the constant-velocity filter. Defaults suit a low-altitude
    multirotor survey (gentle accelerations, ~metre-scale visual fixes)."""
    accel_std: float = 1.0        # process noise as an expected accel (m/s^2)
    fix_std_m: float = 8.0        # default absolute-fix std when none supplied (m)
    vo_vel_std: float = 0.5       # std of a VO-derived velocity measurement (m/s)
    gate_chi2: float = 13.816     # 2-dof chi-square gate (~99.9%); <=0 disables
    init_pos_std: float = 50.0    # initial position uncertainty (m)
    init_vel_std: float = 10.0    # initial velocity uncertainty (m/s)


@dataclass
class FusionStep:
    """One time step fed to the filter, in the local east/north metre plane.

    ``pos``/``vel`` are ``None`` when that measurement is absent at this step
    (an unlocked frame has ``pos=None``). ``*_std`` override the config defaults
    per step — pass a fix's own uncertainty when the matcher provides one."""
    t_s: float
    pos: tuple[float, float] | None = None       # (east, north) metres
    pos_std: float | None = None
    vel: tuple[float, float] | None = None       # (v_east, v_north) m/s
    vel_std: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class FusionEstimate:
    """Smoothed (or filtered) state at one step, back in metres."""
    t_s: float
    east: float
    north: float
    v_east: float
    v_north: float
    pos_std_m: float                  # 1-sigma position uncertainty (trace-based)
    fix_used: bool                    # a position fix was accepted at this step
    fix_rejected: bool                # a position fix was present but gated out
    meta: dict = field(default_factory=dict)


def _F(dt: float) -> np.ndarray:
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], float)


def _Q(dt: float, accel_std: float) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance."""
    q = accel_std ** 2
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    return q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                         [0, dt4 / 4, 0, dt3 / 2],
                         [dt3 / 2, 0, dt2, 0],
                         [0, dt3 / 2, 0, dt2]], float)


_H_POS = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
_H_VEL = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], float)


def _update(x, P, z, H, R, gate_chi2):
    """One linear KF measurement update with optional chi-square gating.
    Returns (x, P, accepted)."""
    y = z - H @ x
    S = H @ P @ H.T + R
    if gate_chi2 and gate_chi2 > 0:
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > gate_chi2:
            return x, P, False
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P
    return x, P, True


def fuse_track(steps: list[FusionStep], cfg: FusionConfig | None = None,
               *, smooth: bool = True) -> list[FusionEstimate]:
    """Forward Kalman filter (+ optional RTS smoother) over ``steps``.

    The state is initialised at the first step carrying a position fix; steps
    before it are still emitted (predicted from that anchor). With ``smooth``
    the returned estimates use the whole flight (future fixes correct the past),
    which is the honest offline answer; ``smooth=False`` gives the causal filter
    an online system would produce."""
    cfg = cfg or FusionConfig()
    n = len(steps)
    if n == 0:
        return []

    first_pos = next((s.pos for s in steps if s.pos is not None), None)
    if first_pos is None:
        raise ValueError("fuse_track needs at least one step with a position fix")

    x = np.array([first_pos[0], first_pos[1], 0.0, 0.0], float)
    P = np.diag([cfg.init_pos_std ** 2, cfg.init_pos_std ** 2,
                 cfg.init_vel_std ** 2, cfg.init_vel_std ** 2]).astype(float)

    # Forward pass — store priors/posteriors for the RTS backward sweep.
    xs_prior, Ps_prior, xs_post, Ps_post, Fs = [], [], [], [], []
    used, rejected = [False] * n, [False] * n
    prev_t = steps[0].t_s
    for i, s in enumerate(steps):
        dt = max(s.t_s - prev_t, 0.0)
        prev_t = s.t_s
        F = _F(dt)
        if i == 0:                       # no propagation onto the initial anchor
            F = _F(0.0)
        x = F @ x
        P = F @ P @ F.T + _Q(max(dt, 1e-6), cfg.accel_std)
        xs_prior.append(x.copy()); Ps_prior.append(P.copy()); Fs.append(F)

        if s.pos is not None:
            R = np.eye(2) * (s.pos_std if s.pos_std is not None else cfg.fix_std_m) ** 2
            x, P, ok = _update(x, P, np.asarray(s.pos, float), _H_POS, R, cfg.gate_chi2)
            used[i], rejected[i] = ok, not ok
        if s.vel is not None:
            R = np.eye(2) * (s.vel_std if s.vel_std is not None else cfg.vo_vel_std) ** 2
            x, P, _ = _update(x, P, np.asarray(s.vel, float), _H_VEL, R, 0.0)
        xs_post.append(x.copy()); Ps_post.append(P.copy())

    xs, Ps = xs_post, Ps_post
    if smooth and n > 1:                  # Rauch–Tung–Striebel backward sweep
        xs = [a.copy() for a in xs_post]
        Ps = [a.copy() for a in Ps_post]
        for i in range(n - 2, -1, -1):
            F = Fs[i + 1]
            C = Ps_post[i] @ F.T @ np.linalg.inv(Ps_prior[i + 1])
            xs[i] = xs_post[i] + C @ (xs[i + 1] - xs_prior[i + 1])
            Ps[i] = Ps_post[i] + C @ (Ps[i + 1] - Ps_prior[i + 1]) @ C.T

    out = []
    for i, s in enumerate(steps):
        xi, Pi = xs[i], Ps[i]
        out.append(FusionEstimate(
            t_s=s.t_s, east=float(xi[0]), north=float(xi[1]),
            v_east=float(xi[2]), v_north=float(xi[3]),
            pos_std_m=float(np.sqrt(max(Pi[0, 0] + Pi[1, 1], 0.0))),
            fix_used=used[i], fix_rejected=rejected[i], meta=s.meta,
        ))
    return out


def _local_to_lonlat(east, north, ref_lat, ref_lon):
    m_lat, m_lon = meters_per_degree_lat(ref_lat)
    return ref_lat + north / m_lat, ref_lon + east / m_lon


@dataclass
class FusedFix:
    """A fused per-frame position in geographic coordinates."""
    frame: int
    t_s: float
    lat: float
    lon: float
    pos_std_m: float
    fix_used: bool
    fix_rejected: bool


def fuse_frame_scores(rows, cfg: FusionConfig | None = None, *,
                      smooth: bool = True,
                      ref: tuple[float, float] | None = None) -> list[FusedFix]:
    """Fuse a list of ``validate.FrameScore`` into one smooth, drift-free track.

    Only LOCKED frames contribute a position measurement; unlocked frames are
    bridged by the motion model, and locked-but-inconsistent frames are gated
    out (``fix_rejected``). Returns a fused position for EVERY input frame,
    ordered by frame index. ``ref`` fixes the local-plane origin (defaults to
    the mean of locked estimates) so results are comparable across runs."""
    rows = sorted(rows, key=lambda r: r.frame)
    if not rows:
        return []
    locked = [r for r in rows if r.locked and r.est_lat is not None]
    if not locked:
        raise ValueError("fuse_frame_scores needs at least one locked frame")
    ref_lat, ref_lon = ref or (float(np.mean([r.est_lat for r in locked])),
                               float(np.mean([r.est_lon for r in locked])))

    steps = []
    for r in rows:
        pos = None
        if r.locked and r.est_lat is not None:
            e, nth = lonlat_to_local_m([r.est_lat], [r.est_lon], ref_lat, ref_lon)[0]
            pos = (float(e), float(nth))
        steps.append(FusionStep(t_s=r.t_s, pos=pos, meta={"frame": r.frame}))

    est = fuse_track(steps, cfg, smooth=smooth)
    out = []
    for r, e in zip(rows, est):
        lat, lon = _local_to_lonlat(e.east, e.north, ref_lat, ref_lon)
        out.append(FusedFix(frame=r.frame, t_s=r.t_s, lat=lat, lon=lon,
                            pos_std_m=e.pos_std_m, fix_used=e.fix_used,
                            fix_rejected=e.fix_rejected))
    return out
