"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into SunPy, never change a return value and
never touch global random state. Re-calling checkers work on copies and run with
checking suppressed (re-entrancy guard).

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale`` with
each factor justified at the call site.

All imports of sunpy itself are lazy (inside functions) so that this module sits
outside the package import layering.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import warnings

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_ENV = "SCIBENCH_TRIGGER_LOG"
_MAX_PIXELS = 4_000_000
_JD_LO, _JD_HI = 2415020.5, 2488069.5  # 1900-01-01 .. 2100-01-01 (built-in ephemeris span)


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


SWALLOWED = []
REACHED = {}


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into SunPy."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None
        finally:
            _ACTIVE = False

    return wrapper


def _clone(smap):
    """Shallow clone so property caches filled by a checker never reach the production map
    (a cached ``wcs`` would swallow the metadata warning the user's first access should emit)."""
    return copy.copy(smap)


def _is_now(time):
    """P/X: the string 'now' resolves to a different instant on every parse."""
    return isinstance(time, str) and time.strip().lower() == "now"


def _eps_of(*quantities):
    """Largest unit round-off among the floating dtypes of the inputs (P/T: float32 stays float32)."""
    eps = _EPS
    for q in quantities:
        dt = np.asarray(getattr(q, "value", q)).dtype
        if dt.kind == "f":
            eps = max(eps, float(np.finfo(dt).eps))
    return eps


def _wrap180(x_deg):
    return (np.asarray(x_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _in_window(time):
    """P: observation time inside the built-in ephemeris span."""
    from sunpy.time import parse_time

    jd = np.asarray(parse_time(time).jd, dtype=np.float64)
    return bool(np.all((jd >= _JD_LO) & (jd <= _JD_HI)))


# ================================================================ ephemeris
@_guarded
def check_angular_radius(radius, distance, theta):
    """SP-EPH-001: sin(theta) = x <= theta <= tan(theta) with x = R/D."""
    import astropy.units as u

    x = np.asarray((radius / distance).to_value(u.one), dtype=np.float64)
    th = np.asarray(theta.to_value(u.rad), dtype=np.float64)
    x, th = np.broadcast_arrays(x, th)
    ok = np.isfinite(x) & np.isfinite(th) & (x > 0) & (x < 1)
    if not ok.any():
        return
    x, th = x[ok], th[ok]
    # theta carries a few ulps (arcsin plus two unit conversions): C * eps * theta
    # (float32 radius/distance make theta itself a float32 result: slack scales with its unit round-off)
    eps = _eps_of(radius, distance, theta)
    lo = x * (1.0 - _C * eps)
    hi = x / np.sqrt(1.0 - x * x) * (1.0 + _C * eps)
    trigger_if(np.any((th < lo) | (th > hi)), "SP-EPH-001")


@_guarded
def check_earth_distance(obstime, dist):
    """SP-EPH-002 (orbital envelope) and SP-EPH-003 (agrees with get_earth radius)."""
    import astropy.units as u

    if not _in_window(obstime):
        return
    r_au = np.asarray(dist.to_value(u.AU), dtype=np.float64)
    trigger_if(np.any((r_au < 0.9825) | (r_au > 1.0175)), "SP-EPH-002")
    from sunpy.coordinates.ephemeris import get_earth

    r_hgs = np.asarray(get_earth(obstime).radius.to_value(u.m), dtype=np.float64)
    r_vec = np.asarray(dist.to_value(u.m), dtype=np.float64)
    trigger_if(np.any(np.abs(r_hgs - r_vec) > 1e-11 * np.abs(r_vec)), "SP-EPH-003")


@_guarded
def check_b0(time, b0):
    """SP-EPH-004: sin(B0) = pole . (Sun->Earth unit vector), both in ICRS."""
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric

    from sunpy.sun import constants
    from sunpy.time import parse_time

    if _is_now(time) or not _in_window(time):
        return
    t = parse_time(time)
    vec = (get_body_barycentric("earth", t) - get_body_barycentric("sun", t)).xyz.to_value(u.m)
    unit = vec / np.linalg.norm(vec, axis=0)
    a0 = float(constants.get("alpha_0").to_value(u.rad))
    d0 = float(constants.get("delta_0").to_value(u.rad))
    pole = np.array([np.cos(d0) * np.cos(a0), np.cos(d0) * np.sin(a0), np.sin(d0)])
    ref = np.arcsin(np.tensordot(pole, unit, axes=(0, 0)))
    diff = np.abs(np.asarray(b0.to_value(u.rad)) - ref)
    # |B0| < 0.13 rad so arcsin is well conditioned; ~30 float64 operations
    trigger_if(np.any(diff > 1e-11), "SP-EPH-004")


@_guarded
def check_obliquity(time, true_obliquity):
    """SP-EPH-005: |true - mean obliquity| (nutation in obliquity) <= 10.5 arcsec."""
    import astropy.units as u

    from sunpy.coordinates.sun import mean_obliquity_of_ecliptic

    mean = mean_obliquity_of_ecliptic(time)
    d = np.abs((true_obliquity - mean).to_value(u.arcsec))
    trigger_if(np.any(d > 10.5), "SP-EPH-005")


@_guarded
def check_apparent_lon(time, lon):
    """SP-EPH-006: apparent - true longitude <= aberration + nutation (42 arcsec)."""
    import astropy.units as u

    from sunpy.coordinates.sun import true_longitude

    if not _in_window(time):
        return
    d = _wrap180((lon - true_longitude(time)).to_value(u.deg)) * 3600.0
    trigger_if(np.any(np.abs(d) > 42.0), "SP-EPH-006")


@_guarded
def check_apparent_equatorial(time, result, which):
    """SP-EPH-007: apparent RA/Dec of date equals the astropy TETE coordinates."""
    import astropy.units as u
    from astropy.coordinates import TETE, SkyCoord

    from sunpy.time import parse_time

    if _is_now(time) or not _in_window(time):
        return
    t = parse_time(time)
    ref = SkyCoord(0 * u.deg, 0 * u.deg, 0 * u.AU, frame="hcrs", obstime=t).transform_to(TETE(obstime=t))
    val = np.asarray(result.to_value(u.deg), dtype=np.float64)
    if which == "ra":
        d = _wrap180(val - ref.ra.to_value(u.deg)) * np.cos(ref.dec.to_value(u.rad))
    else:
        d = val - ref.dec.to_value(u.deg)
    # both chains use IAU 2006/2000A; rounding is <1e-9 arcsec, slack covers model-level differences
    trigger_if(np.any(np.abs(d) * 3600.0 > 1e-3), "SP-EPH-007")


@_guarded
def check_body_hgs(body, observer, obstime, body_hgs):
    """SP-EPH-008: Earth lies on the HGS prime meridian (no light-time shift)."""
    import astropy.units as u

    if observer is not None or not isinstance(body, str) or body.lower() != "earth":
        return
    if not _in_window(obstime):
        return
    lon = _wrap180(body_hgs.lon.to_value(u.deg))
    trigger_if(np.any(np.abs(lon) > 1e-9), "SP-EPH-008")


@_guarded
def check_light_time(body, observer_icrs, light_travel_time, emitted_time, obstime):
    """SP-EPH-009: the light-time iteration reached its fixed point."""
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric
    from astropy.constants import c as speed_of_light

    if not _in_window(obstime):
        return
    dist = (get_body_barycentric(body, emitted_time) - observer_icrs).norm()
    resid = np.abs((light_travel_time - dist / speed_of_light).to_value(u.s))
    # loop stops at 1e-8 s; the map is a contraction with rate v/c ~ 1e-4
    trigger_if(np.any(resid > 1e-7), "SP-EPH-009")


@_guarded
def check_eclipse(s, m, d, fraction):
    """SP-EPH-010: spherical-cap overlap fraction equals the planar two-disc overlap."""
    s, m, d, frac = (np.asarray(a, dtype=np.float64) for a in np.broadcast_arrays(s, m, d, fraction))
    ok = np.isfinite(s) & np.isfinite(m) & np.isfinite(d) & np.isfinite(frac)
    ok &= (s < 0.02) & (m < 0.02) & (s > 0) & (m > 0)
    if not ok.any():
        return
    s, m, d, frac = s[ok], m[ok], d[ok], frac[ok]
    sep = (d > 0)
    dd = np.where(sep, d, 1.0)
    a1 = np.clip((dd**2 + s**2 - m**2) / (2 * dd * s), -1, 1)
    a2 = np.clip((dd**2 + m**2 - s**2) / (2 * dd * m), -1, 1)
    root = np.clip((-dd + s + m) * (dd + s - m) * (dd - s + m) * (dd + s + m), 0, None)
    lens = s**2 * np.arccos(a1) + m**2 * np.arccos(a2) - 0.5 * np.sqrt(root)
    planar = np.where(d >= s + m, 0.0, np.where(d <= np.abs(s - m), np.minimum(m, s) ** 2 * np.pi, lens))
    planar = planar / (np.pi * s**2)
    # curvature correction is O(theta^2) ~ 1e-4; margin 20x
    trigger_if(np.any(np.abs(planar - frac) > 2e-3), "SP-EPH-010")


# ================================================================= Carrington
@_guarded
def check_carrington_number(time, number):
    """SP-CAR-001 (synodic rate) and SP-CAR-003 (constants consistent)."""
    import astropy.units as u

    from sunpy.coordinates.sun import carrington_rotation_number
    from sunpy.sun import constants

    p_syn = constants.mean_synodic_period.to_value(u.day)
    omega = constants.sidereal_rotation_rate.to_value(u.deg / u.day)
    ref = 1.0 / (omega / 360.0 - 1.0 / 365.256363004)
    trigger_if(abs(p_syn - ref) > 1e-4 * ref, "SP-CAR-003")
    if not _in_window(time):
        return
    later = carrington_rotation_number(time + 1 * u.day)
    rate = (np.asarray(later) - np.asarray(number)) * 360.0
    trigger_if(np.any((rate < 13.10) | (rate > 13.30)), "SP-CAR-001")


@_guarded
def check_carrington_roundtrip(crot, t):
    """SP-CAR-002: number(time(c)) = c to the documented 0.11 s."""
    import astropy.units as u

    from sunpy.coordinates.sun import carrington_rotation_number

    if not _in_window(t):
        return
    back = carrington_rotation_number(t)
    err_s = np.abs(np.asarray(back) - np.asarray(crot.to_value(u.one))) * 27.2753 * 86400.0
    trigger_if(np.any(err_s > 0.2), "SP-CAR-002")


@_guarded
def check_disk_center(obstime, l0, ltt, nearest, aberration):
    """SP-CAR-004: the Sun-disk centre seen from Earth has Carrington longitude L0."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from sunpy.coordinates.frames import HeliographicCarrington, Helioprojective

    if not (ltt and nearest and not aberration) or not obstime.isscalar or not _in_window(obstime):
        return
    centre = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective, observer="earth", obstime=obstime)
    hgc = centre.make_3d().transform_to(HeliographicCarrington(observer="earth", obstime=obstime))
    d = _wrap180(hgc.lon.to_value(u.deg) - l0.to_value(u.deg))
    trigger_if(abs(d) > 1e-9, "SP-CAR-004")


# ============================================================ differential rot
def _rates_deg_per_s(model):
    import astropy.units as u

    table = {"howard": [2.894, -0.428, -0.370] * u.urad / u.second,
             "snodgrass": [2.851, -0.343, -0.474] * u.urad / u.second,
             "allen": [14.44, -3.0, 0.0] * u.deg / u.day,
             "rigid": None}
    return table[model]


@_guarded
def check_diffrot(duration, latitude, model, frame_time, result):
    """SP-DRM-001..006 (re-calls differential_rotation with transformed inputs)."""
    import astropy.units as u

    from sunpy.sun.models import differential_rotation as dr
    from sunpy.sun import constants

    lat = latitude.to(u.deg)
    if not np.all(np.isfinite(lat.value)) or not np.all(np.isfinite(duration.to_value(u.s))):
        return
    dt_s = np.abs(duration.to_value(u.s))
    rot = result.to_value(u.deg)
    one_day = 1 * u.day
    # scale of the unwrapped rotation: (|A|+|B|+|C|) |dt| (+ synodic term), in degrees
    if model == "rigid":
        rate = constants.sidereal_rotation_rate.to_value(u.deg / u.s)
        scale = rate * dt_s
    else:
        abc = np.abs(_rates_deg_per_s_vals(model))
        scale = abc.sum() * dt_s
    scale = scale + 0.9856 / 86400.0 * dt_s
    eps = _eps_of(latitude, duration)
    # Longitude output is wrapped into [0, 360): absolute round-off up to eps*360 deg per value
    tol = _C * eps * np.maximum(scale, 360.0)

    # 001: hemisphere symmetry
    flip = dr(duration, -lat, model=model, frame_time=frame_time).to_value(u.deg)
    trigger_if(np.any(np.abs(_wrap180(flip - rot)) > tol), "SP-DRM-001")

    # 002: rate non-increasing toward the pole
    lat2 = np.sign(lat.value) * np.minimum(np.abs(lat.value) + 5.0, 90.0) * u.deg
    r1 = dr(one_day, lat, model=model, frame_time=frame_time).to_value(u.deg)
    r2 = dr(one_day, lat2, model=model, frame_time=frame_time).to_value(u.deg)
    trigger_if(np.any(r2 > r1 + 1e3 * eps * 15.0), "SP-DRM-002")

    # 003: linear in elapsed time (modulo 360)
    half = dr(duration / 2, lat, model=model, frame_time=frame_time).to_value(u.deg)
    trigger_if(np.any(np.abs(_wrap180(2 * half - rot)) > tol), "SP-DRM-003")

    # 004: equatorial rate close to the rigid sidereal rate
    eq = dr(one_day, 0 * u.deg, model=model, frame_time="sidereal").to_value(u.deg)
    ref = constants.sidereal_rotation_rate.to_value(u.deg / u.day)
    trigger_if(abs(eq / ref - 1.0) > 0.05, "SP-DRM-004")

    # 005: synodic correction equals the Earth's mean motion
    if np.all(dt_s < 1e5 * 86400.0):
        sid = dr(duration, lat, model=model, frame_time="sidereal").to_value(u.deg)
        syn = dr(duration, lat, model=model, frame_time="synodic").to_value(u.deg)
        expected = 360.0 / 365.256363004 / 86400.0 * duration.to_value(u.s)
        err = np.abs(_wrap180((sid - syn) - expected))
        trigger_if(np.any(err > 1e-4 * 0.9856 / 86400.0 * dt_s + tol), "SP-DRM-005")

    # 006: differential-rotation models agree (|lat| <= 60 deg)
    if model in ("howard", "snodgrass", "allen") and np.all(np.abs(lat.value) <= 60.0):
        rates = {m: dr(one_day, lat, model=m, frame_time="sidereal").to_value(u.deg)
                 for m in ("howard", "snodgrass", "allen")}
        others = np.mean([rates[m] for m in rates if m != model], axis=0)
        trigger_if(np.any(np.abs(rates[model] - others) > 0.10 * others), "SP-DRM-006")


def _rates_deg_per_s_vals(model):
    import astropy.units as u

    q = _rates_deg_per_s(model)
    return np.asarray(q.to_value(u.deg / u.s))


# ================================================================== frames
@_guarded
def check_observer_invariance(from_coo, to_frame, out):
    """SP-FRM-003: changing the observer relabels the point; it is the same in HGS."""
    import astropy.units as u

    from sunpy.coordinates.frames import HeliographicStonyhurst

    if not np.all(from_coo.obstime == to_frame.obstime):
        return
    hgs_frame = HeliographicStonyhurst(obstime=from_coo.obstime)
    a = from_coo.transform_to(hgs_frame).cartesian.xyz.to_value(u.m)
    b = out.transform_to(hgs_frame).cartesian.xyz.to_value(u.m)
    ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    if not np.any(ok):
        return
    a, b = a[:, ok], b[:, ok]
    # float64 error is relative to the largest distance entering the conversion: the point (a screen can sit
    # at 1e25 m) and both observers (a point near the Sun seen from 30 AU cancels terms of 3e12 m);
    # C eps ~ 1.4e-14 -> 1e-12
    scale = np.maximum(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    for obs in (from_coo.observer, to_frame.observer):
        scale = np.maximum(scale, np.max(obs.radius.to_value(u.m)))
    # float32 observer lon/lat build float32 HCC<->HGS rotations, applied to the Sun-centred point: C eps |p|;
    # float32 radii or coordinate components round the O(L) translation: C eps L
    obs = (from_coo.observer, to_frame.observer)
    eps_ang = _eps_of(*[getattr(o, c, None) for o in obs for c in ("lon", "lat")])
    eps_len = _eps_of(*[getattr(o, "radius", None) for o in obs], *[getattr(from_coo.data, c) for c in from_coo.data.components])
    pt = np.maximum(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    tol = 1e-3 + max(1e-12, _C * eps_len) * scale + (_C * eps_ang * pt if eps_ang > _EPS else 0.0)
    trigger_if(np.any(np.linalg.norm(a - b, axis=0) > tol), "SP-FRM-003")


@_guarded
def check_helio_angle(coord, angle):
    """SP-FRM-004: law of sines in the triangle observer - Sun centre - point."""
    import astropy.units as u

    hcc = coord.heliocentric
    # convert in float64: a unit multiplication in the input's float32 would round the position
    xyz = hcc.cartesian.xyz
    p = np.asarray(xyz.value, dtype=np.float64) * xyz.unit.to(u.m)
    rad = hcc.observer.radius
    big_d = np.asarray(rad.value, dtype=np.float64) * rad.unit.to(u.m)
    o = np.zeros_like(p)
    o[2] = big_d
    to_sun = -o
    to_pt = p - o
    cross = np.cross(to_sun, to_pt, axis=0)
    alpha = np.arctan2(np.linalg.norm(cross, axis=0), np.sum(to_sun * to_pt, axis=0))
    r = np.linalg.norm(p, axis=0)
    rhs = big_d / r * np.sin(alpha)
    lhs = np.sin(angle.to_value(u.rad))
    ok = np.isfinite(rhs) & np.isfinite(lhs) & (r > 0)
    # theta is an angle between O(max(D, r)) vectors that differ by |p - o|: the inputs carry eps*max(D, r)
    # absolute rounding, so theta (and alpha) are only defined to C eps max(D, r) / |p - o| radians
    eps_in = _eps_of(xyz, rad)
    cond = np.maximum(big_d, r) / np.maximum(np.linalg.norm(to_pt, axis=0), np.finfo(np.float64).tiny)
    trigger_if(np.any(ok & (np.abs(lhs - rhs) > 1e-8 * np.abs(rhs) + 1e-12 + _C * eps_in * cond)), "SP-FRM-004")


@_guarded
def check_limb(limb, observer, rsun):
    """SP-FRM-005: limb points sit at angular distance arcsin(rsun/D) from disk centre."""
    import astropy.units as u

    from sunpy.coordinates.frames import Helioprojective

    hpc = limb.transform_to(Helioprojective(observer=observer, obstime=observer.obstime))
    lon = hpc.Tx.to_value(u.rad)
    lat = hpc.Ty.to_value(u.rad)
    x = np.cos(lat) * np.cos(lon)
    yz = np.hypot(np.cos(lat) * np.sin(lon), np.sin(lat))
    alpha = np.arctan2(yz, x)
    # reference in float64 from the exact (possibly float32) inputs; arctan2 form is accurate at grazing (D -> R)
    r64 = float(np.float64(rsun.to_value(u.m)))
    d64 = float(np.float64(observer.radius.to_value(u.m)))
    if not (d64 > r64 > 0):
        return
    x = r64 / d64
    theta_l = float(np.arctan2(r64, np.sqrt((d64 - r64) * (d64 + r64))))
    # the limb is built from observer.radius and rsun: unit round-off eps_in in R/D moves theta_l by
    # eps_in * x / sqrt(1 - x^2) (conditioning of arcsin; diverges as the observer approaches the photosphere)
    eps = _eps_of(observer.radius, rsun)
    cond = x / np.sqrt((1.0 - x) * (1.0 + x))
    tol = max(1e-9, _C * eps) * theta_l + 4.0 * eps * cond + _C * _EPS
    trigger_if(np.any(np.abs(alpha - theta_l) > tol), "SP-FRM-005")


@_guarded
def check_great_arc(arc, points, pts_cartesian):
    """SP-FRM-006: arc points advance at uniform angle along the geodesic."""
    p = np.asarray(points, dtype=np.float64).reshape(-1)
    theta = float(arc.inner_angle.value)
    sin_t = abs(np.sin(theta))
    if not (sin_t >= 1e-6) or np.any((p < 0) | (p > 1)):
        return
    w = np.asarray(pts_cartesian, dtype=np.float64) - arc.center_cartesian[np.newaxis, :]
    v1, v2 = arc.v1, arc.v2

    def angle(a, b):
        return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))

    a1 = angle(np.broadcast_to(v1, w.shape), w)
    a2 = angle(w, np.broadcast_to(v2, w.shape))
    # the points are O(|center|) floats: absolute rounding C eps |center|, as an angle about the center
    # (float32 start/end/centre: their own unit round-off)
    eps_in = _eps_of(v1, v2, arc.center_cartesian)
    tol = _C * eps_in * (1.0 + np.linalg.norm(arc.center_cartesian) / max(np.linalg.norm(v1), np.finfo(np.float64).tiny)) / sin_t
    trigger_if(np.any(np.abs(a1 - p * theta) > tol) or np.any(np.abs(a2 - (1 - p) * theta) > tol), "SP-FRM-006")


# ================================================================== screens
@_guarded
def check_spherical_screen(centre, unit_rep, distance, radius):
    """SP-SCR-001: promoted point is on the screen sphere (far root)."""
    import astropy.units as u

    uvec = np.asarray(unit_rep.to_cartesian().xyz.to_value(u.one), dtype=np.float64)
    c = np.asarray(centre.xyz.to_value(u.m), dtype=np.float64)
    c = np.broadcast_to(c.reshape((3,) + (1,) * (uvec.ndim - 1)) if c.ndim == 1 else c, uvec.shape)
    d = np.asarray(distance.to_value(u.m), dtype=np.float64)
    r = float(radius.to_value(u.m))
    pt = uvec * d
    dist_to_centre = np.linalg.norm(pt - c, axis=0)
    cnorm = np.linalg.norm(c, axis=0)
    ok = np.isfinite(dist_to_centre) & np.isfinite(d)
    if not np.any(ok):
        return
    # d solves a quadratic with eps |c|^2 of cancellation; that moves the point by eps |c|^2 / r along the ray
    # (second order in the tangent direction), and the near/far roots merge within sqrt(C eps) |c| of tangency
    # float32 radius/distance/centre: their unit round-off enters the sphere radius relatively
    eps_in = _eps_of(radius, distance, centre.xyz)
    tol = 1e-9 * r + _C * _EPS * (cnorm + cnorm**2 / r) + _C * eps_in * (r + cnorm)
    proj = np.sum(uvec * c, axis=0)
    bad = (np.abs(dist_to_centre - r) > tol) | (d < proj - np.sqrt(_C * _EPS) * cnorm - tol)
    trigger_if(np.any(ok & bad), "SP-SCR-001")


@_guarded
def check_screen_iteration(screen, coord, distance, screen_frame):
    """SP-SCR-002: the returned distance is a fixed point of the 3D->2D->3D map."""
    from astropy.coordinates import UnitSphericalRepresentation
    import astropy.units as u

    other_3d = coord.realize_frame(coord.represent_as(UnitSphericalRepresentation) * distance)
    native_3d = other_3d.transform_to(screen_frame)
    native_2d = native_3d.realize_frame(native_3d.represent_as(UnitSphericalRepresentation))
    delta = screen.calculate_distance(native_2d) - native_3d.spherical.distance
    rel = np.abs((delta / distance).to_value(u.one))
    bad = np.isfinite(rel) & (rel > 1e-9)
    if not np.any(bad):
        return

    # A residual alone is not a defect: where the ray never meets the rotated screen the fixed-point equation has no
    # root and the library's warned best guess is legitimate. Count an element only if the residual delta(d) (d the
    # distance along the ray) changes sign continuously (a root, not a pole) somewhere on a wide log grid around the
    # returned distance, so a root exists that the solver failed to reach.
    def delta_of(c, dist):
        o3 = c.realize_frame(c.represent_as(UnitSphericalRepresentation) * dist)
        n3 = o3.transform_to(screen_frame)
        n2 = n3.realize_frame(n3.represent_as(UnitSphericalRepresentation))
        return np.atleast_1d((screen.calculate_distance(n2) - n3.spherical.distance).to_value(u.m))

    coords = np.atleast_1d(coord)
    dists = distance.reshape(-1) if distance.ndim else distance.reshape(1)
    # The library refines the static-screen distance locally (secant, seeded near it). Count an element only if
    # its distance is positive (a point in front of the observer) and the fixed point is locally well posed:
    # exactly one genuine root (sign change that bisects to |delta|/d < 1e-6, not a pole) in 0.8-1.25 x the
    # returned distance, on 0.05 % steps. Several roots there (multi-turn sheared rotation) or only a far root
    # (another branch) are not solver failures.
    fac = np.exp(np.linspace(np.log(0.8), np.log(1.25), 893))
    bad_flat = np.atleast_1d(bad).reshape(-1)
    for i in np.nonzero(bad_flat)[0]:
        ci, di = coords.reshape(-1)[i], dists[i]
        if not float(di.to_value(u.m)) > 0:
            continue
        vals = delta_of(ci, di * fac)
        ok = np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] * vals[1:] < 0)
        nroots = 0
        for j in np.nonzero(ok)[0]:
            lo, hi, flo = float(fac[j]), float(fac[j + 1]), vals[j]
            fm, mid = flo, lo
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fm = delta_of(ci, di * mid)[0]
                if not np.isfinite(fm):
                    break
                if fm * flo > 0:
                    lo, flo = mid, fm
                else:
                    hi = mid
            if np.isfinite(fm) and abs(fm) / float((di * mid).to_value(u.m)) < 1e-6:
                nroots += 1
        if nroots == 1:
            trigger_if(True, "SP-SCR-002")
            return


# ======================================================== rotated coordinates
@_guarded
def check_solar_rotate(coordinate, observer_arg, new_observer, diff_rot_kwargs, result):
    """SP-DRC-001 (reversible) and SP-DRC-002 (prograde in HCI)."""
    import astropy.units as u

    from sunpy.coordinates import frames
    from sunpy.physics.differential_rotation import solar_rotate_coordinate
    from sunpy.sun.models import differential_rotation

    frame = getattr(coordinate, "frame", coordinate)
    if not isinstance(frame, frames.Helioprojective) or frame.obstime is None:
        return
    old_obs = frame.observer
    if not isinstance(old_obs, frames.BaseCoordinateFrame) and not hasattr(old_obs, "frame"):
        return
    if getattr(old_obs, "obstime", None) is None:
        return

    # 002: prograde about the Sun's rotation axis in an inertial (ICRS-oriented) frame. The reference is
    # the longitude of the HCRS Cartesian vector in the de-tilted frame (Z = solar north); the frame
    # HCI is not used because its own orientation matrix carries ~4e-7 rad of rounding noise
    # (ecliptic pole from a 1 m vector shifted by ~1e9 m), far above the 1e-8 deg tolerance.
    from astropy.coordinates import HCRS, SphericalRepresentation

    from sunpy.coordinates._transformations import _SUN_DETILT_MATRIX

    def inertial_lon(coord, obstime):
        cart = coord.transform_to(HCRS(obstime=obstime)).cartesian
        return cart.transform(_SUN_DETILT_MATRIX).represent_as(SphericalRepresentation).lon

    hgs_old = coordinate.transform_to(frames.HeliographicStonyhurst(obstime=coordinate.obstime))
    interval = (new_observer.obstime - coordinate.obstime).to(u.s)
    drot = differential_rotation(interval, hgs_old.lat.to(u.deg), **diff_rot_kwargs)
    d = _wrap180((inertial_lon(result, new_observer.obstime) - inertial_lon(coordinate, coordinate.obstime) - drot).to_value(u.deg))
    fin = np.isfinite(d)
    # the longitude is conditioned as 1/cos(latitude): the rounding of the O(1 AU) transform is C eps rad there
    cos_lat = np.maximum(np.cos(hgs_old.lat.to_value(u.rad)), 1e-12)
    # float32 coordinate components or observer angles/radii: the frame rotations run at their unit round-off
    eps_ang = _eps_of(coordinate.Tx, coordinate.Ty, old_obs.lon, old_obs.lat, new_observer.lon, new_observer.lat)
    tol = 1e-8 + np.rad2deg(max(1e3 * _EPS, _C * eps_ang)) / cos_lat
    trigger_if(np.any(fin & (np.abs(d) > tol)), "SP-DRC-002")

    # 001: reversible (only when an explicit observer was supplied)
    if observer_arg is None or old_obs.obstime != coordinate.obstime:
        return
    kwargs = {k: v for k, v in diff_rot_kwargs.items() if k != "frame_time"}
    back = solar_rotate_coordinate(result, observer=old_obs, **kwargs)
    ref = coordinate.make_3d()
    dtx = (back.Tx - ref.Tx).to_value(u.arcsec)
    dty = (back.Ty - ref.Ty).to_value(u.arcsec)
    dd = np.abs((back.distance - ref.distance).to_value(u.m))
    ok = np.isfinite(dtx) & np.isfinite(dty) & np.isfinite(dd)
    # positions are O(L) floats, L the largest observer distance involved: rounding C eps L, seen from distance d
    dm = np.abs(ref.distance.to_value(u.m))
    big_l = float(np.nanmax([np.nanmax(old_obs.radius.to_value(u.m)), np.nanmax(new_observer.radius.to_value(u.m)), np.nanmax(dm)]))
    # float32 coordinate components or observer radius enter the round trip at their own unit round-off,
    # relative to the angular offset itself (Tx, Ty in arcsec) and to the distances
    eps_in = _eps_of(coordinate.Tx, coordinate.Ty, ref.distance, old_obs.radius, new_observer.radius)
    # float32 observer lon/lat: float32 Sun-centred rotations applied to the point (|p| = its heliocentric radius)
    eps_o = _eps_of(old_obs.lon, old_obs.lat, new_observer.lon, new_observer.lat)
    pr = float(np.nanmax(np.abs(hgs_old.radius.to_value(u.m)))) if eps_o > _EPS else 0.0
    off = float(np.nanmax(np.maximum(np.abs(ref.Tx.to_value(u.arcsec)), np.abs(ref.Ty.to_value(u.arcsec)))))
    ang = 1e-6 + np.rad2deg(_C * eps_in * big_l / dm) * 3600.0 + _C * eps_in * off + np.rad2deg(_C * eps_o * pr / dm) * 3600.0
    bad = (np.abs(dtx) > ang) | (np.abs(dty) > ang) | (dd > 1e-12 * dm + _C * eps_in * big_l + _C * eps_o * pr)
    trigger_if(np.any(ok & bad), "SP-DRC-001")


# ======================================================================= maps
def _scale_deg(smap):
    import astropy.units as u

    return float(max(abs(smap.scale.axis1.to_value(u.deg / u.pix)), abs(smap.scale.axis2.to_value(u.deg / u.pix))))


def _corner_pix(nx, ny):
    return (np.array([-0.5, nx - 0.5, -0.5, nx - 0.5]), np.array([-0.5, -0.5, ny - 0.5, ny - 0.5]))


def _world(smap, x, y):
    lon, lat = smap.wcs.wcs_pix2world(x, y, 0)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _max_shift_pixels(smap_old, lon_o, lat_o, lon_n, lat_n):
    """Largest difference of two sets of world coordinates, in old pixel scales."""
    dlon = _wrap180(lon_n - lon_o)
    dlat = lat_n - lat_o
    return float(np.max(np.hypot(dlon, dlat))) / _scale_deg(smap_old)


@_guarded
def check_resample_footprint(old, new):
    """SP-MAP-001: resample keeps the outer corners of the pixel grid on the sky."""
    old, new = _clone(old), _clone(new)
    ny, nx = old.data.shape
    my, mx = new.data.shape
    lo, la = _world(old, *_corner_pix(nx, ny))
    ln, lb = _world(new, *_corner_pix(mx, my))
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(ln))):
        return
    trigger_if(_max_shift_pixels(old, lo, la, ln, lb) > 1e-6, "SP-MAP-001")


@_guarded
def check_superpixel_footprint(old, new, dimensions, offset):
    """SP-MAP-002: superpixel grid corners coincide with old pixel-edge corners."""
    old, new = _clone(old), _clone(new)
    my, mx = new.data.shape
    ex, ey = _corner_pix(mx, my)
    ox = offset[0] - 0.5 + dimensions[0] * (ex + 0.5)
    oy = offset[1] - 0.5 + dimensions[1] * (ey + 0.5)
    lo, la = _world(old, ox, oy)
    ln, lb = _world(new, ex, ey)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(ln))):
        return
    trigger_if(_max_shift_pixels(old, lo, la, ln, lb) > 1e-6, "SP-MAP-002")


@_guarded
def check_resample_extrema(orig, dimensions, method, offset, m1, data):
    """SP-MAP-003: nearest/linear interpolation does not create new extrema."""
    if method not in ("nearest", "linear"):
        return
    shape = np.asarray(orig.shape, dtype=np.float64)
    scale = (shape - m1) / (dimensions - m1)
    lo_c = offset * scale
    hi_c = (dimensions - 1 + offset) * scale
    if np.any(lo_c < offset - 1e-9) or np.any(hi_c > shape - 1 + offset + 1e-9):
        return
    src = np.asarray(orig, dtype=np.float64)
    fin = np.isfinite(src)
    if not fin.any():
        return
    lo, hi = float(src[fin].min()), float(src[fin].max())
    out = np.asarray(data, dtype=np.float64)
    out = out[np.isfinite(out)]
    if out.size == 0:
        return
    eps = max(_EPS, float(np.finfo(np.asarray(orig).dtype).eps) if np.asarray(orig).dtype.kind == "f" else _EPS)
    tol = _C * eps * max(abs(lo), abs(hi))
    trigger_if(out.min() < lo - tol or out.max() > hi + tol, "SP-MAP-003")


@_guarded
def check_rotate_center(old, new, recenter, rmatrix=None):
    """SP-MAP-004: rotation about the array centre keeps the centre coordinate."""
    if recenter:
        return
    old, new = _clone(old), _clone(new)
    ny, nx = old.data.shape
    my, mx = new.data.shape
    lo, la = _world(old, np.array([(nx - 1) / 2.0]), np.array([(ny - 1) / 2.0]))
    ln, lb = _world(new, np.array([(mx - 1) / 2.0]), np.array([(my - 1) / 2.0]))
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(ln))):
        return
    # the centre is held fixed through the rotation matrix, applied to the offset of the reference pixel
    # from the array centre: its unit round-off (float32 angle) scales with that offset
    eps_in = _eps_of(rmatrix) if rmatrix is not None else _EPS
    d_ref = 0.0
    for m in (old, new):
        my_, mx_ = m.data.shape
        rx, ry = (float(v) for v in (m.reference_pixel.x.value, m.reference_pixel.y.value))
        d_ref = max(d_ref, float(np.hypot(rx - (mx_ - 1) / 2.0, ry - (my_ - 1) / 2.0)))
    trigger_if(_max_shift_pixels(old, lo, la, ln, lb) > 1e-6 + _C * eps_in * d_ref, "SP-MAP-004")


@_guarded
def check_submap(parent, sub):
    """SP-MAP-005: a submap keeps the pixel grid, the scale and the data."""
    import astropy.units as u

    parent, sub = _clone(parent), _clone(sub)
    ny, nx = sub.data.shape
    if ny == 0 or nx == 0:
        return
    wp, ws = parent.wcs.wcs, sub.wcs.wcs
    same_lin = (np.array_equal(wp.crval, ws.crval) and np.array_equal(wp.cdelt, ws.cdelt)
                and np.array_equal(wp.get_pc(), ws.get_pc()) and list(wp.ctype) == list(ws.ctype))
    if same_lin:
        # same linear WCS: sub pixel (0,0) is parent pixel crpix_parent - crpix_sub, with no world round trip
        dxy = np.asarray(wp.crpix, dtype=np.float64) - np.asarray(ws.crpix, dtype=np.float64)
        x0, y0 = dxy[:1], dxy[1:2]
        gtol = 1e-9 * (1.0 + float(np.max(np.abs(wp.crpix))))
    else:
        lon, lat = _world(sub, np.array([0.0]), np.array([0.0]))
        x0, y0 = parent.wcs.wcs_world2pix(lon, lat, 0)
        # world<->pixel round trip loses ~eps*360 deg absolute (longitude wrap) on each axis:
        # eps*360/(finest absolute axis scale in deg) pixels
        fine = min(min(abs(m.scale.axis1.to_value(u.deg / u.pix)), abs(m.scale.axis2.to_value(u.deg / u.pix)))
                   for m in (parent, sub))
        gtol = 1e-6 + _C * _EPS * 360.0 / fine
    if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(y0))):
        return
    ix, iy = int(np.rint(x0[0])), int(np.rint(y0[0]))
    # a round trip that cannot resolve a quarter pixel carries no information about the grid
    resolvable = gtol <= 0.25
    grid = resolvable and (abs(x0[0] - ix) > gtol or abs(y0[0] - iy) > gtol)
    ps = np.array([parent.scale.axis1.to_value(u.arcsec / u.pix), parent.scale.axis2.to_value(u.arcsec / u.pix)])
    ss = np.array([sub.scale.axis1.to_value(u.arcsec / u.pix), sub.scale.axis2.to_value(u.arcsec / u.pix)])
    scale_bad = np.any(np.abs(ps - ss) > 1e-12 * np.abs(ps))
    data_bad = False
    py, px = parent.data.shape
    if resolvable and 0 <= ix and 0 <= iy and ix + nx <= px and iy + ny <= py:
        data_bad = not np.array_equal(sub.data, parent.data[iy:iy + ny, ix:ix + nx], equal_nan=sub.data.dtype.kind in "fc")
    trigger_if(grid or scale_bad or data_bad, "SP-MAP-005")


def _disk_pixels_ok(smap, min_radius_pixels=0.0):
    """P: small map (<= 262144 px), field of view <= 5 deg, disk >= min_radius_pixels."""
    import astropy.units as u

    from sunpy.map.maputils import solar_angular_radius

    ny, nx = smap.data.shape
    if nx * ny > 262144:
        return False
    sc = max(smap.scale.axis1.to_value(u.deg / u.pix), smap.scale.axis2.to_value(u.deg / u.pix))
    if max(nx, ny) * sc > 5.0:
        return False
    if min_radius_pixels:
        # pixel step: the longest lattice vector of the full linear map (PC scaled by CDELT), not CDELT alone
        pc = np.asarray(smap.rotation_matrix, dtype=np.float64)
        cd = np.array([smap.scale.axis1.to_value(u.deg / u.pix), smap.scale.axis2.to_value(u.deg / u.pix)])
        lat = np.stack([pc[0, :] * cd[0], pc[1, :] * cd[1]])  # columns = lattice vectors a_j
        lens = np.hypot(lat[0], lat[1])
        step = float(np.max(lens))
        rad = solar_angular_radius(smap.coordinate_frame).to_value(u.deg)
        if rad < min_radius_pixels * max(sc, step):
            return False
        # skewed lattice: between two edge samples |a_j| apart the limb bulges by |a_j|^2 / (8 R); it must stay
        # within half the perpendicular row spacing h_j = |det| / |a_j| or an interior row can see a sliver the
        # edge samples straddle (unskewed square pixels: R >= |a|/4, already implied by the 2-step rule)
        det = abs(float(np.linalg.det(lat)))
        if det == 0.0 or np.any(rad < lens ** 3 / (4.0 * det)):
            return False
    return True


@_guarded
def check_disk_partition(smap, contains_limb):
    """SP-MAP-006: exactly one of all-on-disk / all-off-disk / contains-limb holds."""
    from sunpy.map.maputils import is_all_off_disk, is_all_on_disk

    smap = _clone(smap)
    flags = [bool(is_all_on_disk(smap)), bool(is_all_off_disk(smap)), bool(contains_limb)]
    trigger_if(sum(flags) != 1, "SP-MAP-006")


@_guarded
def check_all_on_disk(smap, result):
    """SP-MAP-007: edge test 'all on disk' implies every pixel centre is on disk."""
    from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

    if not result:
        return
    smap = _clone(smap)
    if not _disk_pixels_ok(smap):
        return
    trigger_if(not np.all(coordinate_is_on_solar_disk(all_coordinates_from_map(smap))), "SP-MAP-007")


@_guarded
def check_all_off_disk(smap, result):
    """SP-MAP-008: edge test 'all off disk' implies no pixel centre is on disk."""
    from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

    if not result:
        return
    smap = _clone(smap)
    if not _disk_pixels_ok(smap, 2.0):
        return
    trigger_if(np.any(coordinate_is_on_solar_disk(all_coordinates_from_map(smap))), "SP-MAP-008")


# ======================================================== kernels and headers
@_guarded
def check_resample_ramp(shape, dimensions, method, offset, m1):
    """SP-IMG-001: linear resampling reproduces an affine ramp at the sample positions."""
    from sunpy.image.resample import _resample_nearest_linear

    if method != "linear" or len(shape) != 2 or min(shape) < 2:
        return
    if int(np.prod(shape)) > _MAX_PIXELS or int(np.prod(dimensions)) > _MAX_PIXELS:
        return
    a, b, c = 1.5, -0.75, 3.25
    i = np.arange(shape[0], dtype=np.float64)[:, None]
    j = np.arange(shape[1], dtype=np.float64)[None, :]
    ramp = a * i + b * j + c
    out = _resample_nearest_linear(ramp, dimensions, "linear", offset, m1)
    scale = (np.asarray(shape) - m1) / (dimensions - m1)
    xi = (np.arange(dimensions[0], dtype=np.float64) + offset) * scale[0] - offset
    yj = (np.arange(dimensions[1], dtype=np.float64) + offset) * scale[1] - offset
    expected = a * xi[:, None] + b * yj[None, :] + c
    # affine interpolation: a few ulps of the ramp magnitude; factor 10 for the extrapolated cell
    tol = 10 * _C * _EPS * (abs(a) * shape[0] + abs(b) * shape[1] + abs(c))
    trigger_if(np.nanmax(np.abs(out - expected)) > tol, "SP-IMG-001")


@_guarded
def check_frame_wcs(frame, wcs):
    """SP-WCS-001: frame -> WCS -> frame is lossless for observer, obstime and rsun."""
    import astropy.units as u

    from sunpy.coordinates import frames
    from sunpy.coordinates.wcs_utils import solar_wcs_frame_mapping

    back = solar_wcs_frame_mapping(wcs)
    bad = type(back) is not type(frame)
    if not bad and frame.obstime is not None:
        # DATE-OBS is written with the frame time's `precision` digits: the string resolves 10**-precision s
        res = max(1e-3, 10.0 ** -int(getattr(frame.obstime, "precision", 3))) + 1e-6
        bad = bool(abs((back.obstime - frame.obstime).to_value(u.s)) > res)
    if not bad and hasattr(frame, "rsun") and frame.rsun is not None:
        bad = bool(abs(back.rsun.to_value(u.m) - frame.rsun.to_value(u.m)) > 1e-12 * frame.rsun.to_value(u.m))
    obs = getattr(frame, "observer", None)
    if not bad and isinstance(frame, frames.Helioprojective | frames.HelioprojectiveRadial | frames.Heliocentric):
        o1 = obs.frame if hasattr(obs, "frame") else obs
        if isinstance(o1, frames.BaseCoordinateFrame):
            o2 = back.observer
            o2 = o2.frame if hasattr(o2, "frame") else o2
            bad = (type(o1) is not type(o2)
                   or abs(_wrap180(o1.lon.to_value(u.deg) - o2.lon.to_value(u.deg))) > 1e-12 * 360
                   or abs(o1.lat.to_value(u.deg) - o2.lat.to_value(u.deg)) > 1e-12 * 90
                   or abs(o1.radius.to_value(u.m) - o2.radius.to_value(u.m)) > 1e-12 * o1.radius.to_value(u.m))
    trigger_if(bad, "SP-WCS-001")


@_guarded
def check_fits_header(coordinate, reference_pixel, scale, rotation_matrix, shape, meta_wcs):
    """SP-WCS-002: the header places the reference coordinate at the reference pixel."""
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS

    if rotation_matrix is not None:
        return
    if reference_pixel is None:
        reference_pixel = u.Quantity([(shape[1] - 1) / 2.0 * u.pixel, (shape[0] - 1) / 2.0 * u.pixel])
    hdr = fits.Header()
    for k, v in dict(meta_wcs).items():
        if isinstance(v, str | int | float | bool | np.floating | np.integer):
            hdr[str(k).upper()] = v
    w = WCS(hdr)
    rx, ry = (float(reference_pixel[0].to_value(u.pixel)), float(reference_pixel[1].to_value(u.pixel)))
    lon0, lat0 = w.wcs_pix2world(rx, ry, 0)
    lon1, lat1 = w.wcs_pix2world(rx + 1, ry, 0)
    lon2, lat2 = w.wcs_pix2world(rx, ry + 1, 0)
    sph = coordinate.spherical
    clon, clat = sph.lon.to_value(u.deg), sph.lat.to_value(u.deg)
    cdelt1 = float(scale[0].to_value(u.Unit(meta_wcs["cunit1"]) / u.pixel))
    cdelt2 = float(scale[1].to_value(u.Unit(meta_wcs["cunit2"]) / u.pixel))
    s1 = (cdelt1 * u.Unit(meta_wcs["cunit1"])).to_value(u.deg)
    s2 = (cdelt2 * u.Unit(meta_wcs["cunit2"])).to_value(u.deg)
    if max(abs(s1), abs(s2)) > 0.1 or not np.isfinite([lon0, lat0, lon1, lat1, lon2, lat2]).all():
        return
    err = np.hypot(_wrap180(lon0 - clon) * np.cos(np.deg2rad(clat)), lat0 - clat) / max(abs(s1), abs(s2))

    def sep(lo_a, la_a, lo_b, la_b):
        a1, a2, b1, b2 = (np.deg2rad(v) for v in (lo_a, la_a, lo_b, la_b))
        h = np.sin((b2 - a2) / 2) ** 2 + np.cos(a2) * np.cos(b2) * np.sin((b1 - a1) / 2) ** 2
        return np.rad2deg(2 * np.arcsin(np.sqrt(h)))

    step1 = sep(lon0, lat0, lon1, lat1)
    step2 = sep(lon0, lat0, lon2, lat2)
    # a zenithal (TAN) pixel step of plane size x spans atan(x), a relative O(x^2/3) projection term
    # plus the rounding of two O(360 deg) coordinates differenced over one step
    tol1 = 1e-9 + np.deg2rad(abs(s1)) ** 2 / 2 + _C * _EPS * 360.0 / abs(s1)
    tol2 = 1e-9 + np.deg2rad(abs(s2)) ** 2 / 2 + _C * _EPS * 360.0 / abs(s2)
    # world coordinates are O(360 deg) floats: absolute rounding C eps 360 deg, expressed in pixels
    coord_tol = 1e-9 + _C * _EPS * 360.0 / max(abs(s1), abs(s2))
    # cdelt is the step in the native plane; it equals the sky step at the reference point only where the
    # projection has unit local scale there (zenithal, CAR, MER, CEA, SFL), not e.g. MOL, PAR, HPX, TSC, CSC, QSC
    unit_scale = str(meta_wcs["ctype1"])[-3:].upper() in {"TAN", "SIN", "ARC", "STG", "ZEA", "CAR", "MER", "CEA", "SFL"}
    bad = err > coord_tol or (unit_scale and (abs(step1 - abs(s1)) > tol1 * abs(s1) or abs(step2 - abs(s2)) > tol2 * abs(s2)))
    trigger_if(bad, "SP-WCS-002")
