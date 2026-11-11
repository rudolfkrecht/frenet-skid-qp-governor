#!/usr/bin/env python3
"""
identification_script_v4.py

End-to-end parameter identification for a heavy, low-speed 4WD skid-steer robot
using ROS 2 bag (.db3) logs.

Signals:
  - IMU yaw rate: /vectornav/imu (sensor_msgs/msg/Imu) angular_velocity.z
  - ODrive per-axis: /odrive_axis*/controller_status vel_estimate, torque_estimate
  - Command: /tyr/aut/cmd_vel (geometry_msgs/msg/Twist) linear.x, angular.z
    Used ONLY for sign calibration + gating, not as physical input.

Main fixes / features:
  1) Per-axis wheel-speed sign calibration using corr(w_i, v_cmd) on near-straight segments
  2) Per-axis torque sign calibration using corr(tau_i, vdot) on acceleration segments
  3) Yaw-rate sign calibration using rotate runs: corr(tau_diff, rdot) >= 0
  4) Low-speed longitudinal model (default): drop quadratic drag term c2
        m vdot = (Fgain/Rw)*tau_sum - c0*sgn(v) - c1*v
     Optional: include c2 with --include_c2
  5) Yaw identification uses a robust 2-stage method:
       (i) estimate d0 from decay segments where |tau_diff| is small (actuation ~ off)
       (ii) fit Mgain with d0 fixed (optionally fit d1, but default keeps d1=0 for low-speed)

Models:
  v = Rw/2 * (wR + wL)
  tau_sum  = tauL + tauR,  tau_diff = tauR - tauL

  Longitudinal (default):
    m vdot = (1/Rw)*Fgain*tau_sum - c0*sgn(v) - c1*v        [ - c2*v|v| if enabled ]

  Yaw (2-stage):
    Iz rdot ≈ -d0*r                       on decay segments (|tau_diff| small)
    Iz rdot + d0*r = (b/Rw)*Mgain*tau_diff   (optionally - d1*|v|*r)

Outputs:
  JSON with sign maps, estimated parameters, and metadata.

Example:
  python3 identification_script_v4.py \
      --root identification_carpet \
      --left_axes 1,3 --right_axes 2,4 \
      --vel_units radps \
      --m 600 --Rw 0.15 --b 0.5 --Iz 200 \
      --out identified_params_v4.json
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict

TOPIC_IMU = "/vectornav/imu"
TOPIC_CMD = "/tyr/aut/cmd_vel"
AXES = [1, 2, 3, 4]
TOPIC_CTRL = {a: f"/odrive_axis{a}/controller_status" for a in AXES}


# ------------------------- ROS2 bag helpers -------------------------

def read_topics_and_types(db_path: str) -> Dict[str, str]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name, type FROM topics")
    out = {n: t for (n, t) in cur.fetchall()}
    con.close()
    return out


def iter_messages(db_path: str, wanted: set):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT topics.name, messages.timestamp, messages.data
        FROM messages JOIN topics ON messages.topic_id = topics.id
        ORDER BY messages.timestamp ASC
    """)
    for name, t_ns, data in cur:
        if name in wanted:
            yield name, int(t_ns), data
    con.close()


def classify(folder: str) -> str:
    f = folder.lower()
    if "rotate" in f:
        return "rotate"
    if "straight" in f:
        return "straight"
    if "circle" in f or "circl" in f or "smooth" in f:
        return "circle"
    return "unknown"


# ------------------------- math utilities -------------------------

def fd(x: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference derivative with simple boundary handling."""
    dx = np.zeros_like(x)
    if len(x) < 3 or dt <= 0:
        return dx
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def sign_from_corr(x: np.ndarray, y: np.ndarray) -> int:
    """Return +1 or -1 to make correlation corr(x, y) nonnegative."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    if n < 30:
        return +1
    x = x[:n]
    y = y[:n]
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return +1
    return +1 if c >= 0 else -1


# ------------------------- message extraction -------------------------

def get_d(dct, path: str):
    cur = dct
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def extract_imu_rz(msg) -> Optional[float]:
    d = message_to_ordereddict(msg)
    rz = get_d(d, "angular_velocity.z")
    return float(rz) if isinstance(rz, (int, float)) else None


def extract_cmd(msg) -> Tuple[Optional[float], Optional[float]]:
    d = message_to_ordereddict(msg)
    vx = get_d(d, "linear.x")
    wz = get_d(d, "angular.z")
    vx = float(vx) if isinstance(vx, (int, float)) else None
    wz = float(wz) if isinstance(wz, (int, float)) else None
    return vx, wz


def extract_ctrl(msg) -> Tuple[Optional[float], Optional[float]]:
    d = message_to_ordereddict(msg)
    ve = d.get("vel_estimate", None)
    te = d.get("torque_estimate", None)
    ve = float(ve) if isinstance(ve, (int, float)) else None
    te = float(te) if isinstance(te, (int, float)) else None
    return ve, te


# ------------------------- bag loading -------------------------

def load_bag(
    db_path: str,
    fs: float,
    left_axes: List[int],
    right_axes: List[int],
    vel_units: str,
    debug: bool = True,
) -> Optional[Dict[str, np.ndarray]]:

    types = read_topics_and_types(db_path)
    wanted = {TOPIC_IMU, TOPIC_CMD, *TOPIC_CTRL.values()}
    available = wanted.intersection(types.keys())
    if TOPIC_IMU not in available:
        if debug:
            print(f"[WARN] Missing IMU in {db_path}")
        return None

    msg_cls = {tp: get_message(types[tp]) for tp in available}

    t0 = None
    t_imu, rz = [], []
    t_cmd, vcmd, wcmd = [], [], []
    t_w = {a: [] for a in AXES}
    w = {a: [] for a in AXES}
    t_tau = {a: [] for a in AXES}
    tau = {a: [] for a in AXES}

    for tp, t_ns, raw in iter_messages(db_path, available):
        if t0 is None:
            t0 = t_ns
        t = (t_ns - t0) * 1e-9

        msg = deserialize_message(raw, msg_cls[tp])

        if tp == TOPIC_IMU:
            r = extract_imu_rz(msg)
            if r is not None:
                t_imu.append(t)
                rz.append(r)

        elif tp == TOPIC_CMD:
            vx, wz_ = extract_cmd(msg)
            if vx is not None and wz_ is not None:
                t_cmd.append(t)
                vcmd.append(vx)
                wcmd.append(wz_)

        else:
            m = re.match(r"^/odrive_axis(\d+)/controller_status$", tp)
            if m:
                a = int(m.group(1))
                ve, te = extract_ctrl(msg)
                if ve is not None:
                    t_w[a].append(t)
                    w[a].append(ve)
                if te is not None:
                    t_tau[a].append(t)
                    tau[a].append(te)

    if debug:
        print(
            f"[COUNT] {Path(db_path).parent.name}: imu={len(t_imu)} cmd={len(t_cmd)} "
            f"w={[len(t_w[a]) for a in AXES]} tau={[len(t_tau[a]) for a in AXES]}"
        )

    # minimal viability checks
    if len(t_imu) < 20:
        return None
    if not all(len(t_w[a]) > 5 for a in left_axes + right_axes):
        return None

    # Determine overlap window for uniform resampling
    starts = [t_imu[0]]
    ends = [t_imu[-1]]

    if t_cmd:
        starts.append(t_cmd[0])
        ends.append(t_cmd[-1])

    for a in left_axes + right_axes:
        starts.append(t_w[a][0])
        ends.append(t_w[a][-1])

    for a in left_axes + right_axes:
        if len(t_tau[a]) > 5:
            starts.append(t_tau[a][0])
            ends.append(t_tau[a][-1])

    t_start = max(starts)
    t_end = min(ends)
    if t_end <= t_start + 0.3:
        return None

    dt = 1.0 / fs
    tg_abs = np.arange(t_start, t_end, dt)
    tg = tg_abs - tg_abs[0]

    # interpolate signals onto tg_abs
    r_g = np.interp(tg_abs, np.array(t_imu), np.array(rz))

    def interp_axis(tt, xx):
        return np.interp(tg_abs, np.array(tt), np.array(xx))

    w_axis = {a: interp_axis(t_w[a], w[a]) for a in (left_axes + right_axes)}
    tau_axis = {
        a: (interp_axis(t_tau[a], tau[a]) if len(t_tau[a]) > 5 else np.full_like(tg_abs, np.nan))
        for a in (left_axes + right_axes)
    }

    # convert speed units to rad/s
    if vel_units == "revps":
        for a in w_axis:
            w_axis[a] = w_axis[a] * (2.0 * np.pi)
    elif vel_units != "radps":
        raise ValueError("vel_units must be 'radps' or 'revps'")

    if t_cmd:
        vcmd_g = np.interp(tg_abs, np.array(t_cmd), np.array(vcmd))
        wcmd_g = np.interp(tg_abs, np.array(t_cmd), np.array(wcmd))
    else:
        vcmd_g = np.zeros_like(tg_abs)
        wcmd_g = np.zeros_like(tg_abs)

    return {
        "t": tg,
        "r": r_g,
        "v_cmd": vcmd_g,
        "w_cmd": wcmd_g,
        "w_axis": w_axis,
        "tau_axis": tau_axis,
    }


def recompute_aggregates(d: Dict[str, np.ndarray], left_axes: List[int], right_axes: List[int]):
    d["wL"] = np.mean(np.vstack([d["w_axis"][a] for a in left_axes]), axis=0)
    d["wR"] = np.mean(np.vstack([d["w_axis"][a] for a in right_axes]), axis=0)
    d["tauL"] = np.nansum(np.vstack([d["tau_axis"][a] for a in left_axes]), axis=0)
    d["tauR"] = np.nansum(np.vstack([d["tau_axis"][a] for a in right_axes]), axis=0)
    d["tau_sum"] = d["tauL"] + d["tauR"]
    d["tau_diff"] = d["tauR"] - d["tauL"]


# ------------------------- identification -------------------------

def fit_longitudinal(data_list: List[Dict[str, np.ndarray]], m: float, Rw: float, include_c2: bool):
    """
    Low-speed longitudinal model (default):
      m vdot = (Fgain/Rw)*tau_sum - c0*sgn(v) - c1*v   [- c2*v|v| if include_c2]
    """
    X_all, y_all = [], []
    for d in data_list:
        t = d["t"]
        v = (Rw / 2.0) * (d["wR"] + d["wL"])
        dt = float(np.median(np.diff(t)))
        vdot = fd(v, dt)

        cols = [
            (1.0 / Rw) * d["tau_sum"],     # Fgain
            -np.sign(v + 1e-6),            # c0
            -v,                             # c1
        ]
        names = ["Fgain", "c0", "c1"]

        if include_c2:
            cols.append(-(v * np.abs(v)))
            names.append("c2")

        X = np.column_stack(cols)
        y = m * vdot

        X_all.append(X)
        y_all.append(y)

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {names[i]: float(theta[i]) for i in range(len(names))}


def fit_yaw_unconstrained(data_list: List[Dict[str, np.ndarray]], Iz: float, b: float, Rw: float):
    """
    Unconstrained yaw LS (fallback):
      Iz rdot = (b/Rw)*Mgain*tau_diff - d0*r - d1*|v|*r
    """
    X_all, y_all = [], []
    for d in data_list:
        t = d["t"]
        dt = float(np.median(np.diff(t)))
        v = (Rw / 2.0) * (d["wR"] + d["wL"])
        r = d["r"]
        rdot = fd(r, dt)

        X = np.column_stack([
            (b / Rw) * d["tau_diff"],   # Mgain
            -r,                         # d0
            -(np.abs(v) * r),           # d1
        ])
        y = Iz * rdot
        X_all.append(X)
        y_all.append(y)

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Mgain, d0, d1 = theta.tolist()
    return {"Mgain": float(Mgain), "d0": float(d0), "d1": float(d1)}


def estimate_d0_from_decay(
    datasets: List[Dict[str, np.ndarray]],
    Iz: float,
    tau_diff_thresh: float,
    min_samples: int = 200,
) -> Optional[float]:
    """
    Estimate yaw damping d0 from segments where |tau_diff| is small (actuation ~ off):
      Iz rdot ≈ -d0 r
    Fit y = -d0*r in LS sense over selected samples.

    Returns d0_hat (float) if successful, else None.
    """
    rr = []
    yy = []
    for d in datasets:
        t = d["t"]
        dt = float(np.median(np.diff(t)))
        r = d["r"]
        rdot = fd(r, dt)
        tau_diff = d["tau_diff"]

        mask = np.isfinite(tau_diff) & (np.abs(tau_diff) < tau_diff_thresh)
        mask &= np.isfinite(r) & np.isfinite(rdot)
        mask &= (np.abs(r) > 1e-3)

        if np.sum(mask) < 30:
            continue

        rr.append(r[mask])
        yy.append(Iz * rdot[mask])

    if not rr:
        return None

    r_all = np.concatenate(rr)
    y_all = np.concatenate(yy)

    if len(r_all) < min_samples:
        return None

    X = (-r_all).reshape(-1, 1)
    d0_hat, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    return float(d0_hat[0])


def fit_yaw_with_fixed_d0(
    data_list: List[Dict[str, np.ndarray]],
    Iz: float,
    b: float,
    Rw: float,
    d0_fixed: float,
    fit_d1: bool = False,
) -> Dict[str, float]:
    """
    Fit yaw actuation with d0 fixed:
      Iz rdot + d0*r = (b/Rw)*Mgain*tau_diff   [- d1*|v|*r if fit_d1]
    """
    X_all, y_all = [], []
    for d in data_list:
        t = d["t"]
        dt = float(np.median(np.diff(t)))
        v = (Rw / 2.0) * (d["wR"] + d["wL"])
        r = d["r"]
        rdot = fd(r, dt)

        y = Iz * rdot + d0_fixed * r

        cols = [(b / Rw) * d["tau_diff"]]
        names = ["Mgain"]
        if fit_d1:
            cols.append(-(np.abs(v) * r))
            names.append("d1")

        X = np.column_stack(cols)
        X_all.append(X)
        y_all.append(y)

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    out = {"Mgain": float(theta[0]), "d0": float(d0_fixed)}
    out["d1"] = float(theta[1]) if fit_d1 else 0.0
    return out


def choose_yaw_sign(rotate_list: List[Dict[str, np.ndarray]]) -> int:
    """
    Choose yaw-rate sign s_r so that corr(tau_diff, rdot) >= 0 on rotate runs.
    If negative correlation is observed, flip r (s_r = -1).
    """
    if not rotate_list:
        return +1

    xs = []
    ys = []
    for d in rotate_list:
        t = d["t"]
        dt = float(np.median(np.diff(t)))
        rdot = fd(d["r"], dt)

        # gate to near-zero linear command if available
        mask = np.abs(d.get("v_cmd", np.zeros_like(d["r"]))) < 0.05
        if np.sum(mask) < 80:
            mask = slice(None)

        xs.append(d["tau_diff"][mask])
        ys.append(rdot[mask])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    n = min(len(x), len(y))
    if n < 50:
        return +1

    c = np.corrcoef(x[:n], y[:n])[0, 1]
    if np.isnan(c):
        return +1
    return +1 if c >= 0 else -1


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--out", default="identified_params_v4.json")
    ap.add_argument("--left_axes", default="1,3")
    ap.add_argument("--right_axes", default="2,4")
    ap.add_argument("--vel_units", default="radps", choices=["radps", "revps"])

    ap.add_argument("--m", type=float, default=600.0)
    ap.add_argument("--Rw", type=float, default=0.15)
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--Iz", type=float, default=200.0)

    # calibration gates
    ap.add_argument("--calib_wcmd_thresh", type=float, default=0.05)
    ap.add_argument("--calib_min_samples", type=int, default=80)
    ap.add_argument("--calib_vcmd_min", type=float, default=0.02)
    ap.add_argument("--calib_acc_min", type=float, default=0.05)

    # model knobs
    ap.add_argument("--include_c2", action="store_true",
                    help="Include quadratic drag c2*v|v| in longitudinal fit (not recommended at low speed).")
    ap.add_argument("--fit_d1", action="store_true",
                    help="Fit d1 term in yaw with fixed d0 (usually weakly identifiable at low speed).")

    # yaw decay identification knobs
    ap.add_argument("--tau_diff_decay_thresh", type=float, default=1.0,
                    help="Threshold for |tau_diff| to treat yaw as unactuated (for d0 estimation).")
    ap.add_argument("--min_decay_samples", type=int, default=200)

    args = ap.parse_args()

    left_axes = [int(x) for x in args.left_axes.split(",")]
    right_axes = [int(x) for x in args.right_axes.split(",")]
    axes_used = left_axes + right_axes

    root = Path(args.root).expanduser()
    db3s = sorted(root.rglob("*.db3"))
    if not db3s:
        raise SystemExit("No .db3 files found")

    straight, rotate, circle = [], [], []
    for db in db3s:
        kind = classify(db.parent.name)
        d = load_bag(str(db), args.fs, left_axes, right_axes, args.vel_units, debug=True)
        if d is None:
            continue
        if kind == "straight":
            straight.append(d)
        elif kind == "rotate":
            rotate.append(d)
        elif kind == "circle":
            circle.append(d)

    # --- wheel speed sign calibration vs v_cmd on near-straight segments ---
    w_sign = {a: +1 for a in axes_used}
    calib_sets = []
    for d in straight:
        mask = (np.abs(d["w_cmd"]) < args.calib_wcmd_thresh) & (np.abs(d["v_cmd"]) > args.calib_vcmd_min)
        if np.sum(mask) >= args.calib_min_samples:
            calib_sets.append((d, mask))
    if calib_sets:
        for a in axes_used:
            x = np.concatenate([d["w_axis"][a][mask] for d, mask in calib_sets])
            y = np.concatenate([d["v_cmd"][mask] for d, mask in calib_sets])
            w_sign[a] = sign_from_corr(x, y)
    print("[CAL] wheel speed sign map:", w_sign)

    for d in (straight + rotate + circle):
        for a, s in w_sign.items():
            d["w_axis"][a] = s * d["w_axis"][a]
        recompute_aggregates(d, left_axes, right_axes)

    # --- torque sign calibration vs vdot on straight acceleration segments ---
    tau_sign = {a: +1 for a in axes_used}
    accel_sets = []
    for d in straight:
        v = (args.Rw / 2.0) * (d["wR"] + d["wL"])
        dt = float(np.median(np.diff(d["t"])))
        vdot = fd(v, dt)

        mask = (np.abs(d["w_cmd"]) < args.calib_wcmd_thresh) & (np.abs(vdot) > args.calib_acc_min)
        if np.sum(mask) >= args.calib_min_samples:
            accel_sets.append((d, mask, vdot))

    if accel_sets:
        for a in axes_used:
            x = np.concatenate([d["tau_axis"][a][mask] for d, mask, _ in accel_sets])
            y = np.concatenate([vdot[mask] for _, mask, vdot in accel_sets])
            tau_sign[a] = sign_from_corr(x, y)
    print("[CAL] torque sign map:", tau_sign)

    for d in (straight + rotate + circle):
        for a, s in tau_sign.items():
            d["tau_axis"][a] = s * d["tau_axis"][a]
        recompute_aggregates(d, left_axes, right_axes)

    # --- yaw-rate sign calibration using rotate runs ---
    r_sign = choose_yaw_sign(rotate)
    print("[CAL] yaw rate sign:", r_sign)
    for d in (straight + rotate + circle):
        d["r"] = r_sign * d["r"]

    # --- fit longitudinal (low-speed default) ---
    straight_fit = fit_longitudinal(straight, m=args.m, Rw=args.Rw, include_c2=args.include_c2) if straight else None

    # --- yaw: estimate d0 from decay segments, then fit Mgain with d0 fixed ---
    yaw_set = rotate + circle

    d0_hat = estimate_d0_from_decay(
        yaw_set,
        Iz=args.Iz,
        tau_diff_thresh=args.tau_diff_decay_thresh,
        min_samples=args.min_decay_samples,
    )

    if d0_hat is None or not np.isfinite(d0_hat) or d0_hat <= 0:
        print("[WARN] could not estimate positive d0 from decay; falling back to unconstrained yaw fit")
        yaw_fit = fit_yaw_unconstrained(yaw_set, Iz=args.Iz, b=args.b, Rw=args.Rw) if yaw_set else None
        d0_source = "unconstrained_ls"
    else:
        print("[CAL] estimated yaw damping d0 from decay:", d0_hat)
        yaw_fit = fit_yaw_with_fixed_d0(
            yaw_set,
            Iz=args.Iz,
            b=args.b,
            Rw=args.Rw,
            d0_fixed=d0_hat,
            fit_d1=args.fit_d1,
        ) if yaw_set else None
        d0_source = "decay_ls"

    # --- save results ---
    result = {
        "meta": vars(args),
        "bags_used": {"straight": len(straight), "rotate": len(rotate), "circle": len(circle)},
        "calibration": {
            "wheel_speed_sign_map": w_sign,
            "torque_sign_map": tau_sign,
            "yaw_rate_sign": r_sign,
            "wheel_speed_sign_method": "corr(w_axis_i, v_cmd) >= 0 on near-straight commanded segments",
            "torque_sign_method": "corr(tau_axis_i, vdot) >= 0 on near-straight acceleration segments",
            "yaw_sign_method": "corr(tau_diff, rdot) >= 0 on rotate runs (flip r if negative)",
        },
        "straight_fit": straight_fit,
        "yaw_fit": yaw_fit,
        "yaw_fit_notes": {
            "method": "two_stage" if d0_source == "decay_ls" else "fallback_unconstrained",
            "d0_source": d0_source,
            "tau_diff_decay_thresh": args.tau_diff_decay_thresh,
            "fit_d1": bool(args.fit_d1),
        },
        "notes": {
            "v_source": "v=Rw/2*(wR+wL) from ODrive vel_estimate after per-axis speed sign calibration",
            "torque_source": "ODrive torque_estimate after per-axis torque sign calibration",
            "r_source": "IMU angular_velocity.z after yaw-rate sign calibration",
            "recommended_low_speed": "omit c2 (default) and typically keep d1=0 unless wide velocity range is available",
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n[OK] wrote", args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
