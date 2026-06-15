#!/usr/bin/env python
"""
robot_nav.py -- on-robot real-time crop-row following with a certified TRUST GATE (Jetson-side).

The closed-loop node that turns the perception + conformal trust layer into steering, and LOGS
everything for the controlled closed-loop experiment (E6-real). It runs the SAME readout + trust
score as the offline eval, so the paper's claims and the robot use one pipeline.

Pipeline per frame:
  camera -> detector (eval_guidance.detect_with_logits) -> robust guidance line (irls/ransac)
         -> trust score s (LOO heading dispersion) -> conformal half-width h = qhat * s
         -> GATE: proceed (h<=tau_p) | slow (tau_p<h<=tau_s) | abstain/stop (h>tau_s)
         -> differential-drive command (heading + cross-track) -> serial to MCU
         -> log row (t, heading, cross_track_px/cm, s, h, gate, v_left, v_right)

ARMS (set --arm): nogate (act every frame) | fixedconf (gate on detector confidence) | conformal.

HARDWARE HOOKS to wire to YOUR robot (marked TODO): open_camera(), send_wheel_speeds(),
load_homography(). The perception + gate + control + logging are complete and hardware-agnostic.

Calibrate qhat once offline (eval_conformal / a val set) and pass --qhat; or use --tau-* directly.
Run, e.g.:
  python robot_nav.py --weights best.pt --arm conformal --qhat 3.0 --tau-p 1.5 --tau-s 3.0 \
      --homography H.npy --speed 0.4 --log runs_robot/conformal_run1.csv
"""
from __future__ import annotations
import argparse, csv, math, time

import numpy as np

try:
    import cv2, torch
    from ultralytics import YOLO
    from eval_guidance import detect_with_logits, guidance_line, loo_heading_dispersion  # one pipeline
    import guidance_curve as gc
except Exception as e:
    cv2 = None; _ERR = e


# ----------------------------------------------------------------------------- #
# HARDWARE HOOKS -- wire these to your robot (Son/Dung thesis stack: Jetson + Arduino Mega)
# ----------------------------------------------------------------------------- #
def open_camera(src=0, w=1280, h=720):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def send_wheel_speeds(ser, v_left, v_right):
    """TODO: encode (v_left, v_right) in YOUR serial protocol to the Arduino Mega.
    Placeholder prints; replace with e.g. ser.write(f'L{v_left:.3f}R{v_right:.3f}\\n'.encode())."""
    if ser is not None:
        try:
            ser.write(f"L{v_left:.3f}R{v_right:.3f}\n".encode())
        except Exception:
            pass


def load_homography(path):
    """Image->ground (cm) 3x3 from the camera calibration (Son thesis Ch.1.3). None -> px only."""
    return np.load(path) if path else None


def px_to_cm_lateral(x_px, y_px, H):
    p = H @ np.array([x_px, y_px, 1.0]); p = p[:2] / p[2]
    return float(p[0])  # ground-frame lateral coord (cm)


# ----------------------------------------------------------------------------- #
# control: heading + cross-track -> differential-drive wheel speeds
# ----------------------------------------------------------------------------- #
def control(heading_deg, crosstrack_norm, speed, k_h=0.020, k_e=0.8, wheel_base=1.0):
    """Proportional row-following. heading_deg = atan(a) (deg); crosstrack_norm = lateral offset of
    the row from robot centre, normalised to [-1,1]. Returns (v_left, v_right) in [-1,1]*speed."""
    omega = k_h * heading_deg + k_e * crosstrack_norm          # turn rate command
    v_left = speed - omega * wheel_base / 2.0
    v_right = speed + omega * wheel_base / 2.0
    m = max(1.0, abs(v_left) / speed if speed else 1, abs(v_right) / speed if speed else 1)
    return v_left / m, v_right / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", choices=["nogate", "fixedconf", "conformal"], default="conformal")
    ap.add_argument("--readout", default="irls", choices=["irls", "ransac", "calm", "equalLS"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--camera", default="0", help="cv2 VideoCapture src (index or path)")
    ap.add_argument("--serial", default=None, help="serial port to the MCU, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--homography", default=None, help=".npy image->ground(cm)")
    ap.add_argument("--speed", type=float, default=0.4, help="base speed (0..1)")
    ap.add_argument("--target-offset", type=float, default=0.0,
                    help="desired lateral offset of the row from robot centre (px): 0 = straddle the row; "
                         "+half-row-spacing if the robot drives in the furrow")
    ap.add_argument("--qhat", type=float, default=3.0, help="conformal multiplier (offline-calibrated)")
    ap.add_argument("--tau-p", type=float, default=1.5, help="proceed if half-width h<=tau_p (deg)")
    ap.add_argument("--tau-s", type=float, default=3.0, help="slow if tau_p<h<=tau_s; abstain if h>tau_s")
    ap.add_argument("--conf-gate", type=float, default=0.5, help="fixedconf arm: proceed if mean conf>=this")
    ap.add_argument("--log", default="robot_run.csv")
    ap.add_argument("--max-sec", type=float, default=0, help="auto-stop after N s (0=until 'q')")
    args = ap.parse_args()
    if cv2 is None:
        raise SystemExit(f"needs torch+ultralytics+opencv on the robot: {_ERR}")
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"

    model = YOLO(args.weights); model.model.to(args.device)
    H = load_homography(args.homography)
    ser = None
    if args.serial:
        import serial  # pyserial
        ser = serial.Serial(args.serial, args.baud, timeout=0.05)
    cam_src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = open_camera(cam_src)

    f = open(args.log, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "frame_ok", "n_central", "heading_deg", "crosstrack_px", "crosstrack_cm",
                "s_loo", "half_width", "conf_mean", "gate", "v_left", "v_right"])
    print(f"[robot_nav] arm={args.arm} readout={args.readout} -> logging {args.log}. Ctrl-C to stop.")
    t0 = time.time(); last_heading = 0.0
    try:
        while True:
            ok, frame = cap.read()
            t = time.time() - t0
            if args.max_sec and t > args.max_sec:
                break
            if not ok:
                continue
            Wd = frame.shape[1]
            det = detect_with_logits(model, frame, args.imgsz, args.conf, 0.5, args.device)
            heading = last_heading; ct_px = 0.0; ct_cm = ""; s = float("nan"); hw = float("nan")
            conf_mean = 0.0; n_central = 0; gate = "abstain"; vL = vR = 0.0
            gl = guidance_line(det, args.readout) if det is not None else None
            if gl is not None:
                # curved central row from the central-row points (letterbox px); heading = look-ahead tangent
                cyn = gl["cy"].cpu().numpy(); cxn = gl["cx"].cpu().numpy()
                coeffs = gc.fit_band(cyn, cxn, args.imgsz, degree=2)
                heading = gc.heading_lookahead(coeffs, args.imgsz); last_heading = heading
                n_central = int(gl["cx"].numel())
                conf_mean = float(gl.get("conf_mean", 0.0))
                s = float(gl.get("s_loo", float("nan")))
                hw = args.qhat * s if math.isfinite(s) else float("inf")
                # cross-track at the near row (letterbox -> original px), minus the desired offset
                y_ct = gc.Y_CROSSTRACK * args.imgsz
                x_ct_lb = gc.x_at(coeffs, y_ct)
                x_ct = (x_ct_lb - det["left"]) / det["r"]; y_ct_o = (y_ct - det["top"]) / det["r"]
                ct_px = (x_ct - Wd / 2.0) - args.target_offset
                if H is not None:
                    ct_cm = px_to_cm_lateral(x_ct, y_ct_o, H)
                ct_norm = max(-1.0, min(1.0, ct_px / (Wd / 2.0)))
                # GATE
                if args.arm == "nogate":
                    gate = "proceed"
                elif args.arm == "fixedconf":
                    gate = "proceed" if conf_mean >= args.conf_gate else "abstain"
                else:  # conformal
                    gate = "proceed" if hw <= args.tau_p else ("slow" if hw <= args.tau_s else "abstain")
                spd = {"proceed": args.speed, "slow": 0.5 * args.speed, "abstain": 0.0}[gate]
                if gate == "abstain":
                    vL = vR = 0.0
                else:
                    vL, vR = control(heading, ct_norm, spd)
            send_wheel_speeds(ser, vL, vR)
            w.writerow([f"{t:.3f}", int(gl is not None), n_central, f"{heading:.2f}", f"{ct_px:.1f}",
                        (f"{ct_cm:.2f}" if ct_cm != "" else ""), (f"{s:.3f}" if math.isfinite(s) else ""),
                        (f"{hw:.3f}" if math.isfinite(hw) else ""), f"{conf_mean:.3f}", gate,
                        f"{vL:.3f}", f"{vR:.3f}"])
    except KeyboardInterrupt:
        pass
    finally:
        send_wheel_speeds(ser, 0.0, 0.0)
        cap.release(); f.close()
        if ser is not None:
            ser.close()
        print(f"[robot_nav] stopped; wrote {args.log}")


if __name__ == "__main__":
    main()
