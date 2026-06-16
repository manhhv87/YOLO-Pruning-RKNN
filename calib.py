#!/usr/bin/env python
"""
calib.py -- image->ground homography for the cabbage robot's forward camera, so cross-track can be
reported in centimetres instead of pixels.

The cabbage robot and the sibling spray robot share the SAME platform and camera arrangement
(Logitech HD C270 @ 640x480). Its camera was calibrated in the sibling thesis (Nguyen Van Son):
- intrinsics K and distortion D from a 4x7 checkerboard (25.5 mm squares, 21 views);
- a ground-plane homography from 4 pixel<->world(mm) correspondences (Bang 1.1), world origin on the
  robot centreline at the near row, +X lateral-right, +Y forward; validated to ~1.55 mm mean error.

We REUSE that homography here (same robot/camera configuration). CAVEAT: the exact camera height/tilt
for the cabbage runs were not re-measured, so centimetre cross-track is reported UNDER THE ASSUMPTION
that the shared, previously-calibrated configuration held during these runs.

px_to_ground_mm(x,y) -> (X_mm, Y_mm) on the ground plane.
crosstrack_cm(coeffs,H,W) / px_cross_to_cm(ct_px,H,W) -> lateral cross-track in cm.
"""
from __future__ import annotations
import numpy as np
import cv2

import guidance_curve as gc

# Calibration is defined at this image resolution (the C270 stream the robot used).
CALIB_W, CALIB_H = 640, 480

# Bang 1.1 (Nguyen Van Son thesis): image pixel <-> ground world (mm), Z=0 ground plane.
IMG_PTS = np.float32([(320, 433), (600, 248), (374, 155), (67, 233)])
WORLD_MM = np.float32([(0.0, 0.0), (301.0, 261.0), (63.0, 602.0), (-301.0, 294.0)])

# intrinsics / distortion (recorded for completeness; the homography below is what maps px->ground)
K = np.array([[797.4981, 0.0, 323.1862], [0.0, 797.8648, 263.6120], [0.0, 0.0, 1.0]])
DIST = np.array([0.0202, 0.2425, 0.0053, -3.4086e-4, 0.0])  # thesis order [k1, k2, k3, p1, p2]

H_IMG2GND = cv2.getPerspectiveTransform(IMG_PTS, WORLD_MM)


def px_to_ground_mm(x, y):
    p = H_IMG2GND @ np.array([float(x), float(y), 1.0])
    return p[0] / p[2], p[1] / p[2]


def px_cross_to_cm(ct_px, H, W):
    """Convert a pixel cross-track (x_row - W/2 at the near row) to cm on the ground."""
    yn = gc.Y_CROSSTRACK * H
    xr, _ = px_to_ground_mm(W / 2.0 + ct_px, yn)
    xc, _ = px_to_ground_mm(W / 2.0, yn)
    return (xr - xc) / 10.0


def crosstrack_cm(coeffs, H, W):
    yn = gc.Y_CROSSTRACK * H
    xr, _ = px_to_ground_mm(gc.x_at(coeffs, yn), yn)
    xc, _ = px_to_ground_mm(W / 2.0, yn)
    return (xr - xc) / 10.0


if __name__ == "__main__":
    # smoke: the homography must reproduce the 4 calibration correspondences
    err = [np.hypot(*(np.subtract(px_to_ground_mm(*ip), wp))) for ip, wp in zip(IMG_PTS, WORLD_MM)]
    print("reprojection error (mm):", [round(e, 3) for e in err], "max", round(max(err), 3))
    print("1 px lateral at near row ~=", round(abs(px_cross_to_cm(1.0, CALIB_H, CALIB_W)), 4), "cm")
    print("100 px cross-track ~=", round(px_cross_to_cm(100.0, CALIB_H, CALIB_W), 2), "cm")
