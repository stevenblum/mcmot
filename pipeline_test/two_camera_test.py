#!/usr/bin/env python3
"""
Two-camera stereo depth test.

Supports two rectification modes:
- pinhole: assumed pinhole intrinsics + Essential matrix RANSAC + stereoRectify (has Q, pseudo-metric depth)
- uncalibrated: Fundamental matrix RANSAC + stereoRectifyUncalibrated (relative depth only)

Displays:
- raw left camera
- raw right camera
- dense depth map
- calibration match image (exact frames used for calibration)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def open_camera(device_id: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera on device id {device_id}.")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def read_pair(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
) -> tuple[np.ndarray, np.ndarray]:
    ok_left, frame_left = left_cap.read()
    ok_right, frame_right = right_cap.read()
    if not ok_left or frame_left is None:
        raise RuntimeError("Failed to read frame from left camera.")
    if not ok_right or frame_right is None:
        raise RuntimeError("Failed to read frame from right camera.")
    if frame_left.shape[:2] != frame_right.shape[:2]:
        raise RuntimeError(
            "Left and right camera resolutions do not match. "
            f"Left={frame_left.shape[:2]}, Right={frame_right.shape[:2]}"
        )
    return frame_left, frame_right


def match_features(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    max_features: int = 7000,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=6)
    kp_left, des_left = orb.detectAndCompute(gray_left, None)
    kp_right, des_right = orb.detectAndCompute(gray_right, None)
    if des_left is None or des_right is None:
        raise RuntimeError("Feature descriptors not found in one or both frames.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_left, des_right, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.85 * n.distance:
            good.append(m)

    if len(good) < 12:
        matcher_cc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cc_matches = matcher_cc.match(des_left, des_right)
        cc_matches = sorted(cc_matches, key=lambda m: m.distance)
        good = cc_matches[: min(1000, len(cc_matches))]

    if len(good) < 12:
        raise RuntimeError(f"Too few good matches for calibration: {len(good)}")

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return kp_left, kp_right, good, pts_left, pts_right


def draw_match_image(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    kp_left: list[cv2.KeyPoint],
    kp_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    total_matches: int,
    inlier_matches: int,
    title: str,
    max_draw: int = 250,
) -> np.ndarray:
    if len(matches) > max_draw:
        matches = sorted(matches, key=lambda m: m.distance)[:max_draw]

    match_vis = cv2.drawMatches(
        frame_left,
        kp_left,
        frame_right,
        kp_right,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.putText(
        match_vis,
        f"{title} | matches={total_matches}, inliers={inlier_matches}, drawn={len(matches)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return match_vis


def calibrate_uncalibrated(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    h, w = gray_left.shape

    kp_left, kp_right, good_matches, pts_left, pts_right = match_features(gray_left, gray_right)
    total = len(good_matches)

    F, mask_f = cv2.findFundamentalMat(
        pts_left.reshape(-1, 2),
        pts_right.reshape(-1, 2),
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.5,
        confidence=0.999,
    )
    if F is None or mask_f is None:
        raise RuntimeError("RANSAC failed to estimate a valid fundamental matrix.")

    inlier_mask = mask_f.ravel().astype(bool)
    pts_left_in = pts_left[inlier_mask].reshape(-1, 2)
    pts_right_in = pts_right[inlier_mask].reshape(-1, 2)
    if len(pts_left_in) < 12:
        raise RuntimeError(f"Too few inliers after fundamental RANSAC: {len(pts_left_in)}")

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_left_in,
        pts_right_in,
        F,
        imgSize=(w, h),
        threshold=3.0,
    )
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed to produce rectification homographies.")

    inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
    match_vis = draw_match_image(
        frame_left,
        frame_right,
        kp_left,
        kp_right,
        inlier_matches,
        total_matches=total,
        inlier_matches=len(inlier_matches),
        title="Uncalibrated RANSAC",
    )

    print(f"[Uncalibrated RANSAC] matches={total}, inliers={len(inlier_matches)}")
    return H1, H2, match_vis


def calibrate_calibrated(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    k_left: np.ndarray,
    dist_left: np.ndarray,
    k_right: np.ndarray,
    dist_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    h, w = gray_left.shape

    kp_left, kp_right, good_matches, pts_left, pts_right = match_features(gray_left, gray_right)
    total = len(good_matches)

    pts_left_n = cv2.undistortPoints(pts_left, k_left, dist_left)
    pts_right_n = cv2.undistortPoints(pts_right, k_right, dist_right)

    best: dict | None = None
    for threshold in (0.003, 0.006, 0.01):
        E, mask_e = cv2.findEssentialMat(
            pts_left_n,
            pts_right_n,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=threshold,
        )
        if E is None or mask_e is None:
            continue

        candidates = [E] if E.shape == (3, 3) else [E[i : i + 3, :] for i in range(0, E.shape[0], 3)]
        for E_i in candidates:
            try:
                inliers, R, t, pose_mask = cv2.recoverPose(E_i, pts_left_n, pts_right_n, mask=mask_e)
            except cv2.error:
                continue

            inliers = int(inliers)
            if inliers <= 0:
                continue

            inlier_mask = mask_e.ravel().astype(bool)
            if pose_mask is not None and pose_mask.size == mask_e.size:
                inlier_mask &= pose_mask.ravel().astype(bool)

            if best is None or inliers > best["inliers"]:
                best = {
                    "inliers": inliers,
                    "R": R,
                    "t": t,
                    "inlier_mask": inlier_mask,
                }

    if best is None or best["inliers"] < 12:
        raise RuntimeError("Pinhole RANSAC failed: no valid Essential matrix pose recovered.")

    R1, R2, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=k_left,
        distCoeffs1=dist_left,
        cameraMatrix2=k_right,
        distCoeffs2=dist_right,
        imageSize=(w, h),
        R=best["R"],
        T=best["t"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        k_left, dist_left, R1, P1_rect, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        k_right, dist_right, R2, P2_rect, (w, h), cv2.CV_32FC1
    )

    inlier_matches = [m for i, m in enumerate(good_matches) if best["inlier_mask"][i]]
    match_vis = draw_match_image(
        frame_left,
        frame_right,
        kp_left,
        kp_right,
        inlier_matches,
        total_matches=total,
        inlier_matches=len(inlier_matches),
        title="Pinhole RANSAC",
    )

    print(f"[Pinhole RANSAC] matches={total}, inliers={len(inlier_matches)}")
    return map1x, map1y, map2x, map2y, Q, match_vis


def estimate_pinhole_intrinsics(
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    fov_deg = float(fov_deg)
    if not (10.0 <= fov_deg <= 170.0):
        raise ValueError(f"fov_deg must be in [10, 170], got {fov_deg}")

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    fx = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx

    k = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((1, 5), dtype=np.float64)
    return k, dist


def create_matcher(kind: str) -> cv2.StereoMatcher:
    if kind == "bm":
        matcher = cv2.StereoBM_create(numDisparities=16 * 8, blockSize=15)
        matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher.setPreFilterSize(9)
        matcher.setPreFilterCap(31)
        matcher.setTextureThreshold(10)
        matcher.setUniquenessRatio(12)
        matcher.setSpeckleWindowSize(100)
        matcher.setSpeckleRange(16)
        matcher.setDisp12MaxDiff(1)
        matcher.setMinDisparity(0)
        return matcher

    block_size = 5
    num_disparities = 16 * 10
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_dense_disparity(
    matcher: cv2.StereoMatcher,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
) -> np.ndarray:
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    return matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0


def _fallback_disparity_colormap(disparity: np.ndarray) -> np.ndarray:
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_TURBO)


def depth_visualization_calibrated(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3d[:, :, 2]
    finite = np.isfinite(depth)
    if np.count_nonzero(finite) < 500:
        return _fallback_disparity_colormap(disparity)

    if np.nanmedian(depth[finite]) < 0:
        depth = -depth

    valid = finite & (depth > 0.0) & (disparity > 0.5)
    if np.count_nonzero(valid) < 500:
        return _fallback_disparity_colormap(disparity)

    depth_vals = depth[valid]
    lo = float(np.percentile(depth_vals, 5))
    hi = float(np.percentile(depth_vals, 95))
    if hi <= lo:
        hi = lo + 1e-6

    depth_clip = np.clip(depth, lo, hi)
    depth_norm = ((depth_clip - lo) / (hi - lo)).astype(np.float32)
    depth_norm[~valid] = 1.0
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def depth_visualization_relative(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0.5)
    if np.count_nonzero(valid) < 500:
        return _fallback_disparity_colormap(disparity)

    rel_depth = np.zeros_like(disparity, dtype=np.float32)
    rel_depth[valid] = 1.0 / disparity[valid]
    depth_vals = rel_depth[valid]

    lo = float(np.percentile(depth_vals, 5))
    hi = float(np.percentile(depth_vals, 95))
    if hi <= lo:
        hi = lo + 1e-6

    depth_clip = np.clip(rel_depth, lo, hi)
    depth_norm = ((depth_clip - lo) / (hi - lo)).astype(np.float32)
    depth_norm[~valid] = 1.0
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-camera stereo depth test.")
    parser.add_argument("--left-port", type=int, default=4, help="Left camera device id.")
    parser.add_argument("--right-port", type=int, default=6, help="Right camera device id.")
    parser.add_argument("--width", type=int, default=None, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height.")
    parser.add_argument(
        "--mode",
        choices=["pinhole", "uncalibrated", "calibrated"],
        default="pinhole",
        help="Rectification mode ('calibrated' kept as alias for 'pinhole').",
    )
    parser.add_argument(
        "--matcher",
        choices=["sgbm", "bm"],
        default="sgbm",
        help="Stereo matcher used after rectification.",
    )
    parser.add_argument("--fov-deg", type=float, default=70.0, help="Assumed horizontal FOV in pinhole mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matcher = create_matcher(args.matcher)
    mode = "pinhole" if args.mode == "calibrated" else args.mode

    left_cap = open_camera(args.left_port, args.width, args.height)
    right_cap = open_camera(args.right_port, args.width, args.height)

    try:
        for _ in range(20):
            left_cap.read()
            right_cap.read()

        left_frame, right_frame = read_pair(left_cap, right_cap)
        h, w = left_frame.shape[:2]
        k_left, dist_left = estimate_pinhole_intrinsics(w, h, args.fov_deg)
        k_right, dist_right = k_left.copy(), dist_left.copy()

        if mode == "pinhole":
            map1x, map1y, map2x, map2y, Q, match_vis = calibrate_calibrated(
                left_frame, right_frame, k_left, dist_left, k_right, dist_right
            )
            print("Pinhole rectification ready. Press 'r' to recalibrate, 'q' to quit.")
        else:
            H1, H2, match_vis = calibrate_uncalibrated(left_frame, right_frame)
            print("Uncalibrated rectification ready. Press 'r' to recalibrate, 'q' to quit.")

        while True:
            left_frame, right_frame = read_pair(left_cap, right_cap)

            if mode == "pinhole":
                rect_left = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
                rect_right = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)
            else:
                h, w = left_frame.shape[:2]
                rect_left = cv2.warpPerspective(left_frame, H1, (w, h))
                rect_right = cv2.warpPerspective(right_frame, H2, (w, h))

            disparity = compute_dense_disparity(matcher, rect_left, rect_right)
            if mode == "pinhole":
                depth_vis = depth_visualization_calibrated(disparity, Q)
            else:
                depth_vis = depth_visualization_relative(disparity)

            cv2.imshow(f"Camera {args.left_port}", left_frame)
            cv2.imshow(f"Camera {args.right_port}", right_frame)
            cv2.imshow("Dense Depth Map", depth_vis)
            cv2.imshow("Calibration Matches", match_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("r"):
                print(f"Re-running {mode} RANSAC calibration...")
                if mode == "pinhole":
                    map1x, map1y, map2x, map2y, Q, match_vis = calibrate_calibrated(
                        left_frame, right_frame, k_left, dist_left, k_right, dist_right
                    )
                else:
                    H1, H2, match_vis = calibrate_uncalibrated(left_frame, right_frame)

    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
