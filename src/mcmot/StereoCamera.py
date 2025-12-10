from .Camera import Camera
import cv2
import numpy as np
import time


class StereoCamera:
    """
    Stereo camera system with two internal simple cameras:
    - primary
    - secondary

    Responsibilities:
    - Load intrinsics for each camera
    - Calibrate stereo extrinsics from scene matches
    - Rectify and compute a disparity / depth map
    """

    def __init__(
        self,
        primary_device_id,
        secondary_device_id,
        primary_name,
        secondary_name,
        display=True,
    ):
        self.primary = Camera(0, primary_device_id, primary_name)
        self.secondary = Camera(1, secondary_device_id, secondary_name)

        self.display = display

        # Stereo calibration results
        self.calibrated = False
        self.R = None
        self.t = None
        self.P1 = None
        self.P2 = None

        # Rectification / mapping
        self.R1 = None
        self.R2 = None
        self.P1_rect = None
        self.P2_rect = None
        self.Q = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None

        # Stereo matcher (tune these for your setup)
        block_size = 5
        num_disparities = 16 * 10  # must be divisible by 16
        self.stereo_matcher = cv2.StereoSGBM_create(
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

        self.capture_frames()
        self.display_frames()
        self.calibrate_from_scene()

    # ---------- Intrinsics API ----------

    def load_primary_intrinsics(self, npz_path):
        self.primary.load_intrinsics_from_npz(npz_path)

    def load_secondary_intrinsics(self, npz_path):
        self.secondary.load_intrinsics_from_npz(npz_path)

    # ---------- Capture ----------

    def capture_frames(self):
        """Capture a new frame from both cameras."""
        self.primary.capture_frame()
        self.secondary.capture_frame()

    def display_frames(self):
        self.primary.display_frame()
        self.secondary.display_frame()

    # ---------- Calibration from scene points ----------

    def calibrate_from_scene(
        self,
        max_matches=2000,
        ransac_threshold=1.0,
        display_matches=False,
    ):
        """
        Estimate stereo extrinsics (R, t) from scene feature matches.

        Assumes:
        - Both cameras have intrinsics (mtx, dist) loaded.
        - The scene is static and viewed by both cameras with overlap.

        Returns a metrics dict with:
            num_matches, num_inliers, num_pose_inliers,
            inlier_ratio, mean_reprojection_error_px
        """
        if self.primary.mtx is None or self.secondary.mtx is None:
            raise ValueError(
                "Both primary and secondary intrinsics must be loaded before calibration."
            )

        # Grab fresh frames
        self.capture_frames()
        img1 = self.primary.frame
        img2 = self.secondary.frame

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # --- Step 1: detect & describe features ---
        orb = cv2.ORB_create(nfeatures=max_matches)
        kps1, desc1 = orb.detectAndCompute(gray1, None)
        kps2, desc2 = orb.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            raise RuntimeError("Failed to compute descriptors in one or both images.")

        # --- Step 2: match descriptors ---
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda m: m.distance)
        if len(matches) < 8:
            raise RuntimeError(f"Not enough matches for calibration: {len(matches)}")

        if max_matches and len(matches) > max_matches:
            matches = matches[:max_matches]

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
        num_matches = len(pts1)

        # --- Step 3: estimate Essential matrix with RANSAC ---
        # For now, we assume intrinsics similar and use primary K for both.
        K1 = self.primary.mtx
        K2 = self.secondary.mtx

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K1,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_threshold,
        )
        if E is None:
            raise RuntimeError("cv2.findEssentialMat failed.")

        inlier_mask = mask.ravel() == 1
        inliers1 = pts1[inlier_mask]
        inliers2 = pts2[inlier_mask]
        num_inliers = len(inliers1)
        if num_inliers < 8:
            raise RuntimeError(f"Too few inliers after RANSAC: {num_inliers}")

        # --- Step 4: recover R, t using cheirality ---
        _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K1)
        pose_inliers = mask_pose.ravel() == 1
        inliers1 = inliers1[pose_inliers]
        inliers2 = inliers2[pose_inliers]
        num_pose_inliers = len(inliers1)

        self.R = R
        self.t = t

        # --- Step 5: build projection matrices (for reprojection error) ---
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2 @ np.hstack([R, t])
        self.P1 = P1
        self.P2 = P2

        # Triangulate and compute reprojection error
        pts4D_h = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
        pts4D_h /= pts4D_h[3]  # normalize
        proj1 = (P1 @ pts4D_h).T
        proj2 = (P2 @ pts4D_h).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        proj2 = proj2[:, :2] / proj2[:, 2:3]

        err1 = np.linalg.norm(proj1 - inliers1, axis=1)
        err2 = np.linalg.norm(proj2 - inliers2, axis=1)
        mean_reproj_error = float(np.mean(np.concatenate([err1, err2])))

        # --- Step 6: stereo rectification for depth computation ---
        h, w = gray1.shape
        R1, R2, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=K1,
            distCoeffs1=self.primary.dist,
            cameraMatrix2=K2,
            distCoeffs2=self.secondary.dist,
            imageSize=(w, h),
            R=R,
            T=t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self.R1, self.R2 = R1, R2
        self.P1_rect, self.P2_rect, self.Q = P1_rect, P2_rect, Q

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K1, self.primary.dist, R1, P1_rect, (w, h), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            K2, self.secondary.dist, R2, P2_rect, (w, h), cv2.CV_32FC1
        )

        self.calibrated = True

        # optional visualization of inlier matches
        if display_matches:
            inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i]]
            match_img = cv2.drawMatches(
                img1, kps1,
                img2, kps2,
                inlier_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow("Stereo Calibration Inlier Matches", match_img)
            cv2.waitKey(1)

        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
        print("[StereoCalibration] Total matches:", num_matches)
        print("[StereoCalibration] Inliers after RANSAC:", num_inliers)
        print("[StereoCalibration] Inliers after cheirality:", num_pose_inliers)
        print(f"[StereoCalibration] Inlier ratio: {inlier_ratio:.3f}")
        print(f"[StereoCalibration] Mean reprojection error: {mean_reproj_error:.2f} px")

        return {
            "num_matches": int(num_matches),
            "num_inliers": int(num_inliers),
            "num_pose_inliers": int(num_pose_inliers),
            "inlier_ratio": float(inlier_ratio),
            "mean_reprojection_error_px": mean_reproj_error,
        }

    # ---------- Depth / disparity ----------

    def compute_depth_map(
        self,
        normalize_for_display=True,
        return_depth_m=False,
    ):
        """
        Compute disparity (and optional depth) from the current stereo pair.

        Assumes:
        - calibrate_from_scene() has been called successfully.
        - Intrinsics and rectification maps are set.

        Returns:
            disparity_raw: float32 disparity (same size as input images)
            disp_vis: uint8 image for visualization (or None)
            depth_m: float32 depth in meters (or None) if return_depth_m and baseline known
        """
        if not self.calibrated:
            raise RuntimeError("StereoCamera is not calibrated. Call calibrate_from_scene() first.")

        # Fresh frames
        self.capture_frames()
        img1 = self.primary.frame
        img2 = self.secondary.frame

        # Rectify
        rect1 = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo_matcher.compute(gray1, gray2).astype(np.float32) / 16.0

        disp_vis = None
        if normalize_for_display:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = disp_vis.astype(np.uint8)

        depth_m = None
        if return_depth_m and self.baseline_m is not None:
            # fx from rectified projection matrix (P1_rect)
            fx = self.P1_rect[0, 0] if self.P1_rect is not None else self.primary.mtx[0, 0]
            # Avoid division by zero / invalid disparity
            eps = 1e-6
            valid = disparity > eps
            depth_m = np.zeros_like(disparity, dtype=np.float32)
            depth_m[valid] = fx * self.baseline_m / (disparity[valid])

        return disparity, disp_vis, depth_m

    def display_depth_map(self, window_name="Stereo Depth"):
        """
        Compute and display a color-mapped disparity image.
        """
        _, disp_vis, _ = self.compute_depth_map(normalize_for_display=True, return_depth_m=False)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imshow(window_name, disp_color)
        cv2.waitKey(1)
