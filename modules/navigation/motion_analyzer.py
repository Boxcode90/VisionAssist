import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self, feature_params=dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7),
                 lk_params=dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))):
        """
        Initializes the optical flow tracker.
        feature_params: parameters for Shi-Tomasi corner detection.
        lk_params: parameters for Lucas-Kanade optical flow.
        """
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.prev_gray = None
        self.prev_pts = None
        self.flow_vectors = {}  # track_id -> (dx, dy) in normalized units

    def process(self, frame, detections):
        """
        Computes motion for each detected track.
        :param frame: current BGR frame.
        :param detections: list of tuples (x1,y1,x2,y2,conf,label,track_id)
        :return: dict mapping track_id -> motion_state (string) and motion vector (dx, dy)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        motion_results = {}

        if self.prev_gray is None:
            self.prev_gray = gray
            # Initialize features from all detections
            self.prev_pts = []
            for det in detections:
                x1, y1, x2, y2, _, _, track_id = det
                if track_id is None:
                    continue
                # Choose feature points inside the bounding box
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if pts is not None:
                    self.prev_pts.append((track_id, pts))
            return {}

        # 1. Compute optical flow from prev_gray to gray
        if self.prev_pts is None:
            self.prev_gray = gray
            return {}

        # Build list of all points and their corresponding track_ids
        all_prev_pts = []
        id_list = []
        for tid, pts in self.prev_pts:
            for pt in pts:
                all_prev_pts.append(pt)
                id_list.append(tid)
        
        if not all_prev_pts:
            self.prev_gray = gray
            return {}

        all_prev_pts = np.array(all_prev_pts, dtype=np.float32).reshape(-1, 1, 2)

        # Compute optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, all_prev_pts, None, **self.lk_params)

        # 2. Estimate ego-motion (homography) between prev and current frame
        # We'll use the matched points that are likely background (static)
        # For simplicity, we take all points with status==1 and compute homography
        good_prev = []
        good_next = []
        for i, st in enumerate(status):
            if st == 1:
                good_prev.append(all_prev_pts[i].reshape(2))
                good_next.append(next_pts[i].reshape(2))
        
        if len(good_prev) < 4:
            self.prev_gray = gray
            # Update prev_pts for next frame (use detections again)
            self.prev_pts = []
            for det in detections:
                x1, y1, x2, y2, _, _, track_id = det
                if track_id is None:
                    continue
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if pts is not None:
                    self.prev_pts.append((track_id, pts))
            return {}

        # Find homography to compensate ego-motion
        H, _ = cv2.findHomography(np.array(good_prev), np.array(good_next), cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3)

        # 3. For each object, compute its compensated motion
        # We group points by track_id
        points_by_id = {}
        for i, tid in enumerate(id_list):
            if status[i] == 1:
                prev_pt = all_prev_pts[i].reshape(2)
                next_pt = next_pts[i].reshape(2)
                # Compensate next_pt by inverse homography to remove camera motion
                # Homography maps prev to next; we apply inverse to next to get compensated position
                next_compensated = cv2.perspectiveTransform(np.array([[next_pt]], dtype=np.float32), np.linalg.inv(H))
                comp_pt = next_compensated[0][0]
                # Compute compensated displacement
                dx = comp_pt[0] - prev_pt[0]
                dy = comp_pt[1] - prev_pt[1]
                if tid not in points_by_id:
                    points_by_id[tid] = []
                points_by_id[tid].append((dx, dy))

        # 4. Average displacement per track and classify
        for tid, displacements in points_by_id.items():
            if not displacements:
                continue
            avg_dx = np.mean([d[0] for d in displacements])
            avg_dy = np.mean([d[1] for d in displacements])
            # Normalize by frame interval and image size (approximate)
            # We'll use a threshold to classify
            # For now, simple threshold (in pixels, considering image ~640x480)
            # Adjust thresholds later
            speed = np.sqrt(avg_dx**2 + avg_dy**2)
            # Classification (thresholds are empirical)
            if speed < 0.5:   # very small movement
                state = "static"
            else:
                # Determine direction based on angle
                angle = np.arctan2(avg_dy, avg_dx) * 180 / np.pi
                # 0° = right, 90° = down (in image coordinates, y increases downwards)
                # For "approaching" we consider motion towards center (vertical direction)
                # Usually, approaching means object's y-coordinate moves upwards (negative dy)
                # but we also need to consider scale changes; for simplicity we use dy
                # Approaching: object size increases, but we don't have scale yet.
                # For now, use dy (if negative, moving up = towards camera)
                # Actually, in typical forward motion, objects move outward from center.
                # We'll keep it simple: if dy < -1.0, consider approaching.
                # We'll refine later.
                if avg_dy < -1.5:
                    state = "approaching"
                elif avg_dy > 1.5:
                    state = "moving_away"
                elif abs(avg_dx) > abs(avg_dy):
                    state = "crossing"
                else:
                    state = "unknown"
            motion_results[tid] = {"state": state, "dx": avg_dx, "dy": avg_dy}

        # 5. Update prev_gray and prev_pts for next frame (use current detections)
        self.prev_gray = gray
        self.prev_pts = []
        for det in detections:
            x1, y1, x2, y2, _, _, track_id = det
            if track_id is None:
                continue
            mask = np.zeros_like(gray)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if pts is not None:
                self.prev_pts.append((track_id, pts))

        return motion_results