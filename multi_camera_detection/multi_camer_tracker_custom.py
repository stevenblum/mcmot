'''Organization
MultiCameraTracker
   1) Cameras
       a) CameraID
       b) Yolo Model
       c) Intrinsics/Extrinsics
       d) Camera Tracks (object)
           - TrackID
           - Bounding Box
           - Center
    2) Global Tracks (object)
        a) Global Track ID
        b) Camera Tracks (dict, CameraID:TrackID, None for no match)   

For Greedy Heuristic:
'''

import time
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from ortools.sat.python import cp_model
import sys
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
calibration_dict = {9:(0,0,0), 14:(0,12,0), 19:(12,0,0), 25:(12,12,0)}

class Camera:    
    def __init__(self, camera_number, camera_device_id, camera_name, model_path, tracking_yaml):
        self.camera_number = camera_number
        self.camera_device_id = camera_device_id
        print(f"    Opening Camera on Port {camera_device_id}")
        self.cap = cv2.VideoCapture(camera_device_id)
        self.camera_name = camera_name
        self.model_path = model_path
        self.model = None
        self.tracker = None
        self.smoother = None
        self.mtx = None
        self.dist = None
        self.r = None
        self.t = None
        self.projMatrix = None
        self.frame = None
        self.detections = None
        self.detections_smoothed = None

        self.initialize_model()
        self.set_intrinsics()
        self.set_extrinsics()

    def set_intrinsics(self):
        calibration_data = np.load(f"./camera_calibration/{self.camera_name}/{self.camera_name}.npz")
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

    def set_extrinsics(self):
        # Show Camera To Make Sure ArUco Markers are Visible
        while True:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    c = corner[0]
                    center = c.mean(axis=0).astype(int)
                    cv2.putText(frame, f"ID:{ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    pt1 = tuple(c[0].astype(int))
                    pt2 = tuple(c[1].astype(int))
                    cv2.arrowedLine(frame, pt1, pt2, (255,0,0), 2, tipLength=0.3)
            cv2.imshow(f"Camera {self.camera_name} - Place ArUco Markers in View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


        aruco_detections = []
        # Capture the corner of the ArUco markers in 10 frames
        for t in range(10):
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(gray)
                if ids is not None:
                    for id, corner in zip(ids.flatten(), corners):
                        aruco_detections.append((id,corner))
            time.sleep(0.1)

        # Find the average position of each detected ArUco marker
        aruco_positions = {}
        for id, corner in aruco_detections:
            if id in calibration_dict:
                if id not in aruco_positions:
                    aruco_positions[id] = []
                aruco_positions[id].append(corner)
        for id in aruco_positions:
            aruco_positions[id] = np.mean(aruco_positions[id], axis=0)

        # Prepare image and object points for cv2.solvePnP
        object_points = []
        image_points = []
        for id, corner in aruco_positions.items():
            if id in calibration_dict:
                object_points.append(calibration_dict[id])
                image_points.append(np.mean(corner[0], axis=0))  # Average corner position
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        print(object_points)
        print(image_points)

        # Solve for rotation and translation vectors with cv2.solvePnP
        if len(object_points) >= 4:  # Need at least 4 points to solve PnP
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.mtx, self.dist)
            self.r, _ = cv2.Rodrigues(rvec)
            self.t = tvec
            print(f"Camera {self.camera_name} extrinsics set.")
        else:
            print(f"Error: Not enough ArUco markers detected for camera {self.camera_name}. Need at least 4.")


        print(self.r)
        print(self.t)

        # Construct the extrinsic matrix [R|t]
        Rt = np.hstack((self.r, self.t))

        # Calculate the projection matrix
        self.projMatrix = np.dot(self.mtx, Rt)

    def initialize_model(self):
        self.model = YOLO(self.model_path,verbose=False)
        self.tracker = sv.ByteTrack(frame_rate=25)
        self.smoother = sv.DetectionsSmoother()
        colors = sv.ColorPalette.from_hex(["#ff0303",  "#ef16ae", "#0F8807", "#4af84a", "#003CFF", "#49d2f8"])
        self.annotator = sv.BoxAnnotator(color=colors, thickness=2)
        self.annotator_thin = sv.BoxAnnotator(color=colors, thickness=1)

    def detect_and_track(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        result = self.model(frame, conf=0.05, iou=.7, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections_tracker = self.tracker.update_with_detections(detections)
        detections_smoothed = self.smoother.update_with_detections(detections_tracker)
        
        self.tracks = {}
        if detections_smoothed.tracker_id is not None:
            for i in range(len(detections_smoothed)):
                bbox = detections_smoothed.xyxy[i].tolist()
                conf = detections_smoothed.confidence[i] if detections_smoothed.confidence is not None else 1.0
                cls = detections_smoothed.class_id[i] if detections_smoothed.class_id is not None else 0
                tid = detections_smoothed.tracker_id[i]
                self.tracks[tid] = CameraTrack(self.camera_number, tid, bbox, conf, cls)

        self.frame = frame
        self.detections = detections
        self.detections_smoothed = detections_smoothed

class CameraTrack:
    def __init__(self, camera_number, track_id, bbox, score, class_id):
        self.camera_number = camera_number
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        self.score = score
        self.class_id = class_id

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score

class MultiCameraTrack:
    def __init__(self, iou_threshold=0.3):
        self.last_camera_match = {}

    def set_match(self,camera_tracks):
        match = {}
        for ct in camera_tracks:
            match[ct.camera_id] = ct.track_id

        self.last_camera_match = match

class MultiCameraTracker:
    def __init__(self, model_path,camera_device_ids,camera_names):
        self.model_path = model_path
        self.camera_device_ids = camera_device_ids
        self.camera_names = camera_names    
        self.cameras = {}  # {camera_id: Camera}
        self.global_tracks = {}
        self.camera_tracks = {}  # {ctid: {track_id: CameraTrack}}
        self.next_ctid = 0
        self.next_gtid = 0
        self.next_track_id = 0

    def select_cameras(self):
        # Show image from all available cameras
        # todo
        #selected_cameras = util.select_cameras(2)


        for camera_number, camera_device_id in enumerate(self.camera_device_ids):
            print(f"Initializing Camera {camera_number} - Port {camera_device_id} - {self.camera_names[camera_number]}")
            camera_name = self.camera_names[camera_number]
            self.cameras[camera_number] = Camera(camera_number, camera_device_id, camera_name, self.model_path, "botsort.yaml")
            self.cameras[camera_number].set_intrinsics()
            self.cameras[camera_number].set_extrinsics()
            print("    Completed Initialization")
        
    def update_camera_tracks(self):
        self.cameras[0].detect_and_track()
        self.cameras[1].detect_and_track()

    def match_global_tracks(self):
        cam0_tracks = [tid for tid in self.cameras[0].tracks.keys()]
        cam1_tracks = [tid for tid in self.cameras[1].tracks.keys()]

        # Build Cost Matrix with Reprojection Error
        #################################################
        costs = np.ones((len(cam0_tracks), len(cam1_tracks))) * 1e6
        points_3d = np.ones((len(cam0_tracks), len(cam1_tracks), 3))

        for i, tid0 in enumerate(cam0_tracks):
            for j, tid1 in enumerate(cam1_tracks):
                # Convert center points to numpy arrays
                center0 = np.array(self.cameras[0].tracks[tid0].center, dtype=np.float32).reshape(2, 1)
                center1 = np.array(self.cameras[1].tracks[tid1].center, dtype=np.float32).reshape(2, 1)
                
                point_3d = cv2.triangulatePoints(self.cameras[0].projMatrix,
                                        self.cameras[1].projMatrix,
                                        center0,
                                        center1)
                point_3d = point_3d / point_3d[3]
                point3d = point_3d[:3].reshape(-1, 1)

                points_3d[i, j] = point3d.flatten()

                reproject_c0 = cv2.projectPoints(point3d, self.cameras[0].r, self.cameras[0].t, self.cameras[0].mtx, self.cameras[0].dist)[0].flatten()
                reproject_c1 = cv2.projectPoints(point3d, self.cameras[1].r, self.cameras[1].t, self.cameras[1].mtx, self.cameras[1].dist)[0].flatten()
                costs[i,j] = np.linalg.norm(reproject_c0 - self.cameras[0].tracks[tid0].center) + np.linalg.norm(reproject_c1 - self.cameras[1].tracks[tid1].center)

        # Add Rows and columns to the cost matrix for tracks without matches
        #for i in range(unmatched_0):
        #    costs= np.append(costs, np.ones((costs.shape[0], 1)) * (3 + i / 10), axis=1)
        #for i in range(unmatched_1):
        #    costs= np.append(costs, np.ones((1, costs.shape[1])) * (3 + i / 10), axis=0)
        #costs[len(unmatched_0):, len(unmatched_1):] = 1e6

        # Use Greedy Heuristic to Solve Assignment Problem
        ###########################################################

        

        cost = 0
        assignments = {}
        unmatched_0 = cam0_tracks
        unmatched_1 = cam1_tracks
        while cost < 15:
            print(costs)
            min_row, min_col = np.unravel_index(np.argmin(costs), costs.shape)
            print("Min Row, Min Col:", min_row, min_col)
            print(f"Selected min cost: {costs[min_row, min_col]} for tracks {cam0_tracks[min_row]} and {cam1_tracks[min_col]}")
            cost = costs[min_row, min_col]
            if cost < 15:
                tid0 = cam0_tracks[min_row]
                tid1 = cam1_tracks[min_col]
                assignments[(tid0, tid1)] = points_3d[min_row, min_col]
                unmatched_0.remove(tid0)
                unmatched_1.remove(tid1)
                costs[min_row, :] = 1e6
                costs[:, min_col] = 1e6

        for tid in unmatched_0:
            assignments[(tid, None)] = None
        for tid in unmatched_1:
            assignments[(None, tid)] = None

        
        print(assignments)
        self.global_matches = assignments

    def frame_from_cam0(self):
        frame = self.cameras[0].frame.copy()

        annotated_frame = self.cameras[0].annotator.annotate(frame.copy(), self.cameras[0].detections_smoothed)
        annotated_frame = self.cameras[0].annotator_thin.annotate(annotated_frame, self.cameras[0].detections)

        for match, point_3d in self.global_tracks.items():
            if match[0] is not None and match[1] is not None:
                point_3d_reshaped = point_3d.reshape(-1, 1)
                cam0_point = cv2.projectPoints(point_3d_reshaped, self.cameras[0].r, self.cameras[0].t, self.cameras[0].mtx, self.cameras[0].dist)[0].flatten()
                cv2.circle(annotated_frame, (int(cam0_point[0]), int(cam0_point[1])), 5, (0,255,255), -1)
                cv2.putText(annotated_frame, f"{point_3d}", (int(cam0_point[0]), int(cam0_point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return annotated_frame



'''
        # Rematch old matches first
        #####################################################################3
        for mct in self.tracks:

            if len(mct.last_camera_match) < 2:
                continue


            elif len(mct.last_camera_match) == 2:
                prev_location = self.cameras[cids[0]].tracks[tid0].center
                tid0 = mct.last_camera_match[cids[0]]
                tid1 = mct.last_camera_match[cids[1]]

                points_3d = cv2.triangulatePoints(self.cameras[cids[0]].projMatrix,
                                        self.cameras[cids[1]].projMatrix,
                                        self.cameras[cids[0]].tracks[tid0].center,
                                        self.cameras[cids[1]].tracks[tid1].center)
                # If new point is withing .2 of prev, match
                if np.linalg.norm(points_3d - prev_location) < 0.2:
                    mct.set_match([self.cameras[cids[0]].tracks[tid0], self.cameras[cids[1]].tracks[tid1]])
                    self.cameras[cids[0]].tracks[tid0].global_id = mct.global_id
                    self.cameras[cids[1]].tracks[tid1].global_id = mct.global_id


                    unmatched_0.remove(tid0)
                    unmatched_1.remove(tid1)
                    continue
                    


    # Solve the assignment problem

        # Solve Assignment with Google OR Tools SAT Solver
        ###########################

        # Create the model
        model = cp_model.CpModel()

        # Create assignment variables, x[w, t] is 1 if worker w is assigned to task t.
        x = {}
        for w in range(costs.shape[0]):
            for t in range(costs.shape[1]):
                x[w, t] = model.NewBoolVar(f'x_{w}_{t}')

        # Add constraints: Each worker is assigned at most one task.
        for w in range(len(unmatched_0)):
            model.Add(sum(x[w, t] for t in range(costs.shape[1])) <= 1)

        # Add constraints: Each task is assigned to exactly one worker.
        for t in range(len(unmatched_1)):
            model.Add(sum(x[w, t] for w in range(len(unmatched_0))) == 1)

        # Create the objective function.
        objective_terms = []
        for w in range(len(unmatched_0)):
            for t in range(len(unmatched_1)):
                objective_terms.append(costs[w][t] * x[w, t])
        model.Minimize(sum(objective_terms))

        # Create the solver and solve the model.
        solver = cp_model.CpSolver()

        # --- Provide the initial guess (warm start) ---
        single_ids_0 = 0
        single_ids_1 = 0
        initial_assignments = {}
        for gt in self.global_tracks:
            # Check if tracks are in recent update
            if gt.last_camera_match[0] not in unmatched_0:
                gt.last_camera_match[0] = None
            if gt.last_camera_match[1] not in unmatched_1:
                gt.last_camera_match[1] = None

            if gt.last_camera_match[0] is None and gt.last_camera_match[1] is None:
                print("Error: Global track with no camera matches")

            elif gt.last_camera_match[0] is not None and gt.last_camera_match[1] is not None:
                tid0 = gt.last_camera_match[0]
                tid1 = gt.last_camera_match[1]
                row_in_costs = unmatched_0.index(tid0)
                col_in_costs = unmatched_1.index(tid1)
                initial_assignments[(row_in_costs, col_in_costs)] = 1

            elif gt.last_camera_match[0] is not None and gt.last_camera_match[1] is None:
                cid = gt.last_camera_match.keys()[0]
                tid = gt.last_camera_match[cid]
                row_in_costs = unmatched_0.index(tid)
                col_initial_assignment = len(unmatched_1) + single_ids_0
                initial_assignments[(row_in_costs, col_initial_assignment)] = 1
                single_ids_0 += 1
            elif gt.last_camera_match[0] is None and gt.last_camera_match[1] is not None:
                row_initial_assignment = len(unmatched_0) + single_ids_1
                col_initial_assignment = unmatched_1.index(tid)
                initial_assignments[(row_initial_assignment, col_initial_assignment)] = 1
                single_ids_1 += 1
            else:
                print("Global Track Could Not be Resolved To an Inital Guess for Assignement Problem")

        # Apply the initial guess to the solver.
        for (w, t), val in initial_assignments.items():
            solver.AddHint(x[w, t], val)

        # Solve the problem.
        status = solver.Solve(model)

        # Print the solution.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Total cost: {solver.ObjectiveValue()}")
            for w in range(num_workers):
                for t in range(num_tasks):
                    if solver.BooleanValue(x[w, t]):
                        print(f"Worker {w} assigned to task {t}. Cost: {costs[w][t]}")
        else:
            print("No solution found.")



'''