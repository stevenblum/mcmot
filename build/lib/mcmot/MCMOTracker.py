from .Camera import Camera
from .ModelPlus import ModelPlus
from . import MCMOTUtils
import time
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from config.aruco_config import ArucoConfig

<<<<<<< HEAD:MCMOTracker.py
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

detector = cv2.aruco.ArucoDetector(aruco_dict)

class ModelPlus:
    def __init__(self, model_path, tracking_yaml_path, confidence_threshold=0.25):
        self.model = YOLO(model_path,verbose=False)
        self.tracker = sv.ByteTrack(frame_rate=25)
        #self.smoother = sv.DetectionsSmoother()
        self.confidence_threshold = confidence_threshold
        colors = sv.ColorPalette.from_hex(["#ff0303",  "#ef16ae", "#0F8807", "#4af84a", "#003CFF", "#49d2f8"])
        self.annotator_tracks = sv.BoxAnnotator(color=colors, thickness=2)
        self.annotator_detections = sv.BoxAnnotator(color=colors, thickness=1)
        self.annotator_track_tails = sv.TraceAnnotator(color=colors, thickness=2)
        self.frame = None
        self.frame_annotated = None
        self.detections = None
        self.detections_tracked = None
        #self.detections_smoothed = None
        self.frame = None
        self.frame_annotated = None

    def detect_track_and_annotate(self, frame):
        result = self.model(frame, conf=self.confidence_threshold, iou=.7, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections_tracked = self.tracker.update_with_detections(detections)
        #detections_smoothed = self.smoother.update_with_detections(detections_tracked)
        self.frame = frame
        af = self.annotator_tracks.annotate(frame.copy(), detections_tracked)
        af = self.annotator_detections.annotate(af, detections_tracked)
        af = self.annotator_track_tails.annotate(af, detections_tracked)

        self.frame = frame
        self.frame_annotated = af
        self.detections = detections
        self.detections_tracked = detections_tracked
        #self.detections_smoothed = detections_smoothed

class Camera:    
    def __init__(self, camera_number, camera_device_id, camera_name, model_path, tracker_yaml_path, aruco_positions):
        self.camera_number = camera_number
        self.camera_device_id = camera_device_id
        print(f"    Opening Camera on Port {camera_device_id}")
        self.cap = cv2.VideoCapture(camera_device_id)
        self.frame = None
        self.frame_capture_time = None
        self.camera_name = camera_name
        self.mtx = None
        self.dist = None

        self.model_plus = ModelPlus(model_path, tracker_yaml_path)
        self.set_intrinsics()

        if aruco_positions == None:
            self.aruco_positions = None
        else:
            self.rvec=None
            self.tvec=None
            self.Rt = None
            self.aruco_positions = aruco_positions
            self.aruco_config = ArucoConfig("square4")
            self.set_extrinsics()

    def set_intrinsics(self):
        calibration_data = np.load(f"./config/camera_calibration/{self.camera_name}/{self.camera_name}.npz")
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
            if id in self.aruco_config.marker_positions:
                if id not in aruco_positions:
                    aruco_positions[id] = []
                aruco_positions[id].append(corner)
        for id in aruco_positions:
            aruco_positions[id] = np.mean(aruco_positions[id], axis=0)

        # Prepare image and object points for cv2.solvePnP
        object_points = []
        image_points = []
        for id, corner in aruco_positions.items():
            if id in self.aruco_config.marker_positions:
                object_points.append(self.aruco_config.marker_positions[id])
                image_points.append(np.mean(corner[0], axis=0))  # Average corner position
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        print(object_points)
        print(image_points)

        # Solve for rotation and translation vectors with cv2.solvePnP
        if len(object_points) >= 4:  # Need at least 4 points to solve PnP
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.mtx, self.dist)
            self.rvec = rvec
            self.r = cv2.Rodrigues(rvec)[0]
            self.tvec = tvec
            self.t = tvec
            print(f"Camera {self.camera_name} extrinsics set.")
        else:
            print(f"Error: Not enough ArUco markers detected for camera {self.camera_name}. Need at least 4.")

        # Construct the extrinsic matrix [R|t]
        # Rodrigues converts rvec to R matrix
        Rt = np.hstack((cv2.Rodrigues(self.rvec)[0], self.tvec))

        self.Rt = Rt

        # Calculate the projection matrix
        self.projMatrix = np.dot(self.mtx, Rt)


    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
        else:
            self.frame = None
        self.frame_capture_time = time.time()

    def detect_track_and_annotate(self):
        self.model_plus.detect_track_and_annotate(self.frame) 

        if self.aruco_positions is not None:
            self.model_plus.frame_annotated = self.draw_axis(self.model_plus.frame_annotated)


    def draw_axis(self, frame):
        axis_length = 10  # Length of the axis lines in meters
        axis = np.float32([[0,0,0],[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, self.rvec, self.tvec, self.mtx, self.dist)
        corner = tuple(int(x) for x in imgpts[0].ravel())
        frame = cv2.line(frame, corner, tuple(int(x) for x in imgpts[1].ravel()), (255,0,0), 3)  # X-axis in blue
        frame = cv2.line(frame, corner, tuple(int(x) for x in imgpts[2].ravel()), (0,255,0), 3)  # Y-axis in green
        frame = cv2.line(frame, corner, tuple(int(x) for x in imgpts[3].ravel()), (0,0,255), 3)  # Z-axis in red
        return frame

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

=======
>>>>>>> 0c44626d (Reorg, Split Frame/Detect/Annotate, Overhead Plot):build/lib/mcmot/MCMOTracker.py
class MCMOTracker:
    def __init__(self, model_path,camera_device_ids,camera_names,confidence_threshold,tracker_yaml_path, aruco_positions):
        self.model_path = model_path
        self.tracker_yaml_path = tracker_yaml_path
        self.camera_device_ids = camera_device_ids
        self.camera_names = camera_names
        self.aruco_positions = aruco_positions
        self.confidence_threshold = confidence_threshold
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
            self.cameras[camera_number] = Camera(camera_number, camera_device_id, camera_name, self.model_path, self.confidence_threshold, self.tracker_yaml_path, aruco_positions=True)
            print("    Completed Initialization")
<<<<<<< HEAD:MCMOTracker.py
        
    def update_camera_tracks(self):
        self.cameras[0].capture_frame()
        self.cameras[1].capture_frame()
        self.cameras[0].detect_track_and_annotate()
        self.cameras[1].detect_track_and_annotate()
=======
    
    def capture_cameras_frames(self):
        for camera in self.cameras.values():
            camera.capture_frame()

    def update_cameras_tracks(self):
        for camera in self.cameras.values():
            camera.detect_and_track()
>>>>>>> 0c44626d (Reorg, Split Frame/Detect/Annotate, Overhead Plot):build/lib/mcmot/MCMOTracker.py

    def match_global_tracks(self):
        cam0_tracks = self.cameras[0].model_plus.detections_tracked.tracker_id.tolist()
        cam1_tracks = self.cameras[1].model_plus.detections_tracked.tracker_id.tolist()

        if len(cam0_tracks) == 0 and len(cam1_tracks) ==     0:
            self.global_tracks = {}
            return
        elif len(cam0_tracks) == 0:
            self.global_tracks = {(None, tid1): None for tid1 in cam1_tracks}
            return
        elif len(cam1_tracks) == 0:
            self.global_tracks = {(tid0, None): None for tid0 in cam0_tracks}
            return

        # Build Cost Matrix with Reprojection Error
        #################################################
        xyxy = self.cameras[0].model_plus.detections_tracked.xyxy
        centers0 = [( (box[0]+box[2])/2, (box[1]+box[3])/2 ) for box in xyxy]
        xyxy = self.cameras[1].model_plus.detections_tracked.xyxy
        centers1 = [( (box[0]+box[2])/2, (box[1]+box[3])/2 ) for box in xyxy]

        
        costs = np.ones((len(cam0_tracks), len(cam1_tracks))) * 1e6
        points_3d = np.ones((len(cam0_tracks), len(cam1_tracks), 3))

        for i, tid0 in enumerate(cam0_tracks):
            for j, tid1 in enumerate(cam1_tracks):
                # Convert center points to numpy arrays
                center0 = np.array(centers0[i], dtype=np.float32)
                center1 = np.array(centers1[j], dtype=np.float32)
                
                point_3d = cv2.triangulatePoints(self.cameras[0].projMatrix,
                                        self.cameras[1].projMatrix,
                                        center0,
                                        center1)
                point_3d = point_3d / point_3d[3]
                point3d = point_3d[:3].reshape(-1, 1)

                points_3d[i, j] = point3d.flatten()

                reproject_c0 = cv2.projectPoints(point3d, self.cameras[0].r, self.cameras[0].t, self.cameras[0].mtx, self.cameras[0].dist)[0].flatten()
                reproject_c1 = cv2.projectPoints(point3d, self.cameras[1].r, self.cameras[1].t, self.cameras[1].mtx, self.cameras[1].dist)[0].flatten()
                costs[i,j] = np.linalg.norm(reproject_c0 - center0) + np.linalg.norm(reproject_c1 - center1)

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
        unmatched_0 = cam0_tracks.copy()
        unmatched_1 = cam1_tracks.copy()
        print("cam0 tracks:", cam0_tracks)
        print("cam1 tracks:", cam1_tracks)
        while cost < 35:
            print(costs)
            min_row, min_col = np.unravel_index(np.argmin(costs), costs.shape)
            print("Min Row, Min Col:", min_row, min_col)
            print(f"Selected min cost: {costs[min_row, min_col]}")
            #print(f"    for tracks {cam0_tracks[min_row]} and {cam1_tracks[min_col]}")
            cost = costs[min_row, min_col]
            if cost < 35:
                print(cam0_tracks)
                print(cam1_tracks)
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
        self.global_tracks = assignments

    def annotated_frame_from_camera(self,camera_number):
        af = self.cameras[camera_number].annotate_frame() # This triggers the model_plus and camera to annotate the frame

        # Draw Global Track Positions
        for match, point_3d in self.global_tracks.items():
            print(match, point_3d)
            if match[0] is not None and match[1] is not None:
<<<<<<< HEAD:MCMOTracker.py
                cam0_point = cv2.projectPoints(point_3d, self.cameras[camera_number].r, self.cameras[camera_number].t, self.cameras[camera_number].mtx, self.cameras[camera_number].dist)[0].flatten()
                cv2.circle(annotated_frame, (int(cam0_point[0]), int(cam0_point[1])), 5, (0,0,255), -1)
=======
                point_3d_reshaped = point_3d.reshape(-1, 1)
                cam0_point = cv2.projectPoints(point_3d_reshaped, self.cameras[camera_number].r, self.cameras[camera_number].t, self.cameras[camera_number].mtx, self.cameras[camera_number].dist)[0].flatten()
                cv2.circle(af, (int(cam0_point[0]), int(cam0_point[1])), 5, (0,0,255), -1)
>>>>>>> 0c44626d (Reorg, Split Frame/Detect/Annotate, Overhead Plot):build/lib/mcmot/MCMOTracker.py
                p3d = point_3d.flatten()
                cv2.putText(af, f"{p3d[0]:.1f}, {p3d[1]:.1f}, {p3d[2]:.1f}", (int(cam0_point[0]), int(cam0_point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        return af



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