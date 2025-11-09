from .Camera import Camera
from .ModelPlus import ModelPlus
from . import MCMOTUtils
from .config import ArucoConfig
import time
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt
from io import BytesIO

class MCMOTracker:
    def __init__(self, model_path,camera_device_ids,camera_names,confidence_threshold=.1,tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=None):
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
        self.overhead_plot = None

    def select_cameras(self):
        # Show image from all available cameras
        # todo
        # selected_cameras = util.select_cameras(2)

        for camera_number, camera_device_id in enumerate(self.camera_device_ids):
            print(f"Initializing Camera {camera_number} - Port {camera_device_id} - {self.camera_names[camera_number]}")
            camera_name = self.camera_names[camera_number]
            self.cameras[camera_number] = Camera(camera_number, camera_device_id, camera_name, self.model_path, self.confidence_threshold, self.tracker_yaml_path, aruco_positions=True)
            print("    Completed Initialization")
    
    def capture_cameras_frames(self):
        for camera in self.cameras.values():
            camera.capture_frame()

    def update_cameras_tracks(self):
        for camera in self.cameras.values():
            camera.detect_and_track()

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
        class0 = self.cameras[0].model_plus.detections_tracked.class_id
        color0,shape0 = self.class2colorshape(class0)
        confidence0 = self.cameras[0].model_plus.detections_tracked.confidence
        xyxy = self.cameras[1].model_plus.detections_tracked.xyxy
        centers1 = [( (box[0]+box[2])/2, (box[1]+box[3])/2 ) for box in xyxy]
        class1 = self.cameras[1].model_plus.detections_tracked.class_id
        color1,shape1 = self.class2colorshape(class1)
        confidence1 = self.cameras[1].model_plus.detections_tracked.confidence

        
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

                if color0[i]!=color1[j] and shape0[i]==shape1[j]:
                    costs[i,j] = 1.2*costs[i,j]
                if color0[i]==color1[j] and shape0[i]!=shape1[j]:
                    costs[i,j] = 1.1*costs[i,j]
                if color0[i]!=color1[j] and shape0[i]==shape1[j]:
                    costs[i,j] = 1.4*costs[i,j]


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
        while cost < 15:
            print(costs)
            min_row, min_col = np.unravel_index(np.argmin(costs), costs.shape)
            print("Min Row, Min Col:", min_row, min_col)
            print(f"Selected min cost: {costs[min_row, min_col]}")
            #print(f"    for tracks {cam0_tracks[min_row]} and {cam1_tracks[min_col]}")
            cost = costs[min_row, min_col]
            if cost < 15:
                print(cam0_tracks)
                print(cam1_tracks)
                tid0 = cam0_tracks[min_row]
                tid1 = cam1_tracks[min_col]
                
                # Determine the Global Track Color
                if color0[min_row] == color1[min_col]:
                    colorg = color0[min_row]
                elif confidence0[min_row] > confidence1[min_col]:
                    colorg = color0[min_row]
                else:
                    colorg = color1[min_col]

                #Determine the Global Track Shape
                if shape0[min_row] == shape1[min_col]:
                    shapeg = shape0[min_row]
                elif confidence0[min_row] > confidence1[min_col]:
                    shapeg = shape0[min_row]
                else:
                    shapeg = shape1[min_col]

                assignments[(tid0, tid1)] = [points_3d[min_row, min_col],colorg,shapeg]
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

    def class2colorshape(self,class_list):
        color = []
        shape = []
        for c in class_list:
            if c in [0,2,4]:
                shape.append("square")
            else:
                shape.append("circle")

            if c in [0,1]:
                color.append("red")
            elif c in [2,3]:
                color.append("green")
            else:
                color.append("blue")
        return color,shape


    def annotated_frame_from_camera(self, camera_number):
        af = self.cameras[camera_number].annotate_frame() # This triggers the model_plus and camera to annotate the frame

        # Draw Global Track Positions
        for match,values in self.global_tracks.items():
            if values is None:
                continue
            point_3d,color,shape = values
            print(match, point_3d)
            if match[0] is not None and match[1] is not None:
                point_3d_reshaped = point_3d.reshape(-1, 1)
                cam0_point = cv2.projectPoints(point_3d_reshaped, self.cameras[camera_number].r, self.cameras[camera_number].t, self.cameras[camera_number].mtx, self.cameras[camera_number].dist)[0].flatten()
                cv2.circle(af, (int(cam0_point[0]), int(cam0_point[1])), 5, (0,0,255), -1)
                p3d = point_3d.flatten()
                cv2.putText(af, f"{p3d[0]:.1f}, {p3d[1]:.1f}, {p3d[2]:.1f}", (int(cam0_point[0]), int(cam0_point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        return af

    def plot_overhead(self, save_path=None):
            """Plot all cameras' FOVs and all global detections in an overhead view."""
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect('equal')
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_title("Overhead View of Multi-Camera Multi-Object Tracker")

            # Plot global track detections
            if self.global_tracks:
                first = True
                for match,values in self.global_tracks.items():
                    if values is None:
                        continue
                    point,color,shape = values
                    p = np.array(point).flatten()
                    if shape == "square":
                        marker_type = "s"
                    else:
                        marker_type = "o"

                    if p.size >= 2:  # make sure it has x,y
                        ax.scatter(p[0], p[1], s=70, c=color, marker=marker_type,label='Detections' if first else "")
                        first = False


            # Plot each camera's FOV
            for cam_id, cam in self.cameras.items():
                fov = cam.compute_fov_on_ground()
                C, left, right = fov["C"], fov["left"], fov["right"]

                ax.plot([C[0], left[0]], [C[1], left[1]], 'b--', linewidth=1)
                ax.plot([C[0], right[0]], [C[1], right[1]], 'b--', linewidth=1)
                ax.scatter(C[0], C[1], c='b', marker='o')
                ax.text(C[0], C[1], f"Cam {cam_id}", color='b', fontsize=8, ha='right')

            # Convert Matplotlib figure to OpenCV image
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            self.overhead_plot = img
            return self.overhead_plot 


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