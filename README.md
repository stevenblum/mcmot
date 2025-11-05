TODO
- Have Global Tracks Include Color and Shape
- Overhead Include Color/Shape
- Overhead include Aruco Markers
- Draw global contacts with purple
- Utils, have model finder diplay a list of models
- Utils, function to get Aruco positions
- Utils, function to find cameras
- Use one Yolo model for all cameras

Organization
Multi-Camera Multi-Object Tracker Library
    1) ModelPlus
        a) Model-Detection
        c) Tracker
        e) Annotators
    2) Cameras
       a) Camera_Number
       b) Camera_Device_ID
       c) Intrinsics (MTX, DIST)
       d) Extrinsics (R, T, ProjMatrix)
       b) ModelPlus
    3) MCMOTracker
       a) Cameras (dict)
       b) Global Tracks (dict)
       c) Cost Matrix
       d) Assignment Solver

Heavily uses the ultralytics Supervision library because of its tracking and annotations functions
Some important items that are not captured in the online documentation:

Format of sv.Detections, all properties are stored in lists, length of n detections:
    xyxy: [[x1,y1,x2,y2],...]  # list of bounding boxes
    confidence: [conf1, conf2,...]  # list of confidences
    class_id: [cls1, cls2,...]  # list of class IDs
    tracker_id: [tid1, tid2,...]  # list of tracker IDs (after tracking)
    data: a dictonary of any other user data, not currently used