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

## FastAPI + Vue control plane

Backend API (FastAPI) is available at `mcmot.api.app`.

Run backend:

```bash
python3 run_mcmot_api.py
```

or

```bash
PYTHONPATH=src uvicorn mcmot.api.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /api/health`
- `GET /api/session/status`
- `POST /api/session/start`
- `POST /api/session/stop`
- `POST /api/session/recalibrate`
- `GET /api/frame/{stream}.jpg`
- `GET /api/stream/{stream}.mjpeg`
- `WS /ws/events`

Vue frontend is in `webui/`.

Run frontend:

```bash
cd webui
npm install
npm run dev
```

By default Vite proxies `/api` and `/ws` to `http://localhost:8000`.
