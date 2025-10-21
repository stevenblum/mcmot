'''
cvat-cli --server-host http://localhost:8080 \
  --auth scblum2:Basketball1! \
  task auto-annotate 16\
  --function-file /home/scblum/Projects/testbed_cv/cvat_aa_yolo_bidirection.py\
  -p model=str:/home/scblum/Projects/testbed_cv/yolo_training/part_detection_25_10_07_14_28/weights/best.pt \
  -p confidence_threshold=float:0.4
'''


# cvat_aa_yolo_bidirection.py
# Updated for YOLO11 compatibility
# Works with: cvat-cli task auto-annotate --function-file this_file.py
import os, re, cv2, numpy as np
from PIL import Image
from ultralytics import YOLO
import cvat_sdk.auto_annotation as cvataa


class YOLOBiDirFunction:
    def __init__(self, model: str, confidence_threshold: float = 0.5,
                 tracker: str = "botsort.yaml", bidirectional: bool = True):
        print(f"[INFO] Loading YOLO model from: {model}")
        self.yolo = YOLO(model)
        print(f"[INFO] Model successfully loaded. Label classes: {self.yolo.names}")

        self.conf = confidence_threshold
        self.tracker = tracker
        self.bidirectional = bidirectional
        self._ready = False
        self._frames = []
        self._name2idx = {}
        self._by_idx = {}

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        names = self.yolo.names
        label_names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        labels = [cvataa.label_spec(name, i, type="rectangle") for i, name in enumerate(label_names)]
        return cvataa.DetectionFunctionSpec(labels=labels)

    def _collect_frames(self, any_frame_path: str):
        # Do not collect or search for frames; CVAT provides the image directly.
        # Just map the provided frame name to index 0.
        if not any_frame_path:
            raise RuntimeError("No frame path provided by CVAT context.")
        abs_path = os.path.abspath(any_frame_path)
        fname = os.path.basename(abs_path)
        self._frames = [abs_path]
        self._name2idx = {fname: 0}
        print(f"[INFO] Using only CVAT-provided frame: {fname}")

    def _precompute(self, any_frame_path: str):
        if self._ready:
            return
        self._collect_frames(any_frame_path)
        print(f"[DEBUG] Starting precompute on {len(self._frames)} frames")
        if not self._frames:
            self._ready = True
            return

        # Instead of writing a video, just run detection on the provided image
        fwd = {}
        for idx, frame_path in enumerate(self._frames):
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[WARN] Could not read image: {frame_path}")
                fwd[idx] = []
                continue
            # Use YOLO predict, not track, for single images
            results = self.yolo.predict(
                img,
                conf=self.conf,
                verbose=False
            )
            cur = []
            for r in results:
                if r.boxes is not None:
                    for b in r.boxes:
                        cls = int(b.cls[0]) if b.cls is not None else 0
                        tid = -1  # No tracking for single images
                        conf = float(b.conf[0]) if b.conf is not None else 1.0
                        xyxy = b.xyxy[0].tolist()
                        cur.append((cls, xyxy, tid, conf))
            fwd[idx] = cur
        self._by_idx = fwd
        self._ready = True

    def detect(self, context, image):
        if not self._ready:
            self._precompute(getattr(context, 'frame_name', '') or '')
        frame_name = os.path.basename(getattr(context, 'frame_name', ''))
        idx = self._name2idx.get(frame_name)
        if idx is None:
            print(f"[DEBUG] Frame not found in mapping: {frame_name}")
            print(f"[DEBUG] Known frame names: {list(self._name2idx.keys())}")
            return []
        conf_thr = context.conf_threshold if context.conf_threshold is not None else self.conf
        shapes = []
        boxes = self._by_idx.get(idx, [])
        print(f"[DEBUG] Detect called for frame {context.frame_name}, idx={idx}, num_boxes={len(boxes)}")
        for cls, xyxy, tid, conf in boxes:
            if conf < conf_thr:
                continue
            shapes.append(cvataa.rectangle(cls, xyxy, group=None))
        return shapes

def create(model: str, confidence_threshold: float = 0.5, tracker: str = "botsort.yaml", bidirectional: bool = True):
    return YOLOBiDirFunction(model, confidence_threshold, tracker, bidirectional)
def create(model: str,confidence_threshold: float = 0.5,tracker: str = "botsort.yaml",bidirectional: bool = True):
    return YOLOBiDirFunction(model, confidence_threshold, tracker, bidirectional)
                        xyxy = b.xyxy[0].tolist()
                        cur.append((cls, xyxy, tid, conf))
                bwd_tmp[idx] = cur

            merged = {}
            for idx in range(len(self._frames)):
                F = fwd.get(idx, []); B = bwd_tmp.get(idx, []); used = [False]*len(B); out=[]
                for (cls, fbox, fid, fconf) in F:
                    best_j, best_iou, best = -1, 0.0, None
                    for j, (cls2, bbox2, bid, bconf) in enumerate(B):
                        if used[j] or cls2 != cls: continue
                        s = self._iou(fbox, bbox2)
                        if s > best_iou:
                            best_iou, best_j, best = s, j, (cls2, bbox2, bid, bconf)
                    if best and best_iou >= 0.5:
                        avg = ((np.array(fbox)+np.array(best[1]))/2.0).tolist()
                        out.append((cls, avg, fid if fid!=-1 else best[2], max(fconf,best[3])))
                        used[best_j] = True
                    elif best_iou < 0.2:
                        continue
                    else:
                        out.append((cls, fbox, fid, fconf))
                for j, (cls2,bbox2,bid,bconf) in enumerate(B):
                    if not used[j]: out.append((cls2,bbox2,bid,bconf))
                merged[idx]=out
            self._by_idx=merged
        else:
            self._by_idx=fwd

        for p in (fwd_mp4,bwd_mp4):
            try:
                if os.path.exists(p): os.remove(p)
            except Exception: pass
        self._ready=True

    def detect(self, context, image):
        if not self._ready:
            self._precompute(getattr(context, 'frame_name', '') or '')
        frame_name = os.path.basename(getattr(context, 'frame_name', ''))
        idx = self._name2idx.get(frame_name)
        if idx is None:
            print(f"[DEBUG] Frame not found in mapping: {frame_name}")
            print(f"[DEBUG] Known frame names: {list(self._name2idx.keys())}")
            return []
        conf_thr=context.conf_threshold if context.conf_threshold is not None else self.conf
        shapes=[]
        boxes=self._by_idx.get(idx,[])
        print(f"[DEBUG] Detect called for frame {context.frame_name}, idx={idx}, num_boxes={len(boxes)}")
        for cls,xyxy,tid,conf in boxes:
            if conf<conf_thr: continue
            shapes.append(cvataa.rectangle(cls,xyxy,group=(None if tid<0 else tid)))
        return shapes

def create(model: str,confidence_threshold: float = 0.5,tracker: str = "botsort.yaml",bidirectional: bool = True):
    return YOLOBiDirFunction(model, confidence_threshold, tracker, bidirectional)
def create(model: str,confidence_threshold: float = 0.5,tracker: str = "botsort.yaml",bidirectional: bool = True):
    return YOLOBiDirFunction(model, confidence_threshold, tracker, bidirectional)
