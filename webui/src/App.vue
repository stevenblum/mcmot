<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const apiBase = (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");

const status = ref({
  running: false,
  fps: 0,
  last_loop_ms: 0,
  available_streams: [],
  available_camera_ids: [],
  selected_camera_ids: [],
  available_parts_models: [],
  available_arm_models: [],
  selected_parts_model_path: null,
  parts_model_loaded: false,
  parts_confidence_threshold: 0.25,
  parts_nms_enabled: true,
  parts_nms_iou_threshold: 0.7,
  selected_arm_model_path: null,
  arm_model_loaded: false,
  arm_confidence_threshold: 0.25,
  arm_nms_enabled: true,
  arm_nms_iou_threshold: 0.4,
  arm_inference_interval_sec: 0.5,
  arm_connected: false,
  arm_linked_id: null,
  arm_linked_no_control: null,
  available_models: [],
  selected_model_path: null,
  model_loaded: false,
  confidence_threshold: 0.25,
  overhead_update_interval_sec: 2.0,
  calibration_method: null,
  pending_landmark_pixels: {},
  pending_arm_base_selections: {},
  last_error: null,
});

const ui = reactive({
  leftCamera: null,
  rightCamera: null,
  selectedPartsModelPath: "",
  selectedArmModelPath: "",
  confidenceThreshold: 0.25,
  partsNmsEnabled: true,
  partsNmsIouThreshold: 0.7,
  armConfidenceThreshold: 0.25,
  armNmsEnabled: true,
  armNmsIouThreshold: 0.4,
  armInferenceIntervalSec: 0.5,
  overheadIntervalSec: 2.0,
  calibrationChoice: "",
  autoBaseline: 1.0,
  arucoPositions: "square4",
  arucoSampleFrames: 10,
  arucoSampleInterval: 0.05,
});

const calibrationModal = reactive({
  open: false,
  mode: "auto",
});

const armCalibrationModal = reactive({
  open: false,
  ipAddress: "",
  port: 9090,
  armId: "",
  noControl: false,
  runCalibration: true,
  xValuesText: "0.15,0.20,0.25",
  yValuesText: "-0.05,0.00,0.05",
  zValuesText: "0.08,0.12,0.16",
  settleTimeSec: 0.35,
});

const splitX = ref(50);
const splitY = ref(50);
const stageRef = ref(null);
const landmarkPanelRef = ref(null);
const armPanelRef = ref(null);
const loading = ref(false);
const info = ref("Loading...");
const editingConfidence = ref(false);
const editingArmConfidence = ref(false);
const landmarkRows = ref([]);
const activeLandmarkDefine = ref(null);
const landmarkPanelPos = reactive({ x: 36, y: 90 });
const armPanelPos = reactive({ x: 56, y: 92 });
const armBaseSelectionArmed = ref(false);

let pollTimer = null;
let cameraApplyTimer = null;
let dragAxis = null;
let landmarkPanelDrag = null;
let armPanelDrag = null;
let landmarkRowIdCounter = 0;

const partsModelOptions = computed(() =>
  (status.value.available_parts_models || status.value.available_models || []).map((path) => ({
    path,
    label: path.split(/[/\\]/).pop() || path,
  })),
);

const armModelOptions = computed(() =>
  (status.value.available_arm_models || []).map((path) => ({
    path,
    label: path.split(/[/\\]/).pop() || path,
  })),
);

const overheadIntervalOptions = [0.05, 0.1, 0.25, 0.5, 1, 2, 5];
const armIntervalOptions = [0.1, 0.25, 0.5, 1, 2, 5, 10];

const landmarkPanelStyle = computed(() => ({
  left: `${landmarkPanelPos.x}px`,
  top: `${landmarkPanelPos.y}px`,
}));
const armPanelStyle = computed(() => ({
  left: `${armPanelPos.x}px`,
  top: `${armPanelPos.y}px`,
}));

const landmarkCameraNumbers = computed(() => {
  const count = Number(status.value.selected_camera_ids?.length || 0);
  if (count >= 2) return [0, 1];
  if (count === 1) return [0];
  return [];
});

function cameraLabel(cameraNumber) {
  const port = status.value.selected_camera_ids?.[cameraNumber];
  if (port === undefined || port === null) return `cam${cameraNumber}`;
  return `cam${cameraNumber} (port ${port})`;
}

function createLandmarkRow(x = 0, y = 0, z = 0) {
  landmarkRowIdCounter += 1;
  return {
    id: landmarkRowIdCounter,
    x: String(x),
    y: String(y),
    z: String(z),
    cam0x: null,
    cam0y: null,
    cam1x: null,
    cam1y: null,
  };
}

function resetLandmarkRows() {
  landmarkRows.value = [
    createLandmarkRow(0, 0, 0),
    createLandmarkRow(11, 0, 0),
    createLandmarkRow(0, 8.5, 0),
    createLandmarkRow(11, 8.5, 0),
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function paneStyle(pane) {
  const leftWidth = `${splitX.value}%`;
  const rightWidth = `${100 - splitX.value}%`;
  const topHeight = `${splitY.value}%`;
  const bottomHeight = `${100 - splitY.value}%`;

  if (pane === "tl") return { left: "0%", top: "0%", width: leftWidth, height: topHeight };
  if (pane === "tr") return { left: leftWidth, top: "0%", width: rightWidth, height: topHeight };
  if (pane === "bl") return { left: "0%", top: topHeight, width: leftWidth, height: bottomHeight };
  return { left: leftWidth, top: topHeight, width: rightWidth, height: bottomHeight };
}

function streamUrl(streamName) {
  return `${apiBase}/api/stream/${streamName}.mjpeg`;
}

function hasStream(streamName) {
  return (status.value.available_streams || []).includes(streamName);
}

function normalizeStatus(s) {
  status.value = s;
  const cams = s.available_camera_ids || [];
  if (cams.length > 0) {
    if (ui.leftCamera === null || !cams.includes(ui.leftCamera)) ui.leftCamera = cams[0];
    if (ui.rightCamera === null || !cams.includes(ui.rightCamera)) ui.rightCamera = cams[Math.min(1, cams.length - 1)];
  }

  ui.selectedPartsModelPath = s.selected_parts_model_path ?? s.selected_model_path ?? "";
  ui.selectedArmModelPath = s.selected_arm_model_path ?? "";
  if (!editingConfidence.value) {
    ui.confidenceThreshold = Number(
      s.parts_confidence_threshold ?? s.confidence_threshold ?? ui.confidenceThreshold ?? 0.25,
    );
    ui.partsNmsIouThreshold = Number(
      s.parts_nms_iou_threshold ?? ui.partsNmsIouThreshold ?? 0.7,
    );
  }
  ui.partsNmsEnabled = Boolean(s.parts_nms_enabled ?? ui.partsNmsEnabled ?? true);
  if (!editingArmConfidence.value) {
    ui.armConfidenceThreshold = Number(
      s.arm_confidence_threshold ?? ui.armConfidenceThreshold ?? 0.25,
    );
    ui.armNmsIouThreshold = Number(
      s.arm_nms_iou_threshold ?? ui.armNmsIouThreshold ?? 0.4,
    );
  }
  ui.armNmsEnabled = Boolean(s.arm_nms_enabled ?? ui.armNmsEnabled ?? true);
  const armInterval = Number(s.arm_inference_interval_sec);
  if (Number.isFinite(armInterval) && armInterval >= 0) {
    ui.armInferenceIntervalSec = armInterval;
  }
  const interval = Number(s.overhead_update_interval_sec);
  if (Number.isFinite(interval) && interval >= 0) {
    ui.overheadIntervalSec = interval;
  }
}

async function getStatus() {
  const response = await fetch(`${apiBase}/api/session/status`);
  if (!response.ok) throw new Error(`status ${response.status}`);
  normalizeStatus(await response.json());
}

async function post(path, payload = null) {
  const response = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : null,
  });
  const data = await response.json();
  if (data?.status) normalizeStatus(data.status);
  return data;
}

async function refreshOptions() {
  loading.value = true;
  try {
    const data = await post("/api/options/refresh");
    info.value = data.message || "Options refreshed.";
  } catch (err) {
    info.value = `Refresh failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function applyCameraSelection() {
  if (ui.leftCamera === null || ui.rightCamera === null) return;
  const desired = [Number(ui.leftCamera), Number(ui.rightCamera)];
  const current = status.value.selected_camera_ids || [];
  if (current.length === 2 && current[0] === desired[0] && current[1] === desired[1]) return;

  loading.value = true;
  try {
    const data = await post("/api/cameras/select", {
      camera_device_ids: desired,
      camera_names: ["logitech_1", "logitech_1"],
    });
    info.value = data.message || "Cameras updated.";
  } catch (err) {
    info.value = `Camera update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

function scheduleCameraSelection() {
  if (cameraApplyTimer) clearTimeout(cameraApplyTimer);
  cameraApplyTimer = setTimeout(() => {
    applyCameraSelection().catch(() => {});
  }, 200);
}

async function loadPartsModel() {
  if (!ui.selectedPartsModelPath) return;
  loading.value = true;
  try {
    const data = await post("/api/model/parts/select", {
      model_path: ui.selectedPartsModelPath,
      confidence_threshold: Number(ui.confidenceThreshold),
    });
    info.value = data.message || "Parts model loaded.";
  } catch (err) {
    info.value = `Parts model load failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function loadArmModel() {
  if (!ui.selectedArmModelPath) return;
  loading.value = true;
  try {
    const data = await post("/api/model/arm/select", {
      model_path: ui.selectedArmModelPath,
      confidence_threshold: Number(ui.armConfidenceThreshold),
    });
    info.value = data.message || "Arm model loaded.";
  } catch (err) {
    info.value = `Arm model load failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updateConfidenceThreshold() {
  const value = Number(ui.confidenceThreshold);
  if (!Number.isFinite(value)) return;
  loading.value = true;
  try {
    const data = await post("/api/model/parts/confidence", {
      confidence_threshold: value,
    });
    info.value = data.message || "Parts confidence updated.";
  } catch (err) {
    info.value = `Parts confidence update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updatePartsNmsOptions() {
  const iouValue = Number(ui.partsNmsIouThreshold);
  if (!Number.isFinite(iouValue) || iouValue < 0 || iouValue > 1) {
    info.value = "Parts NMS IoU must be between 0 and 1.";
    return;
  }
  loading.value = true;
  try {
    const data = await post("/api/model/parts/nms", {
      enabled: Boolean(ui.partsNmsEnabled),
      iou_threshold: iouValue,
    });
    info.value = data.message || "Parts NMS updated.";
  } catch (err) {
    info.value = `Parts NMS update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updateArmInferenceInterval() {
  const value = Number(ui.armInferenceIntervalSec);
  if (!Number.isFinite(value) || value < 0) return;
  loading.value = true;
  try {
    const data = await post("/api/model/arm/interval", {
      interval_sec: value,
    });
    info.value = data.message || "Arm interval updated.";
  } catch (err) {
    info.value = `Arm interval update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updateArmConfidenceThreshold() {
  const value = Number(ui.armConfidenceThreshold);
  if (!Number.isFinite(value)) return;
  loading.value = true;
  try {
    const data = await post("/api/model/arm/confidence", {
      confidence_threshold: value,
    });
    info.value = data.message || "Arm confidence updated.";
  } catch (err) {
    info.value = `Arm confidence update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updateArmNmsOptions() {
  const iouValue = Number(ui.armNmsIouThreshold);
  if (!Number.isFinite(iouValue) || iouValue < 0 || iouValue > 1) {
    info.value = "Arm NMS IoU must be between 0 and 1.";
    return;
  }
  loading.value = true;
  try {
    const data = await post("/api/model/arm/nms", {
      enabled: Boolean(ui.armNmsEnabled),
      iou_threshold: iouValue,
    });
    info.value = data.message || "Arm NMS updated.";
  } catch (err) {
    info.value = `Arm NMS update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function openArmCalibrationModal() {
  armCalibrationModal.open = true;
  armCalibrationModal.ipAddress = "";
  armCalibrationModal.port = 9090;
  armCalibrationModal.armId = "";
  armCalibrationModal.noControl = Boolean(status.value.arm_linked_no_control ?? false);
  armCalibrationModal.runCalibration = !armCalibrationModal.noControl;
  armBaseSelectionArmed.value = false;
  centerArmPanel();
  try {
    await post("/api/arm/base-selection/clear");
    if (!(calibrationModal.open && calibrationModal.mode === "landmarks")) {
      await setClickMode("none");
    }
    info.value = "Click Select Base Points, then click base keypoint in cam0 and cam1.";
  } catch (err) {
    info.value = `Arm base selection setup failed: ${err}`;
  }
}

async function startArmBaseSelection() {
  loading.value = true;
  try {
    await post("/api/arm/base-selection/clear");
    await setClickMode("arm_base_select");
    armBaseSelectionArmed.value = true;
    info.value = "Base selection active: click base keypoint in cam0, then cam1.";
  } catch (err) {
    info.value = `Arm base selection setup failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function stopArmBaseSelection() {
  armBaseSelectionArmed.value = false;
  try {
    if (!(calibrationModal.open && calibrationModal.mode === "landmarks")) {
      await setClickMode("none");
    }
  } catch (err) {
    info.value = `Arm base selection cleanup failed: ${err}`;
  }
}

async function toggleArmBaseSelection() {
  if (armBaseSelectionArmed.value) {
    await stopArmBaseSelection();
    info.value = "Base selection paused.";
    return;
  }
  await startArmBaseSelection();
}

async function closeArmCalibrationModal() {
  armCalibrationModal.open = false;
  armBaseSelectionArmed.value = false;
  stopArmPanelDrag();
  try {
    if (!(calibrationModal.open && calibrationModal.mode === "landmarks")) {
      await setClickMode("none");
    }
  } catch (err) {
    info.value = `Arm base selection cleanup failed: ${err}`;
  }
}

function parseNumberList(textValue, label) {
  const values = String(textValue)
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    throw new Error(`${label} must contain at least one numeric value.`);
  }
  return values;
}

async function runAddArmToScene() {
  const pendingBases = status.value.pending_arm_base_selections || {};
  const cam0Selection = pendingBases["0"] || null;
  const cam1Selection = pendingBases["1"] || null;
  const armIdText = String(armCalibrationModal.armId ?? "").trim();
  const selectedArmId = armIdText === "" ? null : Number(armIdText);
  if (selectedArmId !== null && (!Number.isFinite(selectedArmId) || selectedArmId < 0)) {
    info.value = "Arm ID must be a non-negative number, or left blank.";
    return;
  }
  const cam0DetectionIndex =
    cam0Selection && Number.isFinite(Number(cam0Selection.detection_index))
      ? Number(cam0Selection.detection_index)
      : null;
  const cam1DetectionIndex =
    cam1Selection && Number.isFinite(Number(cam1Selection.detection_index))
      ? Number(cam1Selection.detection_index)
      : null;

  if (selectedArmId === null && (cam0DetectionIndex === null || cam1DetectionIndex === null)) {
    info.value = "Select base keypoints in both cam0 and cam1, or enter an Arm ID.";
    return;
  }

  loading.value = true;
  try {
    const noControl = Boolean(armCalibrationModal.noControl);
    const runCalibration = Boolean(armCalibrationModal.runCalibration) && !noControl;
    const payload = {
      ip_address: armCalibrationModal.ipAddress.trim() || null,
      port: Number(armCalibrationModal.port),
      arm_id: selectedArmId,
      cam0_detection_index: cam0DetectionIndex,
      cam1_detection_index: cam1DetectionIndex,
      no_control: noControl,
      run_calibration: runCalibration,
      x_values: runCalibration ? parseNumberList(armCalibrationModal.xValuesText, "X values") : [0.15, 0.2, 0.25],
      y_values: runCalibration ? parseNumberList(armCalibrationModal.yValuesText, "Y values") : [-0.05, 0, 0.05],
      z_values: runCalibration ? parseNumberList(armCalibrationModal.zValuesText, "Z values") : [0.08, 0.12, 0.16],
      settle_time_sec: Number(armCalibrationModal.settleTimeSec),
    };
    const data = await post("/api/arm/add", payload);
    if (data?.ok === false) throw new Error(data.message || "Add arm failed.");
    info.value = data.message || "Arm added to scene.";
    await closeArmCalibrationModal();
  } catch (err) {
    info.value = `Add arm failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function clearArmBaseSelections() {
  loading.value = true;
  try {
    const data = await post("/api/arm/base-selection/clear");
    info.value =
      data.message ||
      (armBaseSelectionArmed.value
        ? "Arm base selections cleared. Click cam0 and cam1 base keypoints."
        : "Arm base selections cleared.");
  } catch (err) {
    info.value = `Arm base clear failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function updateOverheadInterval() {
  const value = Number(ui.overheadIntervalSec);
  if (!Number.isFinite(value) || value < 0) return;
  loading.value = true;
  try {
    const data = await post("/api/overhead/interval", {
      interval_sec: value,
    });
    info.value = data.message || "Overhead interval updated.";
  } catch (err) {
    info.value = `Overhead interval update failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function setCalibrationPreview(mode) {
  await post("/api/calibration/preview", {
    mode,
    aruco_positions: ui.arucoPositions,
  });
}

async function setClickMode(mode) {
  await post("/api/ui/click-mode", { mode });
}

function centerLandmarkPanel() {
  const panelWidth = Math.min(920, Math.max(560, window.innerWidth - 24));
  landmarkPanelPos.x = clamp(
    Math.round((window.innerWidth - panelWidth) / 2),
    8,
    Math.max(8, window.innerWidth - panelWidth - 8),
  );
  landmarkPanelPos.y = 74;
}

function centerArmPanel() {
  const panelWidth = Math.min(680, window.innerWidth - 24);
  armPanelPos.x = clamp(
    Math.round(window.innerWidth - panelWidth - 28),
    8,
    Math.max(8, window.innerWidth - panelWidth - 8),
  );
  armPanelPos.y = 84;
}

async function openCalibrationModal(mode) {
  if (!mode) return;
  if (armCalibrationModal.open) {
    armBaseSelectionArmed.value = false;
  }
  calibrationModal.mode = mode;
  calibrationModal.open = true;
  try {
    if (mode === "aruco") await setCalibrationPreview("aruco");
    if (mode === "landmarks") {
      resetLandmarkRows();
      activeLandmarkDefine.value = null;
      centerLandmarkPanel();
      await setCalibrationPreview("landmarks");
      await setClickMode("landmark_select");
      info.value = "Click Define on a row, then click cam0 and cam1 for that world point.";
    }
  } catch (err) {
    info.value = `Calibration setup failed: ${err}`;
  }
}

async function closeCalibrationModal() {
  const mode = calibrationModal.mode;
  calibrationModal.open = false;
  activeLandmarkDefine.value = null;
  stopLandmarkPanelDrag();
  try {
    if (mode === "landmarks") {
      await setClickMode("none");
      await setCalibrationPreview(null);
    } else if (mode === "aruco") {
      await setCalibrationPreview(null);
    }
  } catch (err) {
    info.value = `Calibration cleanup failed: ${err}`;
  }
}

async function runAutoCalibration() {
  loading.value = true;
  try {
    const data = await post("/api/calibration/auto", {
      baseline: Number(ui.autoBaseline),
    });
    info.value = data.message || "Auto calibration complete.";
  } catch (err) {
    info.value = `Auto calibration failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

async function runArucoCalibration() {
  loading.value = true;
  try {
    const data = await post("/api/calibration/aruco", {
      aruco_positions: ui.arucoPositions,
      sample_frames: Number(ui.arucoSampleFrames),
      sample_interval_sec: Number(ui.arucoSampleInterval),
    });
    info.value = data.message || "ArUco calibration complete.";
  } catch (err) {
    info.value = `ArUco calibration failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

function rowPixel(row, cameraNumber) {
  if (cameraNumber === 0 && Number.isFinite(row.cam0x) && Number.isFinite(row.cam0y)) {
    return { x: Number(row.cam0x), y: Number(row.cam0y) };
  }
  if (cameraNumber === 1 && Number.isFinite(row.cam1x) && Number.isFinite(row.cam1y)) {
    return { x: Number(row.cam1x), y: Number(row.cam1y) };
  }
  return null;
}

function setRowPixel(row, cameraNumber, pixelXY) {
  if (cameraNumber === 0) {
    row.cam0x = Number(pixelXY[0]);
    row.cam0y = Number(pixelXY[1]);
  } else if (cameraNumber === 1) {
    row.cam1x = Number(pixelXY[0]);
    row.cam1y = Number(pixelXY[1]);
  }
}

function clearRowPixels(row) {
  row.cam0x = null;
  row.cam0y = null;
  row.cam1x = null;
  row.cam1y = null;
}

function defineButtonLabel(rowId) {
  const active = activeLandmarkDefine.value;
  if (!active || active.rowId !== rowId) return "Define";
  return active.nextCamera === 0 ? "Click cam0..." : "Click cam1...";
}

function startDefineLandmarkRow(rowId) {
  const row = landmarkRows.value.find((item) => item.id === rowId);
  if (!row) return;
  clearRowPixels(row);
  activeLandmarkDefine.value = { rowId, nextCamera: 0 };
  const rowIndex = landmarkRows.value.findIndex((item) => item.id === rowId);
  info.value = `Point ${rowIndex + 1}: click ${cameraLabel(0)}.`;
}

function addLandmarkRow() {
  landmarkRows.value.push(createLandmarkRow(0, 0, 0));
}

function clearLandmarkPixelsInRows() {
  for (const row of landmarkRows.value) clearRowPixels(row);
}

async function clearLandmarks() {
  loading.value = true;
  try {
    const cameraNumbers = landmarkCameraNumbers.value;
    for (const cameraNumber of cameraNumbers) {
      await post(`/api/calibration/landmarks/clear?camera_number=${cameraNumber}`);
    }
    clearLandmarkPixelsInRows();
    activeLandmarkDefine.value = null;
    info.value = "Landmarks cleared.";
  } catch (err) {
    info.value = `Landmark clear failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

function parseWorldPoint(row, index) {
  const x = Number(row.x);
  const y = Number(row.y);
  const z = Number(row.z);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    throw new Error(`Point ${index + 1} has invalid world coordinates.`);
  }
  return [x, y, z];
}

function rowsDefinedForCameras(cameraNumbers) {
  return landmarkRows.value.filter((row) => cameraNumbers.every((cameraNumber) => rowPixel(row, cameraNumber)));
}

async function solveLandmarks() {
  const cameraNumbers = landmarkCameraNumbers.value;
  if (cameraNumbers.length < 2) {
    info.value = "Need two active cameras to run landmark calibration.";
    return;
  }

  const rows = rowsDefinedForCameras(cameraNumbers);
  if (rows.length < 4) {
    info.value = "Need at least 4 points with both cam0 and cam1 pixels defined.";
    return;
  }

  loading.value = true;
  try {
    for (const cameraNumber of cameraNumbers) {
      await post(`/api/calibration/landmarks/clear?camera_number=${cameraNumber}`);
    }

    for (const row of rows) {
      const rowIndex = landmarkRows.value.findIndex((item) => item.id === row.id);
      const world = parseWorldPoint(row, rowIndex);
      for (const cameraNumber of cameraNumbers) {
        const pixel = rowPixel(row, cameraNumber);
        await post("/api/calibration/landmarks/add", {
          camera_number: cameraNumber,
          pixel_xy: [pixel.x, pixel.y],
          world_xyz: world,
        });
      }
    }

    const data = await post("/api/calibration/landmarks/solve-all", {
      camera_numbers: cameraNumbers,
    });
    if (data?.ok === false) throw new Error(data.message || "Landmark calibration failed.");
    info.value = data.message || `Landmark calibration complete with ${rows.length} paired points.`;
    activeLandmarkDefine.value = null;
  } catch (err) {
    info.value = `Landmark solve failed: ${err}`;
  } finally {
    loading.value = false;
  }
}

function streamClickToPixel(event) {
  const el = event.currentTarget;
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;

  const naturalWidth = el.naturalWidth || rect.width;
  const naturalHeight = el.naturalHeight || rect.height;
  if (!naturalWidth || !naturalHeight) return null;

  const scale = Math.min(rect.width / naturalWidth, rect.height / naturalHeight);
  const renderedWidth = naturalWidth * scale;
  const renderedHeight = naturalHeight * scale;
  const padX = (rect.width - renderedWidth) / 2;
  const padY = (rect.height - renderedHeight) / 2;

  const localX = event.clientX - rect.left - padX;
  const localY = event.clientY - rect.top - padY;
  if (localX < 0 || localY < 0 || localX > renderedWidth || localY > renderedHeight) return null;

  return {
    x: Number((localX / scale).toFixed(2)),
    y: Number((localY / scale).toFixed(2)),
    image_width: Math.round(naturalWidth),
    image_height: Math.round(naturalHeight),
  };
}

function streamNameToCameraNumber(streamName) {
  if (!streamName.startsWith("cam")) return null;
  const suffix = streamName.slice(3);
  if (!/^\d+$/.test(suffix)) return null;
  return Number(suffix);
}

function assignPixelToActiveLandmark(cameraNumber, pixelXY) {
  const active = activeLandmarkDefine.value;
  if (!active) return false;

  const row = landmarkRows.value.find((item) => item.id === active.rowId);
  if (!row) return false;
  const rowIndex = landmarkRows.value.findIndex((item) => item.id === row.id);
  const expected = Number(active.nextCamera);

  if (cameraNumber !== expected) {
    info.value = `Point ${rowIndex + 1}: click ${cameraLabel(expected)} next.`;
    return true;
  }

  setRowPixel(row, cameraNumber, pixelXY);
  if (expected === 0) {
    active.nextCamera = 1;
    info.value = `Point ${rowIndex + 1}: ${cameraLabel(0)} saved. Now click ${cameraLabel(1)}.`;
    return true;
  }

  activeLandmarkDefine.value = null;
  info.value = `Point ${rowIndex + 1} complete on cam0 and cam1.`;
  return true;
}

async function onStreamClick(streamName, event) {
  const click = streamClickToPixel(event);
  if (!click) return;
  try {
    const data = await post("/api/ui/click", {
      stream_name: streamName,
      x: click.x,
      y: click.y,
      image_width: click.image_width,
      image_height: click.image_height,
      button: "left",
      modifiers: [],
      metadata: { source: "webui" },
    });

    const resolved = data?.data?.resolved_target || null;
    if (resolved?.action === "set_pending_landmark") {
      const resolvedCamera = Number(resolved.camera_number);
      const pixel = Array.isArray(resolved.pixel_xy) ? resolved.pixel_xy : [click.x, click.y];
      if (calibrationModal.open && calibrationModal.mode === "landmarks") {
        const consumed = assignPixelToActiveLandmark(resolvedCamera, pixel);
        if (consumed) return;
      }
      info.value = `Selected landmark pixel on cam${resolvedCamera}`;
      return;
    }

    if (resolved?.action === "set_pending_arm_base") {
      const pending = data?.data?.pending_arm_base_selections || {};
      const hasCam0 = !!pending["0"];
      const hasCam1 = !!pending["1"];
      if (hasCam0 && hasCam1) {
        armBaseSelectionArmed.value = false;
        if (!(calibrationModal.open && calibrationModal.mode === "landmarks")) {
          await setClickMode("none");
        }
        info.value = "Arm base selected in cam0 and cam1. Click Add Arm or re-run Select Base Points.";
      } else {
        const nextCamera = hasCam0 ? 1 : 0;
        info.value =
          `Selected arm base on cam${resolved.camera_number} (detection ${resolved.detection_index}). ` +
          `Click cam${nextCamera} next.`;
      }
      return;
    }

    const streamCameraNumber = streamNameToCameraNumber(streamName);
    if (
      calibrationModal.open &&
      calibrationModal.mode === "landmarks" &&
      activeLandmarkDefine.value &&
      streamCameraNumber !== null
    ) {
      const consumed = assignPixelToActiveLandmark(streamCameraNumber, [click.x, click.y]);
      if (consumed) return;
    }

    const hit = data?.data?.detection_hit || null;
    if (hit?.track_id !== null && hit?.track_id !== undefined) {
      info.value = `Clicked track ${hit.track_id} on cam${hit.camera_number}`;
      return;
    }
    info.value = data.message || "Click captured.";
  } catch (err) {
    info.value = `Click failed: ${err}`;
  }
}

function startDrag(axis, event) {
  dragAxis = axis;
  event.preventDefault();
  window.addEventListener("pointermove", onDividerDrag);
  window.addEventListener("pointerup", stopDrag);
  document.body.classList.add("dragging-divider");
}

function onDividerDrag(event) {
  if (!dragAxis || !stageRef.value) return;
  const rect = stageRef.value.getBoundingClientRect();
  if (dragAxis === "vertical") {
    const pct = ((event.clientX - rect.left) / rect.width) * 100;
    splitX.value = clamp(pct, 18, 82);
  } else {
    const pct = ((event.clientY - rect.top) / rect.height) * 100;
    splitY.value = clamp(pct, 18, 82);
  }
}

function stopDrag() {
  dragAxis = null;
  window.removeEventListener("pointermove", onDividerDrag);
  window.removeEventListener("pointerup", stopDrag);
  document.body.classList.remove("dragging-divider");
}

function startLandmarkPanelDrag(event) {
  if (event.button !== 0) return;
  landmarkPanelDrag = {
    dx: event.clientX - landmarkPanelPos.x,
    dy: event.clientY - landmarkPanelPos.y,
  };
  window.addEventListener("pointermove", onLandmarkPanelDrag);
  window.addEventListener("pointerup", stopLandmarkPanelDrag);
  document.body.classList.add("dragging-panel");
}

function startArmPanelDrag(event) {
  if (event.button !== 0) return;
  armPanelDrag = {
    dx: event.clientX - armPanelPos.x,
    dy: event.clientY - armPanelPos.y,
  };
  window.addEventListener("pointermove", onArmPanelDrag);
  window.addEventListener("pointerup", stopArmPanelDrag);
  document.body.classList.add("dragging-panel");
}

function onArmPanelDrag(event) {
  if (!armPanelDrag) return;
  const panelWidth = armPanelRef.value?.offsetWidth || 680;
  const panelHeight = armPanelRef.value?.offsetHeight || 460;
  armPanelPos.x = clamp(
    event.clientX - armPanelDrag.dx,
    8,
    Math.max(8, window.innerWidth - panelWidth - 8),
  );
  armPanelPos.y = clamp(
    event.clientY - armPanelDrag.dy,
    8,
    Math.max(8, window.innerHeight - panelHeight - 8),
  );
}

function stopArmPanelDrag() {
  if (!armPanelDrag) return;
  armPanelDrag = null;
  window.removeEventListener("pointermove", onArmPanelDrag);
  window.removeEventListener("pointerup", stopArmPanelDrag);
  document.body.classList.remove("dragging-panel");
}

function onLandmarkPanelDrag(event) {
  if (!landmarkPanelDrag) return;
  const panelWidth = landmarkPanelRef.value?.offsetWidth || 760;
  const panelHeight = landmarkPanelRef.value?.offsetHeight || 440;
  landmarkPanelPos.x = clamp(
    event.clientX - landmarkPanelDrag.dx,
    8,
    Math.max(8, window.innerWidth - panelWidth - 8),
  );
  landmarkPanelPos.y = clamp(
    event.clientY - landmarkPanelDrag.dy,
    8,
    Math.max(8, window.innerHeight - panelHeight - 8),
  );
}

function stopLandmarkPanelDrag() {
  if (!landmarkPanelDrag) return;
  landmarkPanelDrag = null;
  window.removeEventListener("pointermove", onLandmarkPanelDrag);
  window.removeEventListener("pointerup", stopLandmarkPanelDrag);
  document.body.classList.remove("dragging-panel");
}

onMounted(async () => {
  resetLandmarkRows();
  try {
    await getStatus();
    info.value = "Ready.";
  } catch (err) {
    info.value = `Failed to fetch status: ${err}`;
  }
  pollTimer = setInterval(() => {
    getStatus().catch(() => {});
  }, 500);
});

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer);
  if (cameraApplyTimer) clearTimeout(cameraApplyTimer);
  stopDrag();
  stopLandmarkPanelDrag();
  stopArmPanelDrag();
});
</script>

<template>
  <main class="app-shell">
    <header class="toolbar">
      <div class="toolbar-group">
        <label class="toolbar-field">
          Left Camera
          <select v-model.number="ui.leftCamera" @change="scheduleCameraSelection">
            <option v-for="cid in status.available_camera_ids" :key="`left-${cid}`" :value="cid">Port {{ cid }}</option>
          </select>
        </label>
        <label class="toolbar-field">
          Right Camera
          <select v-model.number="ui.rightCamera" @change="scheduleCameraSelection">
            <option v-for="cid in status.available_camera_ids" :key="`right-${cid}`" :value="cid">Port {{ cid }}</option>
          </select>
        </label>
        <label class="toolbar-field">
          Calibration
          <select
            v-model="ui.calibrationChoice"
            @change="
              (event) => {
                const mode = event.target.value;
                event.target.value = '';
                ui.calibrationChoice = '';
                openCalibrationModal(mode);
              }
            "
          >
            <option value="">Select...</option>
            <option value="auto">Auto</option>
            <option value="aruco">ArUco</option>
            <option value="landmarks">Landmarks</option>
          </select>
        </label>
      </div>

      <div class="toolbar-divider" aria-hidden="true"></div>

      <div class="toolbar-group">
        <label class="toolbar-field toolbar-model">
          Parts Detection Model
          <select v-model="ui.selectedPartsModelPath" @change="loadPartsModel">
            <option value="">Select parts model...</option>
            <option v-for="m in partsModelOptions" :key="`parts-${m.path}`" :value="m.path">{{ m.label }}</option>
          </select>
        </label>
        <label class="toolbar-field toolbar-confidence">
          Confidence
          <input
            v-model.number="ui.confidenceThreshold"
            type="number"
            min="0"
            max="1"
            step="0.01"
            @focus="editingConfidence = true"
            @blur="
              editingConfidence = false;
              updateConfidenceThreshold();
            "
            @keydown.enter.prevent="
              editingConfidence = false;
              updateConfidenceThreshold();
            "
          />
        </label>
        <label class="toolbar-field toolbar-inline">
          <span>Parts NMS</span>
          <input
            v-model="ui.partsNmsEnabled"
            type="checkbox"
            @change="updatePartsNmsOptions"
          />
        </label>
        <label class="toolbar-field toolbar-iou">
          NMS IoU
          <input
            v-model.number="ui.partsNmsIouThreshold"
            type="number"
            min="0"
            max="1"
            step="0.01"
            @blur="updatePartsNmsOptions"
            @keydown.enter.prevent="updatePartsNmsOptions"
          />
        </label>
      </div>

      <div class="toolbar-divider" aria-hidden="true"></div>

      <div class="toolbar-group">
        <label class="toolbar-field toolbar-model">
          Arm Detection Model
          <select v-model="ui.selectedArmModelPath" @change="loadArmModel">
            <option value="">Select arm model...</option>
            <option v-for="m in armModelOptions" :key="`arm-${m.path}`" :value="m.path">{{ m.label }}</option>
          </select>
        </label>
        <label class="toolbar-field toolbar-confidence">
          Confidence
          <input
            v-model.number="ui.armConfidenceThreshold"
            type="number"
            min="0"
            max="1"
            step="0.01"
            @focus="editingArmConfidence = true"
            @blur="
              editingArmConfidence = false;
              updateArmConfidenceThreshold();
            "
            @keydown.enter.prevent="
              editingArmConfidence = false;
              updateArmConfidenceThreshold();
            "
          />
        </label>
        <label class="toolbar-field toolbar-inline">
          <span>Arm NMS</span>
          <input
            v-model="ui.armNmsEnabled"
            type="checkbox"
            @change="updateArmNmsOptions"
          />
        </label>
        <label class="toolbar-field toolbar-iou">
          NMS IoU
          <input
            v-model.number="ui.armNmsIouThreshold"
            type="number"
            min="0"
            max="1"
            step="0.01"
            @blur="updateArmNmsOptions"
            @keydown.enter.prevent="updateArmNmsOptions"
          />
        </label>
        <label class="toolbar-field">
          Arm Interval (sec)
          <select v-model.number="ui.armInferenceIntervalSec" @change="updateArmInferenceInterval">
            <option v-for="v in armIntervalOptions" :key="`arm-int-${v}`" :value="v">{{ v }}</option>
          </select>
        </label>
        <button class="secondary" :disabled="loading" @click="openArmCalibrationModal">Add Arm</button>
      </div>

      <div class="toolbar-divider" aria-hidden="true"></div>

      <div class="toolbar-group">
        <label class="toolbar-field">
          Overhead (sec)
          <select v-model.number="ui.overheadIntervalSec" @change="updateOverheadInterval">
            <option v-for="v in overheadIntervalOptions" :key="`ovh-${v}`" :value="v">{{ v }}</option>
          </select>
        </label>
        <button class="secondary" :disabled="loading" @click="refreshOptions">Refresh</button>
      </div>

      <div class="toolbar-status">
        <span>FPS {{ status.fps?.toFixed?.(1) ?? status.fps }}</span>
        <span>Loop {{ status.last_loop_ms?.toFixed?.(1) ?? status.last_loop_ms }} ms</span>
        <span>Overhead {{ status.overhead_update_interval_sec ?? ui.overheadIntervalSec }} s</span>
        <span>Arm {{ status.arm_inference_interval_sec ?? ui.armInferenceIntervalSec }} s</span>
        <span v-if="status.arm_linked_no_control === true">Arm Control: disabled</span>
        <span>Mode {{ status.calibration_method || "none" }}</span>
        <span v-if="status.last_error" class="status-error">Error: {{ status.last_error }}</span>
        <span>{{ info }}</span>
      </div>
    </header>

    <section ref="stageRef" class="frame-stage">
      <article class="pane" :style="paneStyle('tl')">
        <div class="pane-title">Camera 0</div>
        <div class="pane-content">
          <img
            v-if="hasStream('cam0')"
            class="pane-stream clickable"
            :src="streamUrl('cam0')"
            alt="cam0"
            @click="onStreamClick('cam0', $event)"
          />
          <div v-else class="pane-placeholder">No stream</div>
        </div>
      </article>

      <article class="pane" :style="paneStyle('tr')">
        <div class="pane-title">Camera 1</div>
        <div class="pane-content">
          <img
            v-if="hasStream('cam1')"
            class="pane-stream clickable"
            :src="streamUrl('cam1')"
            alt="cam1"
            @click="onStreamClick('cam1', $event)"
          />
          <div v-else class="pane-placeholder">No stream</div>
        </div>
      </article>

      <article class="pane" :style="paneStyle('bl')">
        <div class="pane-title">Reserved</div>
        <div class="pane-content">
          <div class="pane-placeholder">Blank panel</div>
        </div>
      </article>

      <article class="pane" :style="paneStyle('br')">
        <div class="pane-title">Overhead View</div>
        <div class="pane-content">
          <img v-if="hasStream('overhead')" class="pane-stream" :src="streamUrl('overhead')" alt="overhead" />
          <div v-else class="pane-placeholder">No overhead stream</div>
        </div>
      </article>

      <div class="divider divider-v" :style="{ left: `${splitX}%` }" @pointerdown="startDrag('vertical', $event)"></div>
      <div class="divider divider-h" :style="{ top: `${splitY}%` }" @pointerdown="startDrag('horizontal', $event)"></div>
      <div class="divider-cross" :style="{ left: `${splitX}%`, top: `${splitY}%` }"></div>
    </section>

    <div
      v-if="calibrationModal.open && calibrationModal.mode !== 'landmarks'"
      class="modal-backdrop"
      @click.self="closeCalibrationModal"
    >
      <section class="modal-card">
        <div class="modal-head">
          <h3>Calibration: {{ calibrationModal.mode }}</h3>
          <button class="secondary" @click="closeCalibrationModal">Close</button>
        </div>

        <div class="modal-body" v-if="calibrationModal.mode === 'auto'">
          <label class="toolbar-field">
            Baseline
            <input v-model.number="ui.autoBaseline" type="number" step="0.1" min="0.1" />
          </label>
          <button :disabled="loading" @click="runAutoCalibration">Run Auto Calibration</button>
        </div>

        <div class="modal-body" v-else-if="calibrationModal.mode === 'aruco'">
          <label class="toolbar-field">
            ArUco Positions
            <input v-model="ui.arucoPositions" />
          </label>
          <label class="toolbar-field">
            Sample Frames
            <input v-model.number="ui.arucoSampleFrames" type="number" min="1" step="1" />
          </label>
          <label class="toolbar-field">
            Sample Interval (sec)
            <input v-model.number="ui.arucoSampleInterval" type="number" min="0" step="0.01" />
          </label>
          <button :disabled="loading" @click="runArucoCalibration">Run ArUco Calibration</button>
        </div>
      </section>
    </div>

    <section
      v-if="armCalibrationModal.open"
      ref="armPanelRef"
      class="modal-card arm-floating"
      :style="armPanelStyle"
    >
      <div class="modal-head floating-head" @pointerdown="startArmPanelDrag">
          <h3>Add Arm to Scene</h3>
          <button class="secondary" @pointerdown.stop @click="closeArmCalibrationModal">Close</button>
      </div>

      <div class="modal-body arm-body">
          <div class="modal-grid">
            <label class="toolbar-field">
              Robot IP
              <input v-model="armCalibrationModal.ipAddress" placeholder="192.168.0.10" />
            </label>
            <label class="toolbar-field">
              Port
              <input v-model.number="armCalibrationModal.port" type="number" min="1" max="65535" />
            </label>
            <label class="toolbar-field">
              Arm ID (optional)
              <input v-model="armCalibrationModal.armId" placeholder="auto-detect" />
            </label>
          </div>

          <div class="modal-actions">
            <label class="toolbar-field toolbar-inline">
              <span>No Control</span>
              <input
                v-model="armCalibrationModal.noControl"
                type="checkbox"
                @change="
                  () => {
                    if (armCalibrationModal.noControl) armCalibrationModal.runCalibration = false;
                  }
                "
              />
            </label>
            <label class="toolbar-field toolbar-inline">
              <span>Run Calibration</span>
              <input
                v-model="armCalibrationModal.runCalibration"
                type="checkbox"
                :disabled="armCalibrationModal.noControl"
              />
            </label>
          </div>

          <div class="modal-grid">
            <label class="toolbar-field">
              X Grid Values
              <input v-model="armCalibrationModal.xValuesText" :disabled="armCalibrationModal.noControl || !armCalibrationModal.runCalibration" />
            </label>
            <label class="toolbar-field">
              Y Grid Values
              <input v-model="armCalibrationModal.yValuesText" :disabled="armCalibrationModal.noControl || !armCalibrationModal.runCalibration" />
            </label>
            <label class="toolbar-field">
              Z Grid Values
              <input v-model="armCalibrationModal.zValuesText" :disabled="armCalibrationModal.noControl || !armCalibrationModal.runCalibration" />
            </label>
            <label class="toolbar-field">
              Settle Time (sec)
              <input
                v-model.number="armCalibrationModal.settleTimeSec"
                type="number"
                min="0"
                step="0.05"
                :disabled="armCalibrationModal.noControl || !armCalibrationModal.runCalibration"
              />
            </label>
          </div>

          <div class="modal-actions">
            <button class="secondary" :disabled="loading" @click="toggleArmBaseSelection">
              {{ armBaseSelectionArmed ? "Stop Base Selection" : "Select Base Points" }}
            </button>
            <button class="secondary" :disabled="loading" @click="clearArmBaseSelections">Clear Base Picks</button>
          </div>

          <p class="hint">
            Keep arm detections visible in both cameras. If No Control is enabled, this arm is tracked only and cannot be moved.
          </p>
          <p class="hint">
            {{
              armBaseSelectionArmed
                ? "Base selection is active. Click base keypoint in cam0 and cam1."
                : "Base selection is paused. Click Select Base Points to choose base keypoints."
            }}
          </p>
          <p class="hint">
            Cam0 base:
            {{
              status.pending_arm_base_selections?.["0"]
                ? `detection ${status.pending_arm_base_selections["0"].detection_index}`
                : "not selected"
            }}
            |
            Cam1 base:
            {{
              status.pending_arm_base_selections?.["1"]
                ? `detection ${status.pending_arm_base_selections["1"].detection_index}`
                : "not selected"
            }}
          </p>

          <div class="modal-actions">
            <button :disabled="loading" @click="runAddArmToScene">Add Arm</button>
            <button class="secondary" :disabled="loading" @click="closeArmCalibrationModal">Cancel</button>
          </div>
      </div>
    </section>

    <section
      v-if="calibrationModal.open && calibrationModal.mode === 'landmarks'"
      ref="landmarkPanelRef"
      class="modal-card landmark-floating"
      :style="landmarkPanelStyle"
    >
      <div class="modal-head floating-head" @pointerdown="startLandmarkPanelDrag">
        <h3>Calibration: landmarks</h3>
        <button class="secondary" @pointerdown.stop @click="closeCalibrationModal">Close</button>
      </div>

      <div class="modal-body landmark-body">
        <p class="hint">
          Each point needs two clicks in order: first {{ cameraLabel(0) }}, then {{ cameraLabel(1) }}.
        </p>
        <p v-if="landmarkCameraNumbers.length < 2" class="hint status-error">Need two active cameras for this calibration mode.</p>

        <div class="landmark-table-wrap">
          <table class="landmark-table">
            <thead>
              <tr>
                <th>#</th>
                <th>X</th>
                <th>Y</th>
                <th>Z</th>
                <th>Define</th>
                <th>Cam0 Pixel</th>
                <th>Cam1 Pixel</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, idx) in landmarkRows"
                :key="row.id"
                :class="{ 'landmark-row-active': activeLandmarkDefine && row.id === activeLandmarkDefine.rowId }"
              >
                <td>{{ idx + 1 }}</td>
                <td><input v-model="row.x" inputmode="decimal" /></td>
                <td><input v-model="row.y" inputmode="decimal" /></td>
                <td><input v-model="row.z" inputmode="decimal" /></td>
                <td>
                  <button
                    class="secondary landmark-define-btn"
                    :class="{ active: activeLandmarkDefine && row.id === activeLandmarkDefine.rowId }"
                    :disabled="loading"
                    @click="startDefineLandmarkRow(row.id)"
                  >
                    {{ defineButtonLabel(row.id) }}
                  </button>
                </td>
                <td>
                  <span v-if="row.cam0x !== null && row.cam0y !== null" class="landmark-pixel-set">
                    ({{ row.cam0x.toFixed(1) }}, {{ row.cam0y.toFixed(1) }})
                  </span>
                  <span v-else class="landmark-pixel-missing">not defined</span>
                </td>
                <td>
                  <span v-if="row.cam1x !== null && row.cam1y !== null" class="landmark-pixel-set">
                    ({{ row.cam1x.toFixed(1) }}, {{ row.cam1y.toFixed(1) }})
                  </span>
                  <span v-else class="landmark-pixel-missing">not defined</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="modal-actions">
          <button class="secondary" :disabled="loading" @click="addLandmarkRow">Add Point</button>
          <button class="secondary" :disabled="loading" @click="clearLandmarks">Clear Points</button>
          <button :disabled="loading" @click="solveLandmarks">Solve Landmark Calibration</button>
        </div>
      </div>
    </section>
  </main>
</template>
