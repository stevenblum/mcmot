from roboflow import Roboflow

# Weights Path

rf = Roboflow(api_key="j1L0yHCnedl0OCA59vof")
print(rf.workspace())
workspace = rf.workspace("scb-ysrqc")

workspace.deploy_model(
  model_type="yolov11",
  model_path="/home/scblum/Projects/testbed_cv/saved_models/part_detection_25_10_16_03_30",
  project_ids=["testbed_cv-ny15i"],
  model_name="2025-10-16-From-Laptop-w-Hanoi3"
)