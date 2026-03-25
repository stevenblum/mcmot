# python ned2_pose/ned2_pose_train.py
from pathlib import Path

import torch
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.torch_utils import unwrap_model

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

# Download dataset into the script's parent directory.
dataset_dir = parent_dir / "ned2_pose_dataset"
rf = Roboflow(api_key="1XWtPPKCS4CZkESoJgpu")
project = rf.workspace("stevens-workspace-lwc6v").project("ned2_pose")
version = project.version(10)
dataset = version.download("yolov8", location=str(dataset_dir))

# Determine the best available device.
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print("Using CPU")


class PoseLossTrainer(PoseTrainer):
    def validate(self):
        # Run standard validation to get metrics dict.
        metrics, default_fitness = super().validate()
        if metrics is None:
            return None, None

        # Use validation pose loss as fitness target (lower pose loss is better).
        current_pose_loss = metrics.get("val/pose_loss")
        if current_pose_loss is None:
            # Fallback to Ultralytics default fitness if key is unavailable.
            fitness = default_fitness
            if self.best_fitness is None or fitness > self.best_fitness:
                self.best_fitness = fitness
            return metrics, fitness

        fitness = -float(current_pose_loss)

        # Keep an internal best tracker for pose-loss-based fitness only.
        if not hasattr(self, "_best_pose_fitness"):
            self._best_pose_fitness = None
        if self._best_pose_fitness is None or fitness > self._best_pose_fitness:
            self._best_pose_fitness = fitness
        self.best_fitness = self._best_pose_fitness

        return metrics, fitness


class WeightedKeypointLoss(torch.nn.Module):
    """Keypoint loss with per-keypoint weighting."""

    def __init__(self, sigmas: torch.Tensor, kpt_weights: torch.Tensor):
        super().__init__()
        self.sigmas = sigmas
        self.kpt_weights = kpt_weights

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)

        # Increase/reduce contribution for each keypoint index.
        weighted = (1 - torch.exp(-e)) * kpt_mask * self.kpt_weights.view(1, -1)
        return (kpt_loss_factor.view(-1, 1) * weighted).mean()


class WeightedPoseLoss(v8PoseLoss):
    """Pose loss that uses custom per-keypoint weights."""

    def __init__(self, model, kpt_weights: list[float]):
        super().__init__(model)
        nkpt = self.kpt_shape[0]
        if len(kpt_weights) != nkpt:
            raise ValueError(f"Expected {nkpt} keypoint weights, got {len(kpt_weights)}")

        weight_tensor = torch.tensor(kpt_weights, dtype=torch.float32, device=self.device)
        self.keypoint_loss = WeightedKeypointLoss(sigmas=self.keypoint_loss.sigmas, kpt_weights=weight_tensor)


class WeightedPoseLossTrainer(PoseLossTrainer):
    # Keypoint order is [base, shoulder, elbow, wrist, tcp].
    # Increase tcp (index 4) relative to the others.
    KEYPOINT_WEIGHTS = [1.0, 1.0, 1.0, 2.0, 4.0]

    @classmethod
    def build_loss(cls, model):
        # Convenience hook mirroring older trainer patterns.
        return WeightedPoseLoss(model, kpt_weights=list(cls.KEYPOINT_WEIGHTS))

    def _setup_train(self):
        super()._setup_train()

        # Inject weighted pose loss only for this training run.
        # Model class remains the standard Ultralytics PoseModel, so saved checkpoints stay compatible.
        train_model = unwrap_model(self.model)
        train_model.criterion = self.build_loss(train_model)

        # Ensure EMA checkpoint model does not carry custom criterion objects.
        if self.ema is not None and hasattr(self.ema, "ema") and hasattr(self.ema.ema, "criterion"):
            self.ema.ema.criterion = None


model = YOLO("yolo11l-pose.pt")

# Save training outputs in the script's parent directory.
train_output_dir = parent_dir / "ned2_pose_train_results"
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10000,
    patience=100,
    batch=8,
    imgsz=1200,
    freeze=6,
    project=str(train_output_dir),
    name="run",
    plots=True,
    device=device,
    trainer=WeightedPoseLossTrainer,
)
