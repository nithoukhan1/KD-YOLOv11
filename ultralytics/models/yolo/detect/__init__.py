# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
# Add your new trainer here
from.kd_trainer import KDDetectionTrainer

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "KDDetectionTrainer"
