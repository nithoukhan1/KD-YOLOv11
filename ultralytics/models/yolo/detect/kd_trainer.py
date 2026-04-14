import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# Wrapper to inject distillation directly into the model's loss engine
class DistillLossWrapper(nn.Module):
    def __init__(self, teacher, student_criterion, kd_weight):
        super().__init__()
        self.teacher = teacher
        self.base = student_criterion
        self.kd_weight = kd_weight

    def forward(self, preds, batch):
        # 1. Calculate standard Student YOLO losses [box, cls, dfl]
        loss, loss_items = self.base(preds, batch)
        
        # 2. Get Teacher predictions (Soft Targets)
        # In single GPU mode, teacher and student are on the same device
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE alignment between raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # 5. CONCATENATE for logging: [box, cls, dfl] + [kd] = length 4
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # Surgical Pop: Bypass the YOLO argument validation error
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model and wrap its criterion."""
        model = super().get_model(cfg, weights, verbose)
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            print(f"🚀 Initializing Expert Teacher: {self.teacher_weights_path}")
            
            # Load and freeze Teacher model
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # Inject the wrapper directly into the model object
            # This ensures the loss calculation happens inside the training loop
            model.criterion = DistillLossWrapper(teacher_model, model.criterion, self.kd_weight_val)
            
        return model

    def set_model_attributes(self):
        """Force headers and trackers to support 4 values."""
        super().set_model_attributes()
        self.loss_names = self.custom_loss_names
        # Re-initialize tloss to size 4 to prevent value truncation
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Ensure custom loss names survive during validation steps."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Ensure 'train/kd_loss' mapping exists for the CSV logger."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys