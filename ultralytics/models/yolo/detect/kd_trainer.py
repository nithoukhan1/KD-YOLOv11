import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# Wrapper to inject distillation logic directly into the model's loss engine
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
        # Ensure images match teacher precision (FP16 batch vs FP32 teacher)
        dtype = next(self.teacher.parameters()).dtype
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'].to(dtype))
            
        # 3. Calculate KD Loss (MSE alignment between Student/Teacher scale-logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit.to(torch.float32), t_logit.detach().to(torch.float32))
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # 5. CONCATENATE: Ensuring tensor size (4) matches header for proper logging
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def set_model_attributes(self):
        """DEEP INJECTION FIX: Manually build the model's brain before wrapping it."""
        super().set_model_attributes() # Attaches args to trainer
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            print(f"🚀 Rank {getattr(self, 'rank', -1)}: Injecting Distillation Wrapper.")
            
            # Load Expert Teacher
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # Use getattr to safely handle potential DDP rank wrapping
            m = getattr(self.model, 'module', self.model)
            
            # CRITICAL: Attach hyperparameters and MANUALLY INITIALIZE the default criterion
            m.args = self.args
            m.criterion = m.init_criterion()
            
            # WRAP the initialized criterion with our Distillation logic
            m.criterion = DistillLossWrapper(teacher_model, m.criterion, self.kd_weight_val)

        # Force headers and trackers to support 4 values
        self.loss_names = self.custom_loss_names
        # Re-allocate memory buffer for the 4th value to prevent empty/zero logs
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Prevents Rank 0 header reset during validation steps."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Ensure mapping dictionary is correctly formatted for results.csv."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys