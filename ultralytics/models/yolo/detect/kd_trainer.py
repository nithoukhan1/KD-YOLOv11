import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# Wrapper to inject distillation logic directly into the model object
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
        # FIX: Ensure images match teacher precision (FP16 batch vs FP32 teacher)
        device = batch['img'].device
        dtype = next(self.teacher.parameters()).dtype
        if next(self.teacher.parameters()).device!= device:
            self.teacher.to(device)
            
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'].to(dtype))
            
        # 3. Calculate KD Loss (MSE alignment between Student/Teacher raw logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # 5. CONCATENATE: Ensure tensor size (4) matches header for logging
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        # Surgical Pop: Hide custom keys from strict validator
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def set_model_attributes(self):
        """CRITICAL HOOK: Wrap criterion and initialize logging buffer."""
        super().set_model_attributes()
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            print(f"🚀 Initializing and Injecting Distillation Wrapper into Model.")
            
            # Load and freeze Teacher model
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # Use getattr to safely handle potential wrapping
            m = getattr(self.model, 'module', self.model)
            
            # Ensure student criterion is initialized
            if not hasattr(m, 'criterion') or m.criterion is None:
                m.criterion = m.init_criterion()
            
            # Inject our order-aware wrapper
            m.criterion = DistillLossWrapper(teacher_model, m.criterion, self.kd_weight_val)

        # FIX: Re-initialize tloss to size 4 to prevent empty values in logs/CSV
        self.loss_names = self.custom_loss_names
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Ensures custom loss names survive the validation phase header reset."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Override mapping to ensure 'train/kd_loss' reaches results.csv correctly."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys