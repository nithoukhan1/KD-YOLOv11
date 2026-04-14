import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# Wrapper to inject distillation into the model's internal loss engine
class DistillLossWrapper(nn.Module):
    def __init__(self, teacher, student_criterion, kd_weight):
        super().__init__()
        self.teacher = teacher
        self.base = student_criterion
        self.kd_weight = kd_weight

    def forward(self, preds, batch):
        # 1. Standard Student YOLO losses [box, cls, dfl]
        loss, loss_items = self.base(preds, batch)
        
        # 2. Get Teacher predictions (Soft Targets) for the current batch
        # Ensure teacher is synchronized on the correct GPU rank
        device = batch['img'].device
        if next(self.teacher.parameters()).device!= device:
            self.teacher.to(device)
            
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE alignment between raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # 5. Append KD to logging tensor: [box, cls, dfl] + [kd] = length 4
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Define 4 names EARLY so loggers allocate correct CSV header
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # Surgical Pop: Bypass syntax error by removing custom keys before validation
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model and wrap its criterion before DDP/multi-GPU setup."""
        model = super().get_model(cfg, weights, verbose)
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Wrapping Model Criterion for Distillation")
            
            # Load and freeze Teacher model separately in each process
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # Inject the Distillation Wrapper directly into the Model object
            model.criterion = DistillLossWrapper(teacher_model, model.criterion, self.kd_weight_val)
            
        return model

    def set_model_attributes(self):
        """Force headers and average tracker to support 4 loss values across all ranks."""
        super().set_model_attributes()
        self.loss_names = self.custom_loss_names
        # Re-initialize tloss to size 4 to prevent Rank 0 truncation errors
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Ensures custom loss names survive the Rank 0 header reset during evaluation."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Override mapping to ensure 'train/kd_loss' reaches results.csv correctly."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys