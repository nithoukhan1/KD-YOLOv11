import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer
from ultralytics.cfg import DEFAULT_CFG_DICT

# 1. PERMANENT VALIDATION BYPASS
# Adding these to global defaults ensures they are recognized by all DDP subprocesses
if 'teacher_weights' not in DEFAULT_CFG_DICT:
    DEFAULT_CFG_DICT['teacher_weights'] = None
if 'kd_weight' not in DEFAULT_CFG_DICT:
    DEFAULT_CFG_DICT['kd_weight'] = 0.5

# 2. THE DISTILLATION ENGINE
# This wrapper forces the model to calculate and return the 4th loss component
class DistillLossWrapper(nn.Module):
    def __init__(self, teacher, student_criterion, kd_weight):
        super().__init__()
        self.teacher = teacher
        self.base = student_criterion
        self.kd_weight = kd_weight

    def forward(self, preds, batch):
        # A. Calculate standard Student YOLO losses [box, cls, dfl]
        loss, loss_items = self.base(preds, batch)
        
        # B. Get Teacher predictions (Soft Targets) on current GPU device
        device = batch['img'].device
        if next(self.teacher.parameters()).device!= device:
            self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # C. Calculate KD Loss (MSE alignment between raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
        weighted_kd = self.kd_weight * kd_loss
        
        # D. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # E. CRITICAL: Concatenate for logging so it reaches results.csv/console
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # custom_params logic is no longer needed thanks to Step 1
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model and wrap its criterion before DDP initialization."""
        model = super().get_model(cfg, weights, verbose)
        
        if self.args.teacher_weights:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Injecting Distillation into Model Criterion")
            
            # Load and freeze Teacher model separately in each GPU process
            teacher_model = YOLO(self.args.teacher_weights).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # DEEP INJECTION: Replace model criterion with our wrapper
            model.criterion = DistillLossWrapper(teacher_model, model.criterion, float(self.args.kd_weight))
            
        return model

    def _setup_train(self, *args, **kwargs):
        """Force average tracker to support 4 loss values across all ranks."""
        super()._setup_train(*args, **kwargs)
        self.loss_names = self.custom_loss_names
        # Re-initialize tloss tracker to size 4 to prevent Rank 0 truncation
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Prevents Rank 0 header reset during evaluation phase."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Ensures 'train/kd_loss' mapping exists for the CSV logger."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys