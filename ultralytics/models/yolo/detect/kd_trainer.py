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
        
        # 2. Synchronize teacher to current device
        device = batch['img'].device
        if next(self.teacher.parameters()).device!= device:
            self.teacher.to(device)
            
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE alignment between Student/Teacher raw logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # 5. CONCATENATE: Appending KD to the logging tensor [box, cls, dfl, kd]
        # This fixes the "column shifting" by ensuring the tensor size matches the header
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # Surgical Pop: Bypass syntax error by hiding custom keys from the strict validator
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model and manually initialize criterion to allow wrapping."""
        model = super().get_model(cfg, weights, verbose)
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            print(f"🚀 Rank {getattr(self, 'rank', -1)}: Injected Distillation Wrapper.")
            
            # Load Expert Teacher
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # FIX: Manually initialize criterion if it hasn't been already
            if not hasattr(model, 'criterion'):
                model.criterion = model.init_criterion()
            
            # Inject the Distillation Wrapper
            model.criterion = DistillLossWrapper(teacher_model, model.criterion, self.kd_weight_val)
            
        return model

    def set_model_attributes(self):
        """Force headers and trackers to support 4 values across all ranks."""
        super().set_model_attributes()
        self.loss_names = self.custom_loss_names
        # Re-initialize tloss tracker to size 4 so values aren't discarded
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Ensure 'train/kd_loss' mapping exists for the CSV logger."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys