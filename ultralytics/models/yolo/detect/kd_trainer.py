import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# 1. THE DISTILLATION ENGINE
# This wrapper handles precision synchronization (FP16 vs FP32) 
# and ensures the model returns 4 loss components.
class DistillLossWrapper(nn.Module):
    def __init__(self, teacher, student_criterion, kd_weight):
        super().__init__()
        self.teacher = teacher
        self.base = student_criterion
        self.kd_weight = kd_weight

    def forward(self, preds, batch):
        # A. Calculate standard Student YOLO losses [box, cls, dfl]
        loss, loss_items = self.base(preds, batch)
        
        # B. Synchronize Teacher to current GPU and precision (FP16/FP32)
        device = batch['img'].device
        dtype = next(self.teacher.parameters()).dtype
        if next(self.teacher.parameters()).device!= device:
            self.teacher.to(device)
            
        # C. Get Teacher predictions (Soft Targets) for the same batch
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'].to(dtype))
            
        # D. Calculate KD Loss (MSE alignment between Student and Teacher raw logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit.to(torch.float32), t_logit.detach().to(torch.float32))
            
        weighted_kd = self.kd_weight * kd_loss
        
        # E. Combine: Total Loss = Standard + (alpha * KD)
        total_loss = loss + weighted_kd
        
        # F. CRITICAL: Concatenate for logging: [box, cls, dfl] + [kd] = Tensor of size 4
        kd_val = weighted_kd.detach().view(1)
        loss_items = torch.cat([loss_items, kd_val])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Define 4 names EARLY so all ranks allocate the correct header
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # Surgical Pop: Hide custom keys from strict validator to avoid SyntaxError
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def set_model_attributes(self):
        """CRITICAL FIX: Wrap criterion and lock Teacher state across all ranks."""
        # 1. Parent attaches 'args' and initializes default criterion
        super().set_model_attributes()
        
        if self.teacher_weights_path:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Injecting Distillation and Re-Freezing Teacher.")
            
            # 2. Load and freeze Teacher model separately in each GPU process
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            
            # 3. Use getattr to handle DDP wrapping if it exists
            m = getattr(self.model, 'module', self.model)
            
            # 4. Inject the Distillation Wrapper directly into the model object
            m.criterion = DistillLossWrapper(teacher_model, m.criterion, self.kd_weight_val)

            # 5. RE-FREEZE TEACHER: Prevents the parent 'requires_grad=True' warning
            # This ensures teacher weights stay as a static oracle
            for param in m.criterion.teacher.parameters():
                param.requires_grad = False

        # 6. FORCE tracking tensors to size 4 to prevent empty logs/CSV values
        self.loss_names = self.custom_loss_names
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Ensures custom loss names survive the Rank 0 header reset during evaluation."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Override dictionary mapping to ensure 'train/kd_loss' reaches results.csv."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys