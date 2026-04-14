import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Define custom names EARLY to influence setup
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # 2. Surgical Pop: Bypass initial strict argument validation
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 3. Initialize parent trainer with cleaned overrides
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 4. DEEP INJECTION: Force names and re-size the running mean tracker (tloss)
        # This allocates memory for the 4th value so Rank 0 doesn't discard it
        self.loss_names = self.custom_loss_names
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

        # 5. Rank-safe Teacher loading (DDP compatible)
        if self.teacher_weights_path:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Loading Teacher from {self.teacher_weights_path}")
            self.teacher = YOLO(self.teacher_weights_path).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def get_validator(self):
        """Override to ensure loss_names survive Rank 0 evaluation re-initialization."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Override to ensure the result dictionary contains all 4 keys for the CSV."""
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is not None:
            # Move to CPU and round for standard Ultralytics logging format
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    def loss(self, batch, preds=None):
        """Combined loss with 4-element output tensor for progress bar mapping."""
        # Calculate standard Student YOLO losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # Synchronize teacher and batch on the same GPU rank
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # Logit-based Knowledge Distillation (MSE alignment)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # CONCATENATE: [box, cls, dfl] + [kd] = Tensor of size 4
        # This MUST be size 4 to match self.loss_names length
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items