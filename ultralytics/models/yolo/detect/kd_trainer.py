import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Bypassing strict validation by popping custom keys before parent init
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent with cleaned overrides
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 3. Define the 4 loss names for the logging aggregator
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

        # 4. Rank-safe Teacher loading: Loads separately on each GPU to avoid pickling errors
        if self.teacher_weights_path:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Loading Expert Teacher from {self.teacher_weights_path}")
            
            self.teacher = YOLO(self.teacher_weights_path).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def get_validator(self):
        """CRITICAL FIX: Overrides header reset in multi-GPU training."""
        validator = super().get_validator()
        # Force the loss names to persist with 4 entries
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        return validator

    def loss(self, batch, preds=None):
        """Combined loss with 4-element output tensor for proper progress bar mapping."""
        # Calculate standard Student YOLO losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # Synchronize teacher/images on current GPU device
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # Calculate KD Loss (MSE between Student and Teacher raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # CONCATENATE: Join [box, cls, dfl] + [kd] to match self.loss_names length (4)
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items