import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Bypassing the YOLO SyntaxError by popping custom keys
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent with cleaned overrides
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 3. Define 4 names for the Kaggle progress bar
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

        # 4. Rank-safe Teacher loading (Loads separately on each GPU)
        if self.teacher_weights_path:
            from ultralytics import YOLO
            self.teacher = YOLO(self.teacher_weights_path).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def get_validator(self):
        """Ensures custom loss names persist during evaluation."""
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        return super().get_validator()

    def loss(self, batch, preds=None):
        # 1. Calculate standard Student losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get Teacher predictions on the same device as the batch
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate Distillation Loss
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        # 4. Final Weighted Loss
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. CRITICAL: Concatenate for the Logger
        # This makes loss_items length 4, matching self.loss_names
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items