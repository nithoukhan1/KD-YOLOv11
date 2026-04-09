import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Pop custom args before parent validation to avoid SyntaxError
        custom_params = dict(overrides or {})
        teacher_path = custom_params.pop("teacher_weights", None)
        kd_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent with cleaned overrides
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 3. Re-attach and define tracking names
        self.args.teacher_weights = teacher_path
        self.args.kd_weight = kd_val
        self.kd_weight_val = kd_val
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

        # 4. Rank-safe Teacher loading
        if self.args.teacher_weights:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Loading Teacher from {self.args.teacher_weights}")
            self.teacher = YOLO(self.args.teacher_weights).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def get_validator(self):
        """Ensure loss_names are preserved when the validator starts."""
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        return super().get_validator()

    def loss(self, batch, preds=None):
        """Combined loss with 4-element output tensor for logging."""
        # 1. Calculate standard Student losses (box, cls, dfl)
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get Teacher predictions
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE)
        kd_loss = 0.0
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Final Weighted Loss
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. CONCATENATE: [box, cls, dfl] + [weighted_kd] = Tensor of size 4
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items