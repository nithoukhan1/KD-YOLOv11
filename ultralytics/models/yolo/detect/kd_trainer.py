import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Prevent SyntaxError by popping custom args before parent validation
        ovr = dict(overrides or {})
        teacher_path = ovr.pop("teacher_weights", None)
        kd_val = ovr.pop("kd_weight", 0.5)

        # 2. Initialize base trainer with cleaned overrides
        super().__init__(cfg=cfg, overrides=ovr, _callbacks=_callbacks)

        # 3. Manually re-attach them to self.args so they persist for DDP and logging
        self.args.teacher_weights = teacher_path
        self.args.kd_weight = kd_val
        
        # 4. Define 4 loss names for the progress bar display
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        self.kd_weight = float(self.args.kd_weight)

        # 5. Load Teacher model locally in each process/rank (DDP safe)
        if self.args.teacher_weights:
            from ultralytics import YOLO
            # Rank-safe loading to prevent serialization/pickle issues
            self.teacher = YOLO(self.args.teacher_weights).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def loss(self, batch, preds=None):
        """Custom loss combining standard YOLO loss and Knowledge Distillation."""
        # Calculate standard Student YOLO losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # Ensure teacher and images are on the correct GPU device
        device = batch['img'].device
        self.teacher.to(device)
        
        with torch.no_grad():
            # Pass same augmented batch to the teacher
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate Distillation Loss (MSE between raw output logits)
        kd_loss = 0.0
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Combine standard detection loss with weighted distillation loss
        weighted_kd = self.kd_weight * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. Concatenate weighted KD loss to standard losses for logging [box, cls, dfl, kd]
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items