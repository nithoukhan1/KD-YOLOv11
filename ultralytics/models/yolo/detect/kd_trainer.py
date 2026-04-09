import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Surgical Pop: Remove custom keys before parent validation
        custom_params = dict(overrides or {})
        self.teacher_weights = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent with cleaned overrides to avoid SyntaxError
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 3. Add 'kd_loss' to the tracked names for the Kaggle progress bar
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

        # 4. Rank-safe Teacher loading: each GPU rank loads its own copy
        if self.teacher_weights:
            from ultralytics import YOLO
            # Use the rank property to log only on the main process
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Loading Expert Teacher from {self.teacher_weights}")
            
            # Load the model object from the weights path
            self.teacher = YOLO(self.teacher_weights).model
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher = None

    def loss(self, batch, preds=None):
        """
        Custom loss combining standard YOLO detection loss and Logit-based Distillation.
        """
        # 1. Calculate standard Student losses [box, cls, dfl]
        # student_loss is the backward scalar; loss_items is a tensor of size 3
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get Teacher predictions on the same device as the batch
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate Knowledge Distillation Loss (alignment of raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        # 4. Combine standard detection loss with weighted distillation
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. Concatenate for logging: [box, cls, dfl] + [kd] = tensor of size 4
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items