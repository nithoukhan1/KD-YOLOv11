import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Surgical Pop: Intercept custom keys before validation to avoid SyntaxError
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent with cleaned overrides to bypass strict check
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # 3. Define the 4 loss names for the Kaggle progress bar aggregator
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']

        # 4. Rank-safe Teacher loading: Each rank loads its own copy (DDP Safe)
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
        """Override to ensure kd_loss column survives Rank 0 re-initialization."""
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        return super().get_validator()

    def loss(self, batch, preds=None):
        """Combined loss with 4-element output tensor for proper progress bar mapping."""
        # 1. Calculate standard Student YOLO losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Synchronize teacher and images on the current GPU device
        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE alignment between Student and Teacher raw scale logits)
        kd_loss = 0.0
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        # 4. Final Weighted Loss: L_total = L_student + (alpha * L_kd)
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. CONCATENATE: [box, cls, dfl] + [weighted_kd] = Tensor of size 4
        # This MUST match the length of self.loss_names exactly for the progress bar to show
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items