import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # FIX 1: Define 4 names EARLY so the logger allocates the correct CSV header
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # Bypassing strict validation via pop
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # Initialize parent with cleaned overrides
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

        # FIX 2: RE-SIZE THE RUNNING MEAN TRACKER (tloss)
        # This allocates memory for the 4th value so it's not discarded by Rank 0
        self.loss_names = self.custom_loss_names
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

        # Rank-safe Teacher loading
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
        # FIX 3: PROTECT AGAINST RANK-0 HEADER RESET
        # Prevents the framework from defaulting back to 3 columns during evaluation
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        # FIX 4: CSV LOGGING DICTIONARY MAPPING
        # Ensures the CSV writer finds 'train/kd_loss' in every row
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    def loss(self, batch, preds=None):
        """Custom loss combining standard detection and scale-logit distillation."""
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        device = batch['img'].device
        self.teacher.to(device)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        kd_loss = 0.0
        # Student raw outputs matched against teacher soft targets
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        weighted_kd = self.kd_weight_val * kd_loss
        total_loss = student_loss + weighted_kd
        
        # Concatenate: [box, cls, dfl] + [kd] = Tensor of size 4
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items