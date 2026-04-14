import torch
import torch.nn as nn
import torch.nn.functional as F
from.train import DetectionTrainer

# A wrapper to inject distillation into the model's internal loss engine
class DistillLossWrapper(nn.Module):
    def __init__(self, teacher, student_criterion, kd_weight):
        super().__init__()
        self.teacher = teacher
        self.base = student_criterion
        self.kd_weight = kd_weight

    def forward(self, preds, batch):
        # 1. Calculate standard Student YOLO losses [box, cls, dfl]
        loss, loss_items = self.base(preds, batch)
        
        # 2. Get Teacher predictions for the same batch
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate KD Loss (MSE alignment between scale logits)
        kd_loss = 0.0
        # Student raw outputs matched against teacher scale-wise targets
        for s_logit, t_logit in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_logit, t_logit.detach())
            
        weighted_kd = self.kd_weight * kd_loss
        
        # 4. Combine: Total Loss = L_{student} + (\alpha \times L_{kd})
        total_loss = loss + weighted_kd
        
        # 5. Concatenate for logging: [box, cls, dfl] + [kd] = Tensor of size 4
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 1. Define custom names and pop keys before parent validation
        self.custom_loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        custom_params = dict(overrides or {})
        self.teacher_weights_path = custom_params.pop("teacher_weights", None)
        self.kd_weight_val = float(custom_params.pop("kd_weight", 0.5))

        # 2. Initialize parent (self.model is still a string here)
        super().__init__(cfg=cfg, overrides=custom_params, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the model and inject the Distillation Wrapper."""
        # A. Build the actual Student model object
        model = super().get_model(cfg, weights, verbose)
        
        # B. Load the Expert Teacher (Loads separately on each GPU rank)
        if self.teacher_weights_path:
            from ultralytics import YOLO
            if getattr(self, 'rank', -1) in (-1, 0):
                print(f"🚀 Rank {getattr(self, 'rank', -1)}: Injecting Distillation into Model Criterion")
            
            teacher_model = YOLO(self.teacher_weights_path).model
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            # C. Inject the wrapper into the model object
            model.criterion = DistillLossWrapper(teacher_model, model.criterion, self.kd_weight_val)
            
        return model

    def _setup_train(self, *args, **kwargs):
        """Ensure tracking tensors are resized for the 4th loss component."""
        super()._setup_train(*args, **kwargs)
        self.loss_names = self.custom_loss_names
        # Resize tloss tracker so Rank 0 records the distillation value
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

    def get_validator(self):
        """Ensures custom loss names survive the Rank 0 header reset."""
        validator = super().get_validator()
        self.loss_names = self.custom_loss_names
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Override to ensure the result dictionary maps all 4 keys correctly."""
        keys = [f"{prefix}/{name}" for name in self.custom_loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys