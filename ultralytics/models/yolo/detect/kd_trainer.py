import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None, teacher_model=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        # Add this line to show 'kd_loss' in the training table
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        self.teacher = teacher_model
        self.kd_weight = 0.5  # Hyperparameter: influence of the teacher on the student
        
        # Freeze the teacher model so it does not train
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def loss(self, batch, preds=None):
        """
        Custom loss function combining standard detection loss and 
        Logit-based Knowledge Distillation.
        """
        # 1. Calculate standard YOLO loss for the Student
        # student_loss is a scalar for backprop; loss_items is a tensor [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get predictions from the frozen Teacher model
        # Ensure teacher is on the same device as the training batch
        device = batch['img'].device
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'].to(device))
            
        # 3. Calculate Distillation Loss (MSE between output logits)
        kd_loss = 0.0
        # Iterate through the multi-scale prediction tensors (P3, P4, P5)
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Combine standard detection loss with weighted distillation loss
        weighted_kd = self.kd_weight * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. CRITICAL: Format loss_items for the progress bar
        # We append the weighted KD loss to the existing [box, cls, dfl] tensor
        # This makes it a tensor of size 4, matching your self.loss_names
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items