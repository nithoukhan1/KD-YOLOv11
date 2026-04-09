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
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get predictions from the frozen Teacher model
        with torch.no_grad():
            # Ensure teacher and images are on the same device
            device = next(self.model.parameters()).device
            images = batch['img'].to(device)
            teacher_preds = self.teacher(images)
            
        # 3. Calculate Distillation Loss (MSE between output logits)
        # We iterate through the multi-scale prediction tensors (P3, P4, P5)
        kd_loss = 0.0
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Combine standard detection loss with distillation loss
        # Total Loss = L_Student + (alpha * L_KD)
        total_loss = student_loss + (self.kd_weight * kd_loss)
        
        # Update loss items for logging
        loss_items = total_loss.detach()
        
        return total_loss, loss_items