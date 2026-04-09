import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None, teacher_model=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        # 1. Define 4 names so the progress bar shows the new column
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        self.teacher = teacher_model
        self.kd_weight = 0.5  # Hyperparameter for teacher influence
        
        # 2. Freeze teacher model
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def loss(self, batch, preds=None):
        """
        Custom loss combining standard detection loss and Logit-based Knowledge Distillation.
        """
        # 1. Calculate standard Student YOLO losses [box, cls, dfl]
        student_loss, loss_items = super().loss(batch, preds)
        
        if self.teacher is None:
            return student_loss, loss_items

        # 2. Get predictions from the frozen Teacher model
        # Images are already on the correct GPU device in the 'batch' dictionary
        device = batch['img'].device
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'].to(device))
            
        # 3. Calculate Distillation Loss (Mean Squared Error)
        kd_loss = 0.0
        # Iterate through multi-scale heads (P3, P4, P5) to match teacher 'thoughts'
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Combine losses
        weighted_kd = self.kd_weight * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. Format loss_items tensor for the Kaggle progress bar
        # Concatenate the new KD scalar to the existing [box, cls, dfl] tensor
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items