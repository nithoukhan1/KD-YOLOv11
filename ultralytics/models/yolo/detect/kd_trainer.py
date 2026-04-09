import torch
import torch.nn.functional as F
from.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # We remove teacher_model from args and get it from overrides
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
        # 1. Define 4 names so the aggregator knows to expect 4 values
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'kd_loss']
        
        # 2. Extract hyperparameters from the overrides dict
        # Using paths instead of objects makes this DDP-compatible
        teacher_path = self.args.teacher_weights 
        self.kd_weight = self.args.kd_weight if hasattr(self.args, 'kd_weight') else 0.5
        
        # 3. Load Teacher model locally in each rank/process
        from ultralytics import YOLO
        print(f"Rank {getattr(self, 'rank', -1)} loading Teacher: {teacher_path}")
        self.teacher = YOLO(teacher_path).model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def loss(self, batch, preds=None):
        # 1. Calculate standard Student YOLO losses [box, cls, dfl]
        # loss_items is a tensor of length 3
        student_loss, loss_items = super().loss(batch, preds)
        
        # 2. Get predictions from the frozen Teacher model
        # Images are already on the correct GPU device in the 'batch' dictionary
        device = batch['img'].device
        self.teacher.to(device) # Safety synchronization
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # 3. Calculate Distillation Loss (Mean Squared Error)
        kd_loss = 0.0
        # Iterate through multi-scale heads (P3, P4, P5) to match teacher 'thoughts'
        for s_pred, t_pred in zip(preds, teacher_preds):
            kd_loss += F.mse_loss(s_pred, t_pred.detach())
            
        # 4. Combine losses
        weighted_kd = self.kd_weight * kd_loss
        total_loss = student_loss + weighted_kd
        
        # 5. Format loss_items for the DDP logger
        # We append the new KD scalar to the existing [box, cls, dfl] tensor
        kd_scalar = weighted_kd.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, kd_scalar])
        
        return total_loss, loss_items