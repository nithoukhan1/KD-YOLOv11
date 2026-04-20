import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from segment_anything import sam_model_registry
from ultralytics.utils import DEFAULT_CFG

class FeatureDistillLoss(nn.Module):
    def __init__(self, teacher_weights, base_criterion, student_model, alpha=0.5):
        super().__init__()
        # 1. Initialize MedSAM Teacher
        self.teacher = sam_model_registry["vit_b"](checkpoint=teacher_weights)
        self.teacher = self.teacher.image_encoder
        self.teacher.eval()
        self.teacher.half()
        
        # Freeze Teacher weights
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.base_criterion = base_criterion
        self.alpha = alpha
        
        self.student_features = None
        
        def hook_fn(module, input):
            # FIX: Added  to extract the list of feature maps from the PyTorch tuple
            self.student_features = input 
            
        student_model.model[-1].register_forward_pre_hook(hook_fn)

    def forward(self, preds, batch):
        loss, loss_items = self.base_criterion(preds, batch)
        imgs = batch["img"]
        
        if next(self.teacher.parameters()).device!= imgs.device or next(self.teacher.parameters()).dtype!= imgs.dtype:
            self.teacher = self.teacher.to(device=imgs.device, dtype=imgs.dtype)
            
        with torch.no_grad():
            teacher_imgs = imgs.to(next(self.teacher.parameters()).dtype)
            teacher_features = self.teacher(teacher_imgs)
            
        s_feats = self.student_features
        kd_loss = 0.0
        
        for s_feat in s_feats:
            t_feat_resized = F.interpolate(teacher_features, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
            min_c = min(s_feat.shape[1], t_feat_resized.shape[1])
            kd_loss += F.mse_loss(s_feat[:, :min_c,...], t_feat_resized[:, :min_c,...].to(s_feat.dtype))
            
        kd_loss = kd_loss / len(s_feats)
        total_loss = loss + (self.alpha * kd_loss)
        
        return total_loss, loss_items

class KDModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True, teacher_weights=None):
        super().__init__(cfg, ch, nc, verbose)
        self.teacher_weights = teacher_weights

    def init_criterion(self):
        base_criterion = super().init_criterion()
        self.criterion = FeatureDistillLoss(self.teacher_weights, base_criterion, self)
        return self.criterion

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.custom_teacher_weights = overrides.pop('teacher_weights', None)
        if cfg is None:
            cfg = DEFAULT_CFG
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KDModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose, teacher_weights=self.custom_teacher_weights)
        if weights:
            model.load(weights)
        return model