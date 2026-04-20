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
        
        # Freeze Teacher weights
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.base_criterion = base_criterion
        self.alpha = alpha
        
        # 2. Register a forward hook to capture student feature maps
        self.student_features = None
        
        # Correct pre-hook signature (only takes module and input)
        def hook_fn(module, input):
            self.student_features = input 
            
        # Attach hook to the YOLO Detection Head
        student_model.model[-1].register_forward_pre_hook(hook_fn)

    def forward(self, preds, batch):
        # 3. Calculate standard YOLO loss (Box, Cls, DFL)
        loss, loss_items = self.base_criterion(preds, batch)
        
        imgs = batch["img"]
        img_device = imgs.device # Get the specific GPU device of the current batch
        
        with torch.no_grad():
            # FIX 1: Ensure the teacher model is on the same GPU as the images
            if next(self.teacher.parameters()).device!= img_device:
                self.teacher.to(img_device)
                
            # FIX 2: Dynamically cast the input images to match the MedSAM teacher's data type
            teacher_imgs = imgs.to(next(self.teacher.parameters()).dtype)
            
            # Extract MedSAM embeddings using the casted images
            teacher_features = self.teacher(teacher_imgs)
            
        s_feats = self.student_features
        kd_loss = 0.0
        
        # 4. Calculate KD Loss across multi-scale features
        for s_feat in s_feats:
            # Resize teacher features to match student spatial dimensions
            t_feat_resized = F.interpolate(teacher_features, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # Match channel dimensions by taking the minimum available channels
            min_c = min(s_feat.shape[1], t_feat_resized.shape[1])
            
            # Ensure the resized teacher features match the student's AMP data type
            kd_loss += F.mse_loss(s_feat[:, :min_c,...], t_feat_resized[:, :min_c,...].to(s_feat.dtype))
            
        kd_loss = kd_loss / len(s_feats)
        
        # Combine standard loss with scaled KD loss
        total_loss = loss + (self.alpha * kd_loss)
        
        return total_loss, loss_items


# Custom Model that injects the KD Loss
class KDModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True, teacher_weights=None):
        super().__init__(cfg, ch, nc, verbose)
        self.teacher_weights = teacher_weights

    def init_criterion(self):
        # Initialize standard YOLO loss
        base_criterion = super().init_criterion()
        # Wrap it with our Distillation Loss
        self.criterion = FeatureDistillLoss(self.teacher_weights, base_criterion, self)
        return self.criterion


# Custom Trainer that loads the KD Model
class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        
        # 1. Extract our custom argument to bypass YOLO's strict dictionary alignment
        self.custom_teacher_weights = overrides.pop('teacher_weights', None)
        
        # 2. Safety check: ensure cfg is never None
        if cfg is None:
            cfg = DEFAULT_CFG
            
        # 3. Initialize the standard YOLO trainer with the remaining official arguments
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Pass the safely extracted weights into our KDModel
        model = KDModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose, teacher_weights=self.custom_teacher_weights)
        if weights:
            model.load(weights)
        return model