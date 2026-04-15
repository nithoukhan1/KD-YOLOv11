import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer

class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.kd_weight = 0.5 # Adjust the weight of the distillation loss
        
        # 1. Initialize Teacher Model (MedSAM Image Encoder)
        # Note: You will load the actual weights during the Kaggle training script
        from segment_anything import sam_model_registry
        self.teacher = sam_model_registry["vit_b"](checkpoint=overrides.get('teacher_weights', 'medsam_vit_b.pth'))
        self.teacher = self.teacher.image_encoder
        self.teacher.eval()
        
        # Freeze Teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student_features = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        
        # 2. Register a forward hook to capture student feature maps
        # The detect layer is typically the last layer in the model
        detect_layer = model.model[-1] 
        
        def hook_fn(module, input, output):
            # Input to the detect layer is a list of feature maps (P3, P4, P5)
            self.student_features = input
            
        detect_layer.register_forward_pre_hook(hook_fn)
        return model

    def criterion(self, preds, batch):
        # 3. Calculate standard YOLO loss
        loss, loss_items = super().criterion(preds, batch)
        
        # 4. Calculate Knowledge Distillation Loss
        imgs = batch['img']
        with torch.no_grad():
            # MedSAM expects 1024x1024 inputs
            teacher_features = self.teacher(imgs)
            
        # MedSAM outputs a 256-channel feature map
        # We extract the corresponding student feature map (e.g., P3 which is 256 channels)
        student_feat = self.student_features 
        
        # Resize teacher features to match student spatial dimensions if necessary
        if teacher_features.shape[-2:]!= student_feat.shape[-2:]:
            teacher_features = F.interpolate(teacher_features, size=student_feat.shape[-2:], mode='bilinear', align_corners=False)
            
        # Compute Mean Squared Error (MSE) between feature maps
        kd_loss = F.mse_loss(student_feat, teacher_features)
        
        # Combine losses
        total_loss = loss + (self.kd_weight * kd_loss)
        
        # Add KD loss to logging items for monitoring
        loss_items = torch.cat((loss_items, kd_loss.unsqueeze(0)))
        
        return total_loss, loss_items