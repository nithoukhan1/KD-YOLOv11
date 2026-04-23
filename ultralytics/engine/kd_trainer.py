"""
kd_trainer.py
=============
Place at:  KD-YOLOv11/ultralytics/engine/kd_trainer.py

Teacher : YOLOv11L-C3k2-SC-DySample-ResEMA  (frozen)
Student : YOLOv11S-C3k2-SC-DySample-ResEMA  (trainable — your model)
KD Loss : Channel-Wise Distillation (CWD) at ResEMA neck outputs
          YAML layer indices 14, 18, 22, 26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CWD Loss
# ─────────────────────────────────────────────────────────────────────────────
class CWDLoss(nn.Module):
    """
    Channel-Wise Distillation.
    Treats each channel's H×W map as a probability distribution,
    then minimises KL divergence between teacher and student per channel.
    Reference: Shu et al., ICCV 2021.
    """
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student: torch.Tensor,
                teacher: torch.Tensor) -> torch.Tensor:
        # student / teacher : (B, C, H, W)  channel-aligned
        assert student.shape == teacher.shape, (
            f"CWD shape mismatch: {student.shape} vs {teacher.shape}")
        B, C, H, W = student.shape
        s = F.softmax(student.view(B, C, -1) / self.T, dim=-1)
        t = F.softmax(teacher.view(B, C, -1) / self.T, dim=-1)
        return F.kl_div(s.log(), t, reduction="batchmean")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel Adapter  (1×1 conv  student_ch → teacher_ch)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out",
                                nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature Capture (forward hooks)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureCapture:
    def __init__(self, model: nn.Module, layer_indices: list):
        self.features: dict = {}
        self._hooks: list = []
        raw = de_parallel(model)
        for idx in layer_indices:
            h = raw.model[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)
        LOGGER.info(colorstr("KD hooks: ") +
                    f"attached at layers {layer_indices}")

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            self.features[idx] = out
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KD Detection Trainer
# ─────────────────────────────────────────────────────────────────────────────
_KD_LAYERS = [14, 18, 22, 26]   # ResEMA outputs in your neck YAML


class KDDetectionTrainer(DetectionTrainer):
    """
    Adds CWD knowledge distillation to DetectionTrainer.

    Extra keys in overrides dict:
        teacher_weights  str    path to teacher best.pt  (required)
        kd_alpha         float  KD loss weight           (default 1.0)
        kd_temperature   float  CWD temperature          (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        ov = overrides or {}
        self.teacher_weights = ov.get("teacher_weights", None)
        self.kd_alpha        = float(ov.get("kd_alpha", 1.0))
        self.kd_temp         = float(ov.get("kd_temperature", 4.0))
        # runtime state
        self.teacher      = None
        self._s_hooks     = None
        self._t_hooks     = None
        self._adapters    = {}
        self._adapters_ok = False
        self._cwd         = None
        self._step        = 0

    # ── A: build student, then load teacher & inject KD ──────────────────────
    def setup_model(self):
        super().setup_model()
        if not self.teacher_weights:
            LOGGER.warning(colorstr("KD WARNING: ") +
                           "teacher_weights not set — KD disabled.")
            return
        self._load_teacher()
        self._attach_hooks()
        self._cwd = CWDLoss(temperature=self.kd_temp)
        self._inject_kd_loss()

    # ── B: load & freeze teacher ─────────────────────────────────────────────
    def _load_teacher(self):
        path = Path(self.teacher_weights)
        assert path.exists(), f"Teacher weights not found: {path}"
        LOGGER.info(colorstr("KD: ") + f"Loading teacher from {path}")

        from ultralytics import YOLO
        self.teacher = YOLO(str(path)).model   # raw DetectionModel

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        device = next(self.model.parameters()).device
        self.teacher = self.teacher.to(device)
        LOGGER.info(colorstr("KD: ") +
                    f"Teacher frozen on {device}  "
                    f"alpha={self.kd_alpha}  T={self.kd_temp}")

    # ── C: attach hooks ───────────────────────────────────────────────────────
    def _attach_hooks(self):
        self._s_hooks = FeatureCapture(self.model,   _KD_LAYERS)
        self._t_hooks = FeatureCapture(self.teacher, _KD_LAYERS)

    # ── D: build adapters (lazy, first forward) ───────────────────────────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        device = next(self.model.parameters()).device
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
            else:
                LOGGER.info(colorstr("KD adapter: ") +
                            f"layer {idx}: {sc}→{tc} ch")
                self._adapters[idx] = ChannelAdapter(sc, tc).to(device)
        self._adapters_ok = True

    # ── E: compute CWD from captured features ────────────────────────────────
    def _compute_kd_loss(self) -> torch.Tensor:
        sf = self._s_hooks.features
        tf = self._t_hooks.features
        self._maybe_build_adapters(sf, tf)

        kd = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for idx in _KD_LAYERS:
            if idx not in sf or idx not in tf:
                continue
            s, t = sf[idx], tf[idx]
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:],
                                  mode="bilinear", align_corners=False)
            s = self._adapters[idx](s)
            kd = kd + self._cwd(s, t)

        self._s_hooks.clear()
        self._t_hooks.clear()
        return kd / len(_KD_LAYERS)

    # ── F: inject KD into model.loss (shadows BaseModel.loss) ────────────────
    def _inject_kd_loss(self):
        raw_model     = de_parallel(self.model)
        original_loss = raw_model.loss   # original bound method
        trainer       = self

        def kd_loss_fn(batch, preds=None):
            # 1. Standard detection loss  (student forward fires s_hooks)
            det_loss, det_items = original_loss(batch, preds)

            # 2. Teacher forward  (t_hooks fire)
            with torch.no_grad():
                trainer.teacher(batch["img"])

            # 3. CWD
            kd = trainer._compute_kd_loss()
            total = det_loss + trainer.kd_alpha * kd

            # Periodic log
            trainer._step += 1
            if trainer._step % 50 == 0:
                LOGGER.debug(
                    colorstr("KD: ") +
                    f"step={trainer._step}  "
                    f"det={det_loss.item():.4f}  "
                    f"kd={kd.item():.5f}  "
                    f"total={total.item():.4f}"
                )
            return total, det_items

        # Instance-level override shadows the class method
        raw_model.loss = kd_loss_fn
        LOGGER.info(colorstr("KD: ") + "loss injection complete ✓")

    # ── G: keep teacher on correct GPU after DDP ──────────────────────────────
    def _setup_ddp(self, world_size):
        super()._setup_ddp(world_size)
        if self.teacher is not None:
            dev = next(self.model.parameters()).device
            self.teacher = self.teacher.to(dev)

    # ── H: cleanup on finish ──────────────────────────────────────────────────
    def final_eval(self):
        if self._s_hooks:
            self._s_hooks.remove()
        if self._t_hooks:
            self._t_hooks.remove()
        super().final_eval()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry-point  (called from Kaggle notebook)
# ─────────────────────────────────────────────────────────────────────────────
def train_kd(
    student_yaml:     str,
    teacher_weights:  str,
    data_yaml:        str,
    epochs:           int   = 150,
    imgsz:            int   = 1024,
    batch:            int   = 16,
    device:           str   = "0,1",
    kd_alpha:         float = 1.0,
    kd_temperature:   float = 4.0,
    project:          str   = "KD-YOLOv11",
    name:             str   = "kd_run",
    resume:           bool  = False,
):
    """
    Launch KD training from a Kaggle notebook.

    IMPORTANT — batch should be 16 not 32.
    Teacher + student both hold activations in GPU RAM during the forward pass.
    Each T4 has ~15 GB. With batch=16 at 1024px, both models fit comfortably.
    """
    overrides = dict(
        model           = student_yaml,
        data            = data_yaml,
        epochs          = epochs,
        imgsz           = imgsz,
        batch           = batch,
        device          = device,
        optimizer       = "SGD",
        cos_lr          = True,
        cls             = 2.5,
        copy_paste      = 0.5,
        mixup           = 0.2,
        close_mosaic    = 10,
        patience        = 75,
        project         = project,
        name            = name,
        resume          = resume,
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer