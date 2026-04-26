"""
kd_trainer.py
=============
Knowledge Distillation trainer for YOLOv11-C3k2-SC-DySample-ResEMA.

Place at:  KD-YOLOv11/ultralytics/engine/kd_trainer.py

Teacher : YOLOv11L-C3k2-SC-DySample-ResEMA  (frozen)
Student : YOLOv11S-C3k2-SC-DySample-ResEMA  (trainable)
KD Loss : Channel-Wise Distillation (CWD) at ResEMA neck outputs
          YAML layer indices: 14, 18, 22, 26

Fix history:
    v1 — de_parallel → unwrap_model
    v2 — Pop KD keys before super().__init__() (cfg trainer validation)
    v3 — Don't pass cfg=None to super (DEFAULT_CFG crash)
    v4 — Remove world_size from _setup_ddp (fork signature mismatch)
    v5 — Save KD keys into self.args so DDP subprocess can read them
    v6 — Override get_validator() to strip KD keys before validator's
         get_cfg() runs check_dict_alignment(), then restore them.
         Also use self.device (not model params) to place teacher on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

# KD-specific keys that must be hidden from Ultralytics cfg validation
_KD_KEYS = ("teacher_weights", "kd_alpha", "kd_temperature")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CWD Loss
# ─────────────────────────────────────────────────────────────────────────────
class CWDLoss(nn.Module):
    """
    Channel-Wise Distillation loss.
    Treats each channel's H×W map as a probability distribution (softmax at T)
    and minimises KL divergence between teacher and student per channel.
    Shu et al., ICCV 2021.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        assert student.shape == teacher.shape, (
            f"CWD shape mismatch: student {student.shape} vs teacher {teacher.shape}"
        )
        B, C, H, W = student.shape
        s = F.softmax(student.view(B, C, -1) / self.T, dim=-1)
        t = F.softmax(teacher.view(B, C, -1) / self.T, dim=-1)
        return F.kl_div(s.log(), t, reduction="batchmean")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel Adapter
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """1×1 conv: student_ch → teacher_ch. Built lazily from actual shapes."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature Capture (forward hooks)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureCapture:
    """Attaches forward hooks to YOLO layer indices, stores output tensors."""

    def __init__(self, model: nn.Module, layer_indices: list):
        self.features: dict = {}
        self._hooks: list = []
        raw = unwrap_model(model)
        for idx in layer_indices:
            self._hooks.append(
                raw.model[idx].register_forward_hook(self._make_hook(idx))
            )
        LOGGER.info(colorstr("KD hooks: ") + f"attached at layers {layer_indices}")

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
_KD_LAYERS = [14, 18, 22, 26]   # ResEMA neck outputs in your YAML


class KDDetectionTrainer(DetectionTrainer):
    """
    Extends DetectionTrainer with CWD Knowledge Distillation.

    The fundamental DDP challenge:
        Ultralytics DDP spawns a subprocess that rebuilds the trainer from
        vars(self.args). Our KD params must live in self.args so they survive
        into the subprocess. BUT self.args is also passed to the Validator,
        which runs check_dict_alignment() and rejects unknown keys.

    Solution:
        - Store KD params in self.args (needed for DDP subprocess)
        - Override get_validator() to strip KD params from self.args before
          the Validator is created, then restore them immediately after.

    Extra keys for the overrides dict:
        teacher_weights  str    Path to teacher best.pt   (required)
        kd_alpha         float  KD loss weight             (default 1.0)
        kd_temperature   float  CWD temperature            (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):

        # ── 1. Extract KD keys from overrides (normal call path) ─────────────
        ov = dict(overrides) if overrides else {}
        tw    = ov.pop("teacher_weights", None)
        alpha = float(ov.pop("kd_alpha", 1.0))
        temp  = float(ov.pop("kd_temperature", 4.0))

        # ── 2. Extract KD keys from cfg dict (DDP subprocess call path) ───────
        # DDP rebuilds trainer as: KDDetectionTrainer(cfg=vars(self.args))
        # where overrides=None, so KD params arrive via cfg dict.
        if isinstance(cfg, dict):
            tw    = cfg.pop("teacher_weights", tw)
            alpha = float(cfg.pop("kd_alpha", alpha))
            temp  = float(cfg.pop("kd_temperature", temp))

        # ── 3. Call super without cfg=None (would override DEFAULT_CFG) ───────
        if cfg is not None:
            super().__init__(cfg=cfg, overrides=ov, _callbacks=_callbacks)
        else:
            super().__init__(overrides=ov, _callbacks=_callbacks)

        # ── 4. Save KD params into self.args so DDP subprocess sees them ──────
        # self.args is a SimpleNamespace; vars(self.args) feeds the DDP file.
        self.args.teacher_weights = tw
        self.args.kd_alpha        = alpha
        self.args.kd_temperature  = temp

        # Runtime state (populated in setup_model)
        self.teacher      = None
        self._s_hooks     = None
        self._t_hooks     = None
        self._adapters    = {}
        self._adapters_ok = False
        self._cwd         = None
        self._kd_step     = 0

    # ── get_validator: strip KD keys before Ultralytics cfg validation ────────
    def get_validator(self):
        """
        Ultralytics passes self.args to DetectionValidator, which calls
        get_cfg(overrides=args) → check_dict_alignment() → rejects our
        custom keys with SyntaxError.

        Fix: temporarily remove KD keys from self.args, create the validator
        (which now sees a clean args namespace), then restore the keys so
        DDP subprocess generation still works correctly.
        """
        # Save and remove KD keys
        saved = {}
        for key in _KD_KEYS:
            if hasattr(self.args, key):
                saved[key] = getattr(self.args, key)
                delattr(self.args, key)

        try:
            validator = super().get_validator()
        finally:
            # Always restore — even if validator creation raises an exception
            for key, val in saved.items():
                setattr(self.args, key, val)

        return validator

    # ── A: Build student, load teacher ───────────────────────────────────────
    def setup_model(self):
        super().setup_model()

        # Read from self.args (always correct in both normal and DDP paths)
        teacher_weights = getattr(self.args, "teacher_weights", None)
        self._kd_alpha  = float(getattr(self.args, "kd_alpha", 1.0))
        self._kd_temp   = float(getattr(self.args, "kd_temperature", 4.0))

        if not teacher_weights:
            LOGGER.warning(
                colorstr("KD WARNING: ") +
                "teacher_weights not set — KD loss DISABLED."
            )
            return

        self._kd_teacher_weights = teacher_weights
        self._load_teacher()
        self._attach_hooks()
        self._cwd = CWDLoss(temperature=self._kd_temp)
        self._inject_kd_loss()

    # ── B: Load and freeze teacher ────────────────────────────────────────────
    def _load_teacher(self):
        path = Path(self._kd_teacher_weights)
        assert path.exists(), f"Teacher weights not found: {path}"
        LOGGER.info(colorstr("KD: ") + f"Loading teacher from {path}")

        from ultralytics import YOLO
        self.teacher = YOLO(str(path)).model

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Use self.device (set by Ultralytics before setup_model is called)
        # This ensures teacher goes to the correct GPU in DDP mode.
        device = self.device
        self.teacher = self.teacher.to(device)
        LOGGER.info(
            colorstr("KD: ") +
            f"Teacher frozen on {device}.  "
            f"alpha={self._kd_alpha}  temperature={self._kd_temp}"
        )

    # ── C: Attach hooks ───────────────────────────────────────────────────────
    def _attach_hooks(self):
        self._s_hooks = FeatureCapture(self.model,   _KD_LAYERS)
        self._t_hooks = FeatureCapture(self.teacher, _KD_LAYERS)

    # ── D: Build channel adapters (lazy, from first-forward shapes) ───────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        device = self.device
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
                LOGGER.info(colorstr("KD adapter: ") + f"layer {idx}: {sc} ch — Identity")
            else:
                self._adapters[idx] = ChannelAdapter(sc, tc).to(device)
                LOGGER.info(colorstr("KD adapter: ") + f"layer {idx}: {sc}→{tc} — Conv built")
        self._adapters_ok = True

    # ── E: Compute CWD ────────────────────────────────────────────────────────
    def _compute_kd_loss(self) -> torch.Tensor:
        sf = self._s_hooks.features
        tf = self._t_hooks.features
        self._maybe_build_adapters(sf, tf)

        device = self.device
        kd = torch.tensor(0.0, device=device)

        for idx in _KD_LAYERS:
            if idx not in sf or idx not in tf:
                continue
            s, t = sf[idx], tf[idx]
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            s = self._adapters[idx](s)
            kd = kd + self._cwd(s, t)

        self._s_hooks.clear()
        self._t_hooks.clear()
        return kd / len(_KD_LAYERS)

    # ── F: Inject KD into model.loss ──────────────────────────────────────────
    def _inject_kd_loss(self):
        raw_model     = unwrap_model(self.model)
        original_loss = raw_model.loss
        trainer       = self

        def kd_loss_fn(batch, preds=None):
            det_loss, det_items = original_loss(batch, preds)
            with torch.no_grad():
                trainer.teacher(batch["img"])
            kd = trainer._compute_kd_loss()
            total = det_loss + trainer._kd_alpha * kd
            return total, det_items

        raw_model.loss = kd_loss_fn
        LOGGER.info(colorstr("KD: ") + "Loss injection complete ✓")

    # ── G: Keep teacher on correct GPU after DDP ──────────────────────────────
    def _setup_ddp(self):
        super()._setup_ddp()
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)

    # ── H: Clean up hooks ─────────────────────────────────────────────────────
    def final_eval(self):
        if self._s_hooks:
            self._s_hooks.remove()
        if self._t_hooks:
            self._t_hooks.remove()
        super().final_eval()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry-point
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
) -> KDDetectionTrainer:
    """
    Launch KD training from a Kaggle notebook.
    batch=16 not 32 — teacher + student share GPU VRAM each forward pass.
    """
    overrides = dict(
        model        = student_yaml,
        data         = data_yaml,
        epochs       = epochs,
        imgsz        = imgsz,
        batch        = batch,
        device       = device,
        optimizer    = "SGD",
        cos_lr       = True,
        cls          = 2.5,
        copy_paste   = 0.5,
        mixup        = 0.2,
        close_mosaic = 10,
        patience     = 75,
        project      = project,
        name         = name,
        resume       = resume,
        # KD keys — stripped from overrides in __init__ before Ultralytics
        # cfg validation, saved to self.args for DDP, stripped again from
        # self.args before validator creation in get_validator().
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer