"""
kd_trainer.py  —  Final version with all fixes applied
=======================================================
Place at:  KD-YOLOv11/ultralytics/engine/kd_trainer.py

Teacher : YOLOv11L-C3k2-SC-DySample-ResEMA  (frozen)
Student : YOLOv11S-C3k2-SC-DySample-ResEMA  (trainable)
KD Loss : Channel-Wise Distillation (CWD) at ResEMA neck outputs
          YAML layer indices: 14, 18, 22, 26

Fix history:
    v1 — de_parallel → unwrap_model (fork removed de_parallel)
    v2 — Pop KD keys from overrides before super().__init__()
         (Ultralytics check_dict_alignment rejects unknown keys)
    v3 — Don't pass cfg=None to super (overrides DEFAULT_CFG → crash)
    v4 — _setup_ddp() has no world_size arg in this fork
    v5 — Save KD keys into self.args so DDP subprocess reads them
    v6 — Override get_validator() to strip KD keys before validator
         cfg check, then restore them (SyntaxError fix)
    v7 — Remove debug logger from kd_loss_fn
         (det_loss is a 3-element tensor, not a scalar)
    v8 — Cast batch["img"] to teacher dtype before teacher forward
         (AMP validation sends FP16 but teacher weights are FP32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model   # not de_parallel

# KD-specific keys — must never reach Ultralytics cfg validation
_KD_KEYS = ("teacher_weights", "kd_alpha", "kd_temperature")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CWD Loss
# ─────────────────────────────────────────────────────────────────────────────
class CWDLoss(nn.Module):
    """
    Channel-Wise Distillation loss.

    For each channel, treats the H×W spatial map as a probability distribution
    (softmax at temperature T) and minimises KL divergence between the teacher
    and student distributions channel by channel.

    Reference: Shu et al. "Channel-wise Knowledge Distillation for Dense
    Prediction", ICCV 2021.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student: torch.Tensor,
                teacher: torch.Tensor) -> torch.Tensor:
        assert student.shape == teacher.shape, (
            f"CWD shape mismatch: student {student.shape} "
            f"vs teacher {teacher.shape}"
        )
        B, C, H, W = student.shape
        s = F.softmax(student.view(B, C, -1) / self.T, dim=-1)
        t = F.softmax(teacher.view(B, C, -1) / self.T, dim=-1)
        return F.kl_div(s.log(), t, reduction="batchmean")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel Adapter
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """
    Lightweight 1×1 conv projecting student channels → teacher channels
    before CWD loss is computed. Built lazily from actual tensor shapes.

    Expected at your 4 hook points (S vs L scale):
        Layer 14 : S=256 → L=512   (adapter built automatically)
        Layer 18 : S=128 → L=256   (adapter built automatically)
        Layer 22 : S=256 → L=512   (adapter built automatically)
        Layer 26 : S=512 → L=512   (Identity — same channels)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(
            self.proj.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature Capture (forward hooks)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureCapture:
    """
    Attaches forward hooks to specific layer indices of a YOLO model
    and stores their output tensors. Call .clear() between steps,
    .remove() when training ends.
    """

    def __init__(self, model: nn.Module, layer_indices: list):
        self.features: dict = {}
        self._hooks: list = []
        raw = unwrap_model(model)
        for idx in layer_indices:
            self._hooks.append(
                raw.model[idx].register_forward_hook(self._make_hook(idx))
            )
        LOGGER.info(
            colorstr("KD hooks: ") +
            f"attached at layers {layer_indices}"
        )

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
    Extends DetectionTrainer with CWD Knowledge Distillation.

    Design notes:
        • KD keys are popped from overrides/cfg BEFORE super().__init__()
          so Ultralytics cfg validation never sees them.
        • KD keys are saved into self.args AFTER super().__init__()
          so the DDP subprocess (which rebuilds trainer from vars(self.args))
          can read them.
        • get_validator() strips KD keys from self.args temporarily so the
          Validator's internal get_cfg() doesn't reject them.
        • Teacher forward uses batch["img"].to(dtype=teacher_dtype) to handle
          AMP validation where images arrive as FP16 but teacher is FP32.

    Extra keys for overrides dict (consumed here, never forwarded):
        teacher_weights  str    Path to teacher best.pt   (required)
        kd_alpha         float  KD loss weight             (default 1.0)
        kd_temperature   float  CWD temperature            (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):

        # ── FIX v2: pop KD keys from overrides before cfg validation ─────────
        ov = dict(overrides) if overrides else {}
        tw    = ov.pop("teacher_weights", None)
        alpha = float(ov.pop("kd_alpha", 1.0))
        temp  = float(ov.pop("kd_temperature", 4.0))

        # ── FIX v2 (DDP path): also pop from cfg when it is a dict ───────────
        # DDP subprocess rebuilds trainer as:
        #     KDDetectionTrainer(cfg=vars(trainer.args))  # overrides=None
        # So KD keys arrive via the cfg dict in that case.
        if isinstance(cfg, dict):
            tw    = cfg.pop("teacher_weights", tw)
            alpha = float(cfg.pop("kd_alpha", alpha))
            temp  = float(cfg.pop("kd_temperature", temp))

        # ── FIX v3: never pass cfg=None explicitly ────────────────────────────
        # BaseTrainer signature: __init__(self, cfg=DEFAULT_CFG, ...)
        # Passing cfg=None explicitly overrides DEFAULT_CFG with None,
        # causing check_dict_alignment(None, overrides) → AttributeError.
        if cfg is not None:
            super().__init__(cfg=cfg, overrides=ov, _callbacks=_callbacks)
        else:
            super().__init__(overrides=ov, _callbacks=_callbacks)

        # ── FIX v5: persist KD params in self.args for DDP subprocess ─────────
        # vars(self.args) feeds the DDP temp file that recreates the trainer.
        self.args.teacher_weights = tw
        self.args.kd_alpha        = alpha
        self.args.kd_temperature  = temp

        # Runtime state — set up in setup_model()
        self.teacher      = None
        self._s_hooks     = None
        self._t_hooks     = None
        self._adapters    = {}
        self._adapters_ok = False
        self._cwd         = None

    # ── FIX v6: strip KD keys before Validator's cfg check ───────────────────
    def get_validator(self):
        """
        Temporarily removes KD keys from self.args before creating the
        validator (whose internal get_cfg → check_dict_alignment would
        reject them), then restores them so DDP still works.
        """
        saved = {}
        for key in _KD_KEYS:
            if hasattr(self.args, key):
                saved[key] = getattr(self.args, key)
                delattr(self.args, key)
        try:
            validator = super().get_validator()
        finally:
            for key, val in saved.items():
                setattr(self.args, key, val)
        return validator

    # ── A: Build student, load teacher ───────────────────────────────────────
    def setup_model(self):
        super().setup_model()   # builds self.model (student)

        # Always read from self.args — correct in both normal and DDP paths
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

        # Use self.device — set by Ultralytics before setup_model is called
        self.teacher = self.teacher.to(self.device)
        LOGGER.info(
            colorstr("KD: ") +
            f"Teacher frozen on {self.device}.  "
            f"alpha={self._kd_alpha}  temperature={self._kd_temp}"
        )

    # ── C: Attach forward hooks ───────────────────────────────────────────────
    def _attach_hooks(self):
        self._s_hooks = FeatureCapture(self.model,   _KD_LAYERS)
        self._t_hooks = FeatureCapture(self.teacher, _KD_LAYERS)

    # ── D: Build channel adapters lazily on first forward ─────────────────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
                LOGGER.info(
                    colorstr("KD adapter: ") +
                    f"layer {idx}: {sc} ch — Identity"
                )
            else:
                self._adapters[idx] = ChannelAdapter(sc, tc).to(self.device)
                LOGGER.info(
                    colorstr("KD adapter: ") +
                    f"layer {idx}: {sc}→{tc} — Conv built"
                )
        self._adapters_ok = True

    # ── E: Compute CWD from captured features ────────────────────────────────
    def _compute_kd_loss(self) -> torch.Tensor:
        sf = self._s_hooks.features
        tf = self._t_hooks.features
        self._maybe_build_adapters(sf, tf)

        kd = torch.tensor(0.0, device=self.device)
        for idx in _KD_LAYERS:
            if idx not in sf or idx not in tf:
                continue
            s, t = sf[idx], tf[idx]
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(
                    s, size=t.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            # Both must be float32 for CWD — cast student if AMP made it FP16
            s = self._adapters[idx](s.float())
            t = t.float()
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
            # 1. Standard detection loss (student forward → s_hooks fire)
            det_loss, det_items = original_loss(batch, preds)

            # 2. Teacher forward, no grad (t_hooks fire)
            # ── FIX v8: cast image to teacher dtype ──────────────────────────
            # During AMP validation batch["img"] is FP16, but teacher is FP32.
            # Casting to teacher dtype prevents the HalfTensor/FloatTensor crash.
            with torch.no_grad():
                teacher_dtype = next(trainer.teacher.parameters()).dtype
                teacher_img   = batch["img"].to(dtype=teacher_dtype)
                trainer.teacher(teacher_img)

            # 3. CWD loss
            kd = trainer._compute_kd_loss()

            # 4. Combine — det_loss is a 3-element tensor (box+cls+dfl)
            # so we add kd (scalar) and return the combined tensor as-is.
            total = det_loss + trainer._kd_alpha * kd

            return total, det_items

        raw_model.loss = kd_loss_fn
        LOGGER.info(colorstr("KD: ") + "Loss injection complete ✓")

    # ── FIX v4: no world_size arg — this fork's signature has none ────────────
    def _setup_ddp(self):
        super()._setup_ddp()
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)

    # ── G: Clean up hooks when training ends ──────────────────────────────────
    def final_eval(self):
        if self._s_hooks:
            self._s_hooks.remove()
        if self._t_hooks:
            self._t_hooks.remove()
        super().final_eval()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry-point — call from Kaggle notebook
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

    Args:
        student_yaml      Path to YOLOv11-C3k2-SC_DySample_ResEMA-S.yaml
        teacher_weights   Path to trained YOLOv11L best.pt (frozen teacher)
        data_yaml         Path to meta.yaml
        epochs            Total training epochs            (default 150)
        imgsz             Input image size                 (default 1024)
        batch             Batch size — use 16, not 32.
                          Teacher + student both hold activations in VRAM.
        device            GPU devices string, e.g. '0,1'
        kd_alpha          KD loss weight. Sweep: 0.5, 1.0, 2.0, 4.0
        kd_temperature    CWD softmax temperature          (default 4.0)
        project           Output project folder
        name              Run sub-folder name
        resume            Resume from last checkpoint?

    Returns:
        KDDetectionTrainer (training already complete).
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
        # KD keys — popped in __init__ before Ultralytics cfg validation,
        # saved to self.args for DDP, stripped again in get_validator().
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer