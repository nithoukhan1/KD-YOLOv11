"""
kd_trainer.py
=============
Knowledge Distillation trainer for YOLOv11-C3k2-SC-DySample-ResEMA.

Place this file at:
    KD-YOLOv11/ultralytics/engine/kd_trainer.py

Teacher : YOLOv11L-C3k2-SC-DySample-ResEMA  (frozen, same custom arch)
Student : YOLOv11S-C3k2-SC-DySample-ResEMA  (trainable — your model)
KD Loss : Channel-Wise Distillation (CWD) at all 4 ResEMA neck outputs
          YAML layer indices: 14, 18, 22, 26

Fix log:
    v1 — de_parallel → unwrap_model  (this repo removed de_parallel)
    v2 — KD keys (teacher_weights, kd_alpha, kd_temperature) are popped
         from the overrides dict BEFORE calling super().__init__() so that
         Ultralytics check_dict_alignment() never sees them and does not
         raise AttributeError: 'NoneType' object has no attribute 'keys'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model   # not de_parallel


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CWD Loss  —  Channel-Wise Distillation
# ─────────────────────────────────────────────────────────────────────────────
class CWDLoss(nn.Module):
    """
    Channel-Wise Distillation loss.

    For each channel, treats the H×W spatial map as a probability distribution
    (softmax at temperature T) and minimises KL divergence between the teacher
    and student distributions channel-by-channel.

    Reference: Shu et al. "Channel-wise Knowledge Distillation for Dense
    Prediction", ICCV 2021.

    Args:
        temperature (float): Softmax temperature. Higher = softer targets.
                             Recommended: 2.0 – 6.0. Default: 4.0.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        assert student.shape == teacher.shape, (
            f"CWD shape mismatch — student {student.shape} vs "
            f"teacher {teacher.shape}. Check ChannelAdapter."
        )
        B, C, H, W = student.shape
        s = F.softmax(student.view(B, C, -1) / self.T, dim=-1)
        t = F.softmax(teacher.view(B, C, -1) / self.T, dim=-1)
        return F.kl_div(s.log(), t, reduction="batchmean")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel Adapter  —  1×1 conv aligning student → teacher channels
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """
    Lightweight 1×1 conv projecting student feature channels to match teacher.
    Built lazily on first forward pass — no hard-coded channel numbers needed.

    Expected channel sizes at your 4 hook points (S vs L scale):
        Layer 14 : S=256  →  L=512   (adapter built automatically)
        Layer 18 : S=128  →  L=256   (adapter built automatically)
        Layer 22 : S=256  →  L=512   (adapter built automatically)
        Layer 26 : S=512  →  L=512   (Identity — same channels)
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
# 3.  Feature Capture  —  PyTorch forward hooks
# ─────────────────────────────────────────────────────────────────────────────
class FeatureCapture:
    """
    Attaches forward hooks to specific layer indices of a YOLO model and
    stores their output tensors in self.features dict.

    Usage:
        capture = FeatureCapture(model, [14, 18, 22, 26])
        _ = model(img)                  # forward fires hooks
        feats = capture.features        # {14: tensor, 18: tensor, ...}
        capture.clear()                 # reset between steps
        capture.remove()                # detach all hooks when done
    """

    def __init__(self, model: nn.Module, layer_indices: list):
        self.features: dict = {}
        self._hooks: list = []
        raw = unwrap_model(model)
        for idx in layer_indices:
            hook = raw.model[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)
        LOGGER.info(
            colorstr("KD hooks: ") +
            f"attached at layer indices {layer_indices}"
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

# ResEMA output layer indices in your neck YAML
_KD_LAYERS = [14, 18, 22, 26]

# Keys that belong to KD only — must be stripped from overrides before
# passing to super().__init__() so Ultralytics cfg validation doesn't crash.
_KD_KEYS = ("teacher_weights", "kd_alpha", "kd_temperature")


class KDDetectionTrainer(DetectionTrainer):
    """
    Extends Ultralytics DetectionTrainer with CWD Knowledge Distillation.

    Key design decision — KD keys are popped from `overrides` before
    super().__init__() is called.  Ultralytics runs check_dict_alignment()
    inside get_cfg() which rejects any key not in the default config schema.
    Popping them first prevents the AttributeError crash.

    Extra keys for the overrides dict (consumed here, not forwarded):
        teacher_weights  str    Path to teacher best.pt  (required)
        kd_alpha         float  KD loss weight            (default 1.0)
        kd_temperature   float  CWD temperature           (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # ── CRITICAL: pop KD-specific keys BEFORE super().__init__() ────────
        # Ultralytics check_dict_alignment() raises AttributeError / ValueError
        # if it encounters unknown keys.  We extract them here and store them
        # as instance attributes; the parent never sees them.
        ov = dict(overrides) if overrides else {}   # work on a copy
        self._kd_teacher_weights = ov.pop("teacher_weights", None)
        self._kd_alpha           = float(ov.pop("kd_alpha", 1.0))
        self._kd_temp            = float(ov.pop("kd_temperature", 4.0))

        # Now it is safe to call parent — cleaned overrides only
        super().__init__(cfg=cfg, overrides=ov, _callbacks=_callbacks)

        # Runtime state — set up in setup_model()
        self.teacher      = None
        self._s_hooks     = None
        self._t_hooks     = None
        self._adapters    = {}
        self._adapters_ok = False
        self._cwd         = None
        self._kd_step     = 0

    # ── A: Build student, then load teacher ──────────────────────────────────
    def setup_model(self):
        """Called by BaseTrainer.train(). Builds student then injects KD."""
        super().setup_model()   # builds self.model (student)

        if not self._kd_teacher_weights:
            LOGGER.warning(
                colorstr("KD WARNING: ") +
                "teacher_weights not provided — KD loss is DISABLED."
            )
            return

        self._load_teacher()
        self._attach_hooks()
        self._cwd = CWDLoss(temperature=self._kd_temp)
        self._inject_kd_loss()

    # ── B: Load and freeze teacher ────────────────────────────────────────────
    def _load_teacher(self):
        path = Path(self._kd_teacher_weights)
        assert path.exists(), (
            f"Teacher weights not found: {path}\n"
            f"Check the teacher_weights= path."
        )
        LOGGER.info(colorstr("KD: ") + f"Loading teacher from {path}")

        from ultralytics import YOLO
        self.teacher = YOLO(str(path)).model   # raw DetectionModel

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        device = next(self.model.parameters()).device
        self.teacher = self.teacher.to(device)

        LOGGER.info(
            colorstr("KD: ") +
            f"Teacher frozen on {device}.  "
            f"alpha={self._kd_alpha}  temperature={self._kd_temp}"
        )

    # ── C: Attach forward hooks ───────────────────────────────────────────────
    def _attach_hooks(self):
        self._s_hooks = FeatureCapture(self.model,   _KD_LAYERS)
        self._t_hooks = FeatureCapture(self.teacher, _KD_LAYERS)

    # ── D: Build channel adapters (lazy — from first forward shapes) ──────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        device = next(self.model.parameters()).device
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
                LOGGER.info(
                    colorstr("KD adapter: ") +
                    f"layer {idx}: {sc} ch — Identity (channels match)"
                )
            else:
                self._adapters[idx] = ChannelAdapter(sc, tc).to(device)
                LOGGER.info(
                    colorstr("KD adapter: ") +
                    f"layer {idx}: {sc}→{tc} ch — 1×1 Conv built"
                )
        self._adapters_ok = True

    # ── E: Compute CWD from captured features ────────────────────────────────
    def _compute_kd_loss(self) -> torch.Tensor:
        sf = self._s_hooks.features
        tf = self._t_hooks.features
        self._maybe_build_adapters(sf, tf)

        device = next(self.model.parameters()).device
        kd = torch.tensor(0.0, device=device)

        for idx in _KD_LAYERS:
            if idx not in sf or idx not in tf:
                continue
            s, t = sf[idx], tf[idx]
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(
                    s, size=t.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            s = self._adapters[idx](s)
            kd = kd + self._cwd(s, t)

        self._s_hooks.clear()
        self._t_hooks.clear()
        return kd / len(_KD_LAYERS)

    # ── F: Replace model.loss with KD-injected version ────────────────────────
    def _inject_kd_loss(self):
        raw_model     = unwrap_model(self.model)
        original_loss = raw_model.loss
        trainer       = self

        def kd_loss_fn(batch, preds=None):
            # 1. Standard detection loss (student forward → s_hooks fire)
            det_loss, det_items = original_loss(batch, preds)

            # 2. Teacher forward, no grad (t_hooks fire)
            with torch.no_grad():
                trainer.teacher(batch["img"])

            # 3. CWD
            kd = trainer._compute_kd_loss()

            # 4. Combine
            total = det_loss + trainer._kd_alpha * kd

            # Periodic debug log
            trainer._kd_step += 1
            if trainer._kd_step % 50 == 0:
                LOGGER.debug(
                    colorstr("KD: ") +
                    f"step={trainer._kd_step}  "
                    f"det={det_loss.item():.4f}  "
                    f"kd={kd.item():.5f}  "
                    f"total={total.item():.4f}"
                )
            return total, det_items

        raw_model.loss = kd_loss_fn
        LOGGER.info(colorstr("KD: ") + "Loss injection complete ✓")

    # ── G: Keep teacher on correct GPU after DDP wrapping ─────────────────────
    def _setup_ddp(self, world_size):
        super()._setup_ddp(world_size)
        if self.teacher is not None:
            device = next(self.model.parameters()).device
            self.teacher = self.teacher.to(device)

    # ── H: Clean up hooks when training ends ──────────────────────────────────
    def final_eval(self):
        if self._s_hooks:
            self._s_hooks.remove()
        if self._t_hooks:
            self._t_hooks.remove()
        super().final_eval()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry-point — call this from your Kaggle notebook
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
        batch             Batch size — use 16 not 32.
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
        # KD-specific keys — consumed by KDDetectionTrainer.__init__,
        # popped from overrides before Ultralytics cfg validation runs.
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )

    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer