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
    v2 — Pop KD keys before super().__init__() (cfg validation crash)
    v3 — Don't pass cfg=None to super (DEFAULT_CFG crash)
    v4 — Remove world_size from _setup_ddp (DDP signature mismatch)
    v5 — Save KD keys into self.args after super().__init__() so the
         Ultralytics DDP subprocess can read them when it re-creates
         the trainer. Also pop KD keys from cfg dict (DDP case) as
         well as from overrides dict (normal case).

HOW DDP BREAKS KD (the v5 root cause):
    Ultralytics DDP spawns a subprocess using a temp Python file that
    re-creates the trainer as:
        cfg = DEFAULT_CFG_DICT.copy()
        cfg.update(vars(trainer.args))       ← only keys in self.args survive
        trainer = KDDetectionTrainer(cfg=cfg) ← overrides=None in subprocess
    Because teacher_weights was popped from overrides and stored only as
    self._kd_teacher_weights (not in self.args), the subprocess never saw
    it and KD was silently disabled.
    Fix: after super().__init__(), store KD params in self.args so they
    appear in vars(trainer.args) and survive into the DDP subprocess.
    Also: pop KD keys from cfg when cfg is a dict (DDP subprocess path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model


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
    """1×1 conv projecting student channels → teacher channels before CWD."""

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
    """Attaches forward hooks to specific layer indices, stores output tensors."""

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

    Extra overrides keys (handled here, stripped before Ultralytics sees them):
        teacher_weights  str    Path to teacher best.pt   (required)
        kd_alpha         float  KD loss weight             (default 1.0)
        kd_temperature   float  CWD temperature            (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):

        # ── Step 1: extract KD keys from overrides (normal training path) ────
        ov = dict(overrides) if overrides else {}
        tw    = ov.pop("teacher_weights", None)
        alpha = float(ov.pop("kd_alpha", 1.0))
        temp  = float(ov.pop("kd_temperature", 4.0))

        # ── Step 2: also extract from cfg when it is a dict (DDP subprocess) ─
        # Ultralytics DDP re-creates the trainer as:
        #     cfg = DEFAULT_CFG_DICT.copy(); cfg.update(vars(trainer.args))
        #     trainer = KDDetectionTrainer(cfg=cfg)   # overrides=None
        # So KD keys land in cfg dict, not in overrides.
        if isinstance(cfg, dict):
            tw    = cfg.pop("teacher_weights", tw)
            alpha = float(cfg.pop("kd_alpha", alpha))
            temp  = float(cfg.pop("kd_temperature", temp))

        # Store before calling super so they're available in setup_model()
        self._kd_teacher_weights = tw
        self._kd_alpha           = alpha
        self._kd_temp            = temp

        # ── Step 3: call super WITHOUT cfg=None (would override DEFAULT_CFG) ─
        if cfg is not None:
            super().__init__(cfg=cfg, overrides=ov, _callbacks=_callbacks)
        else:
            super().__init__(overrides=ov, _callbacks=_callbacks)

        # ── Step 4: persist KD params into self.args ─────────────────────────
        # self.args is a SimpleNamespace created by Ultralytics from the config.
        # Ultralytics DDP reads vars(trainer.args) to build the subprocess cfg.
        # Saving our params here means they survive into the DDP subprocess.
        self.args.teacher_weights = self._kd_teacher_weights
        self.args.kd_alpha        = self._kd_alpha
        self.args.kd_temperature  = self._kd_temp

        # Runtime state
        self.teacher      = None
        self._s_hooks     = None
        self._t_hooks     = None
        self._adapters    = {}
        self._adapters_ok = False
        self._cwd         = None
        self._kd_step     = 0

    # ── A: Build student, load teacher ───────────────────────────────────────
    def setup_model(self):
        super().setup_model()

        # Read from self.args (always up-to-date, survives DDP)
        teacher_weights = getattr(self.args, "teacher_weights", None)
        self._kd_alpha  = float(getattr(self.args, "kd_alpha", 1.0))
        self._kd_temp   = float(getattr(self.args, "kd_temperature", 4.0))

        if not teacher_weights:
            LOGGER.warning(
                colorstr("KD WARNING: ") +
                "teacher_weights not set — KD loss DISABLED. "
                "Check that teacher_weights= was passed correctly."
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

        device = next(self.model.parameters()).device
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

    # ── D: Build channel adapters (lazy) ──────────────────────────────────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        device = next(self.model.parameters()).device
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
                LOGGER.info(colorstr("KD adapter: ") + f"layer {idx}: {sc} ch — Identity")
            else:
                self._adapters[idx] = ChannelAdapter(sc, tc).to(device)
                LOGGER.info(colorstr("KD adapter: ") + f"layer {idx}: {sc}→{tc} ch — Conv built")
        self._adapters_ok = True

    # ── E: Compute CWD ────────────────────────────────────────────────────────
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

    # ── G: Keep teacher on correct GPU after DDP ──────────────────────────────
    def _setup_ddp(self):
        super()._setup_ddp()
        if self.teacher is not None:
            device = next(self.model.parameters()).device
            self.teacher = self.teacher.to(device)

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
    batch=16 not 32 — teacher + student share GPU VRAM during forward.
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
        # then saved to self.args so DDP subprocess can read them.
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer