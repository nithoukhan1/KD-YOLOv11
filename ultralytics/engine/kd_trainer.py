"""
kd_trainer.py  —  Final version, all fixes applied
====================================================
Place at:  KD-YOLOv11/ultralytics/engine/kd_trainer.py

Teacher : YOLOv11L-C3k2-SC-DySample-ResEMA  (frozen)
Student : YOLOv11S-C3k2-SC-DySample-ResEMA  (trainable)
KD Loss : Channel-Wise Distillation (CWD) at ResEMA neck outputs
          YAML layer indices: 14, 18, 22, 26

Fix history:
    v1  — de_parallel → unwrap_model
    v2  — Pop KD keys from overrides/cfg before super().__init__()
    v3  — Don't pass cfg=None to super (overrides DEFAULT_CFG)
    v4  — _setup_ddp() has no world_size arg in this fork
    v5  — Save KD keys into self.args so DDP subprocess reads them
    v6  — Override get_validator() to strip KD keys before validator check
    v7  — Remove debug logger (det_loss is 3-element tensor not scalar)
    v8  — Cast batch["img"] to teacher dtype (AMP FP16 vs FP32 teacher)
    v9  — Override save_model() to swap loss before/after pickling
    v10 — _KDLossFn and _Hook as picklable top-level classes
    v11 — Inject KD loss in _setup_train() AFTER EMA is created
    v12 — Override check_resume() to fix epoch counter on resume:
          Root cause: self.args.resume is stored as a RELATIVE path
          (e.g. runs/detect/.../last.pt). The DDP subprocess may have a
          different working directory so it cannot resolve the relative
          path, catches the FileNotFoundError silently, and resets
          start_epoch=0 — causing the epoch counter AND LR schedule to
          restart from epoch 1.
          Fix: convert resume path to absolute before calling super, so
          both main process and DDP subprocess reliably find the checkpoint
          and correctly restore start_epoch, optimizer state, and LR.
          Also strips KD keys from checkpoint train_args so get_cfg does
          not reject them with SyntaxError.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

_KD_KEYS   = ("teacher_weights", "kd_alpha", "kd_temperature")
_KD_LAYERS = [14, 18, 22, 26]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CWD Loss
# ─────────────────────────────────────────────────────────────────────────────
class CWDLoss(nn.Module):
    """
    Channel-Wise Distillation loss.
    Treats each channel's H×W map as a probability distribution (softmax at T)
    and minimises KL divergence between teacher and student per channel.
    Reference: Shu et al., ICCV 2021.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student: torch.Tensor,
                teacher: torch.Tensor) -> torch.Tensor:
        assert student.shape == teacher.shape, (
            f"CWD shape mismatch: {student.shape} vs {teacher.shape}"
        )
        B, C, H, W = student.shape
        s = F.softmax(student.view(B, C, -1) / self.T, dim=-1)
        t = F.softmax(teacher.view(B, C, -1) / self.T, dim=-1)
        return F.kl_div(s.log(), t, reduction="batchmean")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Channel Adapter
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """1×1 conv projecting student channels → teacher channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(
            self.proj.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hook callable — picklable top-level class
# ─────────────────────────────────────────────────────────────────────────────
class _Hook:
    """Picklable forward-hook. Module-level so torch.save can serialise it."""

    def __init__(self, store: dict, idx: int):
        self.store = store
        self.idx   = idx

    def __call__(self, module, inp, out):
        self.store[self.idx] = out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Feature Capture
# ─────────────────────────────────────────────────────────────────────────────
class FeatureCapture:
    """Attaches picklable forward hooks and stores output tensors."""

    def __init__(self, model: nn.Module, layer_indices: list):
        self.features: dict = {}
        self._hooks: list   = []
        raw = unwrap_model(model)
        for idx in layer_indices:
            h = raw.model[idx].register_forward_hook(
                _Hook(self.features, idx)
            )
            self._hooks.append(h)
        LOGGER.info(colorstr("KD hooks: ") +
                    f"attached at layers {layer_indices}")

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  KD Loss callable — picklable top-level class
# ─────────────────────────────────────────────────────────────────────────────
class _KDLossFn:
    """
    Picklable replacement for model.loss that adds CWD distillation.
    Module-level class (not a closure) so torch.save can serialise it.
    Injected AFTER EMA creation to avoid deepcopy of dataloader iterator.
    """

    def __init__(self, original_loss, trainer):
        self.original_loss = original_loss
        self.trainer       = trainer

    def __call__(self, batch, preds=None):
        tr = self.trainer

        # 1. Standard detection loss — student forward fires s_hooks
        det_loss, det_items = self.original_loss(batch, preds)

        # 2. Teacher forward, no grad — t_hooks fire
        # FIX v8: cast to teacher dtype — AMP sends FP16, teacher is FP32
        with torch.no_grad():
            teacher_dtype = next(tr.teacher.parameters()).dtype
            teacher_img   = batch["img"].to(dtype=teacher_dtype)
            tr.teacher(teacher_img)

        # 3. CWD
        kd = tr._compute_kd_loss()

        # 4. Combine — det_loss is 3-element tensor (box+cls+dfl)
        total = det_loss + tr._kd_alpha * kd
        return total, det_items


# ─────────────────────────────────────────────────────────────────────────────
# 6.  KD Detection Trainer
# ─────────────────────────────────────────────────────────────────────────────
class KDDetectionTrainer(DetectionTrainer):
    """
    Extends DetectionTrainer with CWD Knowledge Distillation.

    Extra overrides keys (consumed here, never forwarded):
        teacher_weights  str    Path to teacher best.pt   (required)
        kd_alpha         float  KD loss weight             (default 1.0)
        kd_temperature   float  CWD temperature            (default 4.0)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):

        # FIX v2: pop KD keys before cfg validation
        ov    = dict(overrides) if overrides else {}
        tw    = ov.pop("teacher_weights", None)
        alpha = float(ov.pop("kd_alpha", 1.0))
        temp  = float(ov.pop("kd_temperature", 4.0))

        # FIX v2 DDP path: also pop from cfg dict
        if isinstance(cfg, dict):
            tw    = cfg.pop("teacher_weights", tw)
            alpha = float(cfg.pop("kd_alpha", alpha))
            temp  = float(cfg.pop("kd_temperature", temp))

        # FIX v3: never pass cfg=None explicitly
        if cfg is not None:
            super().__init__(cfg=cfg, overrides=ov, _callbacks=_callbacks)
        else:
            super().__init__(overrides=ov, _callbacks=_callbacks)

        # FIX v5: persist KD params in self.args for DDP subprocess
        self.args.teacher_weights = tw
        self.args.kd_alpha        = alpha
        self.args.kd_temperature  = temp

        # Runtime state
        self.teacher        = None
        self._s_hooks       = None
        self._t_hooks       = None
        self._adapters      = {}
        self._adapters_ok   = False
        self._cwd           = None
        self._original_loss = None
        self._kd_ready      = False

    # ── FIX v12: correct epoch counter on resume ──────────────────────────────
    def check_resume(self, overrides):
        """
        Two problems fixed here:

        1. RELATIVE PATH BUG (epoch counter resets):
           self.args.resume is stored as a relative path like
           'runs/detect/.../last.pt'. The DDP subprocess may have a
           different CWD and cannot resolve the relative path. It catches
           the failure silently and resets start_epoch=0, causing the
           epoch counter and LR schedule to restart from epoch 1.
           Fix: resolve the resume path to absolute before calling super.

        2. KD KEYS IN CHECKPOINT:
           When a checkpoint was saved by our trainer, self.args contained
           KD keys. These get persisted in checkpoint train_args. When
           check_resume calls get_cfg(ckpt_args), Ultralytics rejects the
           unknown keys with SyntaxError. We strip them first.
        """
        resume = getattr(self.args, "resume", None)

        if not resume or resume is True:
            # No resume path — nothing to fix
            super().check_resume(overrides)
            return

        # ── FIX 1: make path absolute ─────────────────────────────────────────
        resume_path = Path(str(resume))
        if not resume_path.is_absolute():
            # Try resolving against CWD
            abs_path = Path.cwd() / resume_path
            if abs_path.exists():
                self.args.resume = str(abs_path)
                LOGGER.info(
                    colorstr("KD resume: ") +
                    f"Resolved relative path to absolute: {abs_path}"
                )
            else:
                # Try common Kaggle working dir patterns
                for base in [
                    Path("/kaggle/working/KD-YOLOv11"),
                    Path("/kaggle/working"),
                ]:
                    candidate = base / resume_path
                    if candidate.exists():
                        self.args.resume = str(candidate)
                        LOGGER.info(
                            colorstr("KD resume: ") +
                            f"Found checkpoint at: {candidate}"
                        )
                        break

        # ── FIX 2: strip KD keys from checkpoint train_args ──────────────────
        # Load checkpoint raw dict, strip our custom keys, save to temp file,
        # let super()'s check_resume load the clean version.
        resolved = Path(str(self.args.resume))

        if resolved.exists():
            try:
                import tempfile, os
                ckpt = torch.load(
                    str(resolved), map_location="cpu", weights_only=False
                )
                train_args = ckpt.get("train_args", {})
                kd_vals    = {k: train_args.pop(k, None) for k in _KD_KEYS}
                ckpt["train_args"] = train_args

                # Write clean checkpoint to temp file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".pt", delete=False,
                    dir=str(resolved.parent)
                )
                tmp.close()
                torch.save(ckpt, tmp.name)

                # Point resume at the clean temp file
                original_resume  = self.args.resume
                self.args.resume = tmp.name

                try:
                    super().check_resume(overrides)
                finally:
                    os.unlink(tmp.name)
                    # Restore BOTH resume and model to absolute path
                    # super().check_resume sets self.args.model to the temp
                    # file path — DDP subprocess then fails when it tries
                    # to load the already-deleted temp file.
                    if hasattr(self.args, "resume"):
                        self.args.resume = str(resolved)
                    if hasattr(self.args, "model"):
                        self.args.model = str(resolved)

                # Restore KD values that were in the checkpoint
                for k, v in kd_vals.items():
                    if v is not None and not getattr(self.args, k, None):
                        setattr(self.args, k, v)

                LOGGER.info(
                    colorstr("KD resume: ") +
                    f"Checkpoint loaded cleanly. Epoch will resume correctly."
                )
                return

            except Exception as e:
                LOGGER.warning(
                    colorstr("KD resume WARNING: ") +
                    f"Clean-load attempt failed ({e}), falling back to super."
                )

        # Fallback — let super handle it as-is
        super().check_resume(overrides)

    # ── FIX v6: strip KD keys before Validator cfg check ─────────────────────
    def get_validator(self):
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

    # ── FIX v9: restore original loss before save, re-inject after ────────────
    def save_model(self):
        raw        = unwrap_model(self.model)
        kd_loss_fn = raw.loss if self._original_loss is not None else None
        if kd_loss_fn is not None:
            raw.loss = self._original_loss
        try:
            super().save_model()
        finally:
            if kd_loss_fn is not None:
                raw.loss = kd_loss_fn

    # ── A: Build student and load teacher ─────────────────────────────────────
    #       Does NOT inject model.loss — that happens after EMA is created.
    def setup_model(self):
        super().setup_model()

        teacher_weights  = getattr(self.args, "teacher_weights", None)
        self._kd_alpha   = float(getattr(self.args, "kd_alpha",       1.0))
        self._kd_temp    = float(getattr(self.args, "kd_temperature", 4.0))

        if not teacher_weights:
            LOGGER.warning(colorstr("KD WARNING: ") +
                           "teacher_weights not set — KD DISABLED.")
            return

        self._kd_teacher_weights = teacher_weights
        self._load_teacher()
        self._attach_hooks()
        self._cwd = CWDLoss(temperature=self._kd_temp)

    # ── FIX v11: inject KD AFTER EMA is created ───────────────────────────────
    def _setup_train(self):
        super()._setup_train()
        if self.teacher is not None and not self._kd_ready:
            self._inject_kd_loss()
            self._kd_ready = True

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
        self.teacher = self.teacher.to(self.device)

        LOGGER.info(colorstr("KD: ") +
                    f"Teacher frozen on {self.device}.  "
                    f"alpha={self._kd_alpha}  temperature={self._kd_temp}")

    # ── C: Attach hooks ───────────────────────────────────────────────────────
    def _attach_hooks(self):
        self._s_hooks = FeatureCapture(self.model,   _KD_LAYERS)
        self._t_hooks = FeatureCapture(self.teacher, _KD_LAYERS)

    # ── D: Build channel adapters lazily ──────────────────────────────────────
    def _maybe_build_adapters(self, s_feats: dict, t_feats: dict):
        if self._adapters_ok:
            return
        for idx in _KD_LAYERS:
            sc = s_feats[idx].shape[1]
            tc = t_feats[idx].shape[1]
            if sc == tc:
                self._adapters[idx] = nn.Identity()
                LOGGER.info(colorstr("KD adapter: ") +
                            f"layer {idx}: {sc} ch — Identity")
            else:
                self._adapters[idx] = ChannelAdapter(sc, tc).to(self.device)
                LOGGER.info(colorstr("KD adapter: ") +
                            f"layer {idx}: {sc}→{tc} — Conv built")
        self._adapters_ok = True

    # ── E: Compute CWD loss ───────────────────────────────────────────────────
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
                s = F.interpolate(s, size=t.shape[-2:],
                                  mode="bilinear", align_corners=False)
            s = self._adapters[idx](s.float())
            t = t.float()
            kd = kd + self._cwd(s, t)

        self._s_hooks.clear()
        self._t_hooks.clear()
        return kd / len(_KD_LAYERS)

    # ── F: Inject KD loss ─────────────────────────────────────────────────────
    def _inject_kd_loss(self):
        raw_model           = unwrap_model(self.model)
        self._original_loss = raw_model.loss
        raw_model.loss      = _KDLossFn(self._original_loss, self)
        LOGGER.info(colorstr("KD: ") + "Loss injection complete ✓")

    # ── FIX v4: no world_size arg in this fork ────────────────────────────────
    def _setup_ddp(self):
        super()._setup_ddp()
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)

    # ── G: Clean up hooks ─────────────────────────────────────────────────────
    def final_eval(self):
        if self._s_hooks:
            self._s_hooks.remove()
        if self._t_hooks:
            self._t_hooks.remove()
        super().final_eval()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Public entry-point
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

    For resume:
        Pass the path to last.pt as student_yaml and set resume=True.
        The epoch counter will correctly continue from the saved epoch.
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
        teacher_weights = teacher_weights,
        kd_alpha        = kd_alpha,
        kd_temperature  = kd_temperature,
    )
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()
    return trainer