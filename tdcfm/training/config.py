"""Training configuration. Stages 1-3 stay in physical units; normalisation to
``[-1, 1]`` happens here, in the dataloader.

Every experimental switch in this work lives in :class:`TrainCfg`. Command
lines override fields positionally as ``key=value``, and the resolved
configuration is stored inside each checkpoint, so evaluation can reconstruct
the exact conditioning a model was trained with instead of being told by hand.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass


@dataclass
class TrainCfg:
    # ---- data ----
    data_dir: str = "data/stage3/train"    # stage 3 output (with manifest.json)
    image_size: int = 256
    cond_channels: int = 4                 # [rms, well_val, well_mask, imaging]
    num_classes: int = 6                   # 5 geological patterns + 1 reserved null class
    n_real_classes: int = 5
    v_min: float = 1500.0                  # velocity normalisation range (m/s)
    v_max: float = 6000.0
    random_flip: bool = True               # horizontal flip augmentation

    # ---- RMS condition ----
    # Target *total* smoothing sigma in samples, including the baseline sigma
    # already applied in stage 3. Gaussians compose as
    # sigma_total^2 = sigma_base^2 + sigma_extra^2, so the dataloader only has to
    # add the difference and the dataset never has to be regenerated. Values
    # below the stage-3 baseline cannot be undone and fall back to the baseline.
    # The condition is deliberately smoothed far beyond the stage-3 baseline:
    # a lightly smoothed RMS field is close enough to the answer that the network
    # can copy it instead of learning to generate.
    rms_smooth_tau: float = 59.0
    rms_smooth_x: float = 26.0

    # ---- well condition ----
    # Rebuild the well condition from the label instead of using the channels
    # written by stage 3. The two are identical under matching settings (see
    # tests/test_wells.py); rebuilding is what allows the well count and width
    # to be changed without regenerating the dataset.
    wells_from_target: bool = True
    n_wells: int = 4
    well_width: int = 3
    # Interpolate laterally between the wells. Off by default, and that matters:
    # these velocity models are laterally smooth, so linearly interpolating even
    # 4 wells covering 4.7% of the section reconstructs the truth to ~44 m/s MAE.
    # Switching it on hands the network something very close to the answer and
    # makes any conclusion about how much it relies on the wells meaningless.
    well_interp: bool = False

    # ---- imaging condition ----
    # False drops condition channel 3 entirely; the UNet input shrinks to 3
    # condition channels accordingly (see active_cond_channels).
    use_imaging: bool = True

    # ---- network (torchcfm UNetModel) ----
    num_channel: int = 128                 # model_channels
    num_res_blocks: int = 2
    num_heads: int = 4
    num_head_channels: int = 64
    attention_res: int = 16                # attention at resolution size // 16
    dropout: float = 0.1

    # ---- flow matching / optimisation ----
    model: str = "otcfm"                   # otcfm | icfm | fm | si
    sigma: float = 0.0
    lr: float = 2e-4
    batch_size: int = 48                   # per GPU under DDP
    total_steps: int = 80000
    warmup: int = 2000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    # EMA cold-start correction. Without it, an early checkpoint's EMA weights
    # retain only (1 - decay^step) of the learned displacement (39% at 5k steps
    # with decay=0.9999) and systematically understate training progress.
    ema_warmup: bool = True
    num_workers: int = 8
    seed: int = 0

    # ---- performance ----
    amp: str = "bf16"                      # bf16 | fp16 | off
    tf32: bool = True                      # TF32 for matmul and cuDNN
    cudnn_benchmark: bool = True           # input shape is fixed, so this pays off
    channels_last: bool = False            # optional extra convolution throughput

    # ---- well consistency loss ----
    # L = lambda_fm * L_FM + lambda_well * L_well
    # The constraint acts on x1_hat = x_t + (1-t) * v_theta, which the forward
    # pass of the flow-matching loss already provides, so no ODE solve is
    # involved. L_well is still evaluated and written to loss_log.csv when its
    # weight is zero, so the control run stays directly comparable.
    lambda_fm: float = 1.0
    lambda_well: float = 0.35

    # ---- sampling and checkpointing ----
    ode_steps: int = 100                   # Euler steps for the monitoring samples
    sample_n: int = 4                      # samples per monitoring figure
    save_step: int = 5000
    keep_last_n: int = 3                   # keep only the N newest checkpoints; <=0 keeps all
    log_every: int = 50                    # rows appended to loss_log.csv
    out_dir: str = "runs/train"
    resume: str = ""                       # checkpoint to resume from

    # ---- debugging ----
    overfit: int = 0                       # >0: train on the first N samples only

    def to_dict(self) -> dict:
        return asdict(self)


def active_cond_channels(cfg: TrainCfg) -> int:
    """Number of condition channels actually concatenated onto the input."""
    return cfg.cond_channels - (0 if cfg.use_imaging else 1)


def apply_overrides(cfg: TrainCfg, overrides) -> TrainCfg:
    """Apply ``key=value`` overrides in place, casting to the declared field type.

    Booleans must be handled explicitly: ``from __future__ import annotations``
    turns dataclass field types into strings, and every non-empty string is
    truthy, so without this branch ``flag=False`` would be read as True.
    """
    types = {f.name: f.type for f in dataclasses.fields(cfg)}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        if k not in types:
            raise KeyError(f"unknown configuration field: {k}")
        tp = types[k]
        if tp in (bool, "bool"):
            v = v.strip().lower() in ("1", "true", "yes", "y", "on")
        elif tp in (int, "int"):
            v = int(v)
        elif tp in (float, "float"):
            v = float(v)
        setattr(cfg, k, v)
    return cfg


def cfg_from_ckpt(d: dict) -> TrainCfg:
    """Rebuild a :class:`TrainCfg` from the dict stored in a checkpoint.

    Unknown keys are ignored, so a checkpoint stays loadable after new fields
    are added. Evaluation always goes through this function: evaluating a model
    under conditioning settings other than the ones it was trained with fails
    silently rather than loudly, and produces numbers that look plausible and
    are wrong.
    """
    known = {k: v for k, v in (d or {}).items() if k in TrainCfg.__dataclass_fields__}
    return TrainCfg(**known)


def channel_mult_for(image_size: int):
    """UNet channel multipliers per resolution level."""
    table = {
        512: (0.5, 1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 2, 2),
    }
    if image_size not in table:
        raise ValueError(f"unsupported image_size={image_size}")
    return table[image_size]
