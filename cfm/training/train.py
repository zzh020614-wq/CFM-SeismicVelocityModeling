"""Stage 4 entry point: OT-CFM training with channel and label conditioning.

Single GPU::

    python -m cfm.training.train --data-dir data/stage3/train \\
        --out-dir runs/train total_steps=80000 batch_size=48 amp=bf16

Multi-GPU (DDP; ``batch_size`` is per GPU)::

    torchrun --standalone --nproc_per_node=2 --module cfm.training.train \\
        --data-dir data/stage3/train --out-dir runs/train \\
        total_steps=80000 batch_size=24

Resume with ``--resume runs/train/otcfm/ckpt_step_30000.pt``. Any
:class:`~cfm.training.config.TrainCfg` field can be overridden as a trailing
``key=value`` argument.

Outputs land in ``<out-dir>/<model>/``: ``ckpt_step_*.pt`` (the newest
``keep_last_n`` only), ``sample_step_*.png`` (ground truth on top, generated
below) and ``loss_log.csv``.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

from ..models.unet import build_unet, euler_sample
from .config import TrainCfg, active_cond_channels, apply_overrides
from .dataset import ShardConditionDataset
from .flow import sample_flow_batch
from .physics import combine_losses, physics_loss, physics_metrics


def seed_everything(seed: int):
    """Fix the Python, NumPy and PyTorch random sources used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset_batch_size(dataset_size: int, batch_size: int,
                                world_size: int = 1):
    """Guard against ``drop_last=True`` leaving a rank with no batch at all."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if dataset_size <= 0:
        raise ValueError("training dataset is empty; check --data-dir and its manifest.json")
    required = batch_size * world_size
    if dataset_size < required:
        raise ValueError(
            f"dataset size {dataset_size} < batch_size_per_gpu {batch_size} "
            f"* world_size {world_size} = {required}; with drop_last=True at least "
            "one rank would receive no batch. Reduce batch_size or generate more data.")


def setup_backend(cfg):
    """TF32 and cuDNN autotuning. The input shape is fixed, so benchmarking pays off."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark


def amp_dtype_of(name: str):
    """Autocast dtype; ``'off'`` returns None, meaning pure FP32."""
    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None, "": None}
    if name not in table:
        raise ValueError(f"amp must be bf16, fp16 or off, got {name!r}")
    return table[name]


def rotate_ckpt(savedir: str, keep_last_n: int, protect: str = ""):
    """Keep only the newest ``keep_last_n`` checkpoints; never delete ``protect``."""
    if keep_last_n <= 0:
        return
    found = []
    for fn in os.listdir(savedir):
        m = re.fullmatch(r"ckpt_step_(\d+)\.pt", fn)
        if m:
            found.append((int(m.group(1)), fn))
    found.sort()
    keep = os.path.basename(protect) if protect else ""
    for _, fn in found[:-keep_last_n]:
        if fn == keep:
            continue
        try:
            os.remove(os.path.join(savedir, fn))
            print(f"  rotate: removed old checkpoint {fn}")
        except OSError as e:
            print(f"  rotate: could not remove {fn} ({e})")


def build_fm(name: str, sigma: float):
    variants = {
        "otcfm": ExactOptimalTransportConditionalFlowMatcher,
        "icfm": ConditionalFlowMatcher,
        "fm": TargetConditionalFlowMatcher,
        "si": VariancePreservingConditionalFlowMatcher,
    }
    if name not in variants:
        raise ValueError(f"unknown flow matching variant {name!r}; "
                         f"choose from {sorted(variants)}")
    return variants[name](sigma=sigma)


def warmup_lr(step, warmup):
    return min(step, warmup) / max(1, warmup)


def infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def init_distributed():
    """Set up single-GPU or torchrun multi-GPU execution."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP/NCCL training requires CUDA, but it is unavailable")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        run_device = torch.device("cuda", local_rank)
    else:
        local_rank, rank = 0, 0
        run_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, run_device


def unwrap_model(net):
    return net.module if isinstance(net, DDP) else net


def ema_decay_at(step: int, target: float, warmup: bool) -> float:
    """EMA decay for the current step.

    ``ema_net`` starts as a copy of the randomly initialised network. Expanding
    the recursion with ``theta_t = theta_0 + delta_t`` gives
    ``ema_T = theta_0 + (1 - decay^T) * <delta>``: the *learned displacement* is
    shortened by ``1 - decay^T`` (no random noise is mixed in). At
    ``decay = 0.9999`` that factor is only 0.393 after 5k steps and 0.632 after
    10k, so early EMA snapshots systematically understate progress.
    ``1 - 1/(step+1)`` makes the decay 0 at step 0 (the EMA equals the current
    weights, with nothing left of the initialisation) and rises monotonically to
    ``target``.
    """
    if not warmup:
        return target
    return min(target, 1.0 - 1.0 / (step + 1))


@torch.no_grad()
def zero_pred_baseline(data_iter, device, n_batches=8):
    """``L_FM`` of a network that always predicts zero: ``E[(x1-x0)^2] = E[x1^2] + 1``.

    A loss sitting at this value means the network outputs zero and has learned
    nothing, which is a different failure from "the loss will not come down".
    A loss approaching 0 is the opposite warning: a condition is leaking the
    answer and the network has found a shortcut.

    Samples must come from ``data_iter`` rather than a fresh ``iter(loader)``:
    with ``persistent_workers=True`` a second iterator resets the training one.
    All ranks call this so that DDP data progress stays aligned.
    """
    s = 0.0
    for _ in range(n_batches):
        x1 = next(data_iter)[0]
        s += float((x1.to(device) ** 2).mean())
    ex1sq = s / max(n_batches, 1)
    return ex1sq + 1.0, ex1sq


@torch.no_grad()
def ema_update(net, ema_net, decay):
    for p, ep in zip(net.parameters(), ema_net.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for b, eb in zip(net.buffers(), ema_net.buffers()):
        eb.copy_(b)


def parse_cfg() -> TrainCfg:
    cfg = TrainCfg()
    ap = argparse.ArgumentParser(description="Stage 4: conditional flow matching training")
    ap.add_argument("--data-dir", default=cfg.data_dir)
    ap.add_argument("--out-dir", default=cfg.out_dir)
    ap.add_argument("--resume", default=cfg.resume,
                    help="checkpoint to resume from; empty means train from scratch")
    # Accept the flag some torchrun versions pass; the rank is read from the
    # environment regardless.
    ap.add_argument("--local-rank", "--local_rank", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("overrides", nargs="*", help="key=value overrides for TrainCfg")
    args = ap.parse_args()
    cfg.data_dir, cfg.out_dir, cfg.resume = args.data_dir, args.out_dir, args.resume
    return apply_overrides(cfg, args.overrides)


def main():
    cfg = parse_cfg()
    distributed, rank, local_rank, world_size, run_device = init_distributed()
    is_main = rank == 0
    seed_everything(cfg.seed + rank)
    setup_backend(cfg)
    amp_dtype = amp_dtype_of(cfg.amp)
    use_amp = amp_dtype is not None and run_device.type == "cuda"
    savedir = os.path.join(cfg.out_dir, cfg.model)
    if is_main:
        os.makedirs(savedir, exist_ok=True)
    if distributed:
        dist.barrier()

    if is_main:
        print("==== cfm training ====")
        for k in ("data_dir", "image_size", "num_classes", "model", "lr",
                  "total_steps", "overfit"):
            print(f"  {k}: {getattr(cfg, k)}")
        print(f"  distributed: {distributed}  world_size: {world_size}")
        print(f"  batch_size_per_gpu: {cfg.batch_size}  "
              f"global_batch_size: {cfg.batch_size * world_size}")
        print(f"  amp: {cfg.amp} (enabled={use_amp})  tf32: {cfg.tf32}  "
              f"channels_last: {cfg.channels_last}")
        print(f"  conditions: rms_smooth=({cfg.rms_smooth_tau}, {cfg.rms_smooth_x}) "
              f"n_wells={cfg.n_wells} well_width={cfg.well_width} "
              f"well_interp={cfg.well_interp} use_imaging={cfg.use_imaging} "
              f"-> {active_cond_channels(cfg)} channels")
        print(f"  loss: L = {cfg.lambda_fm}*L_FM + {cfg.lambda_well}*L_well")
        if torch.cuda.is_available():
            print(f"  gpu: {torch.cuda.get_device_name(local_rank)}")

    ds = ShardConditionDataset.from_cfg(cfg, cfg.data_dir)
    if is_main:
        print(f"  dataset size: {len(ds)}")
    validate_dataset_batch_size(len(ds), cfg.batch_size, world_size)

    sampler = (DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=cfg.seed, drop_last=True)
               if distributed else None)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed + rank)
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                        generator=loader_generator,
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
    data_iter = infinite(loader, sampler)

    net = build_unet(cfg).to(run_device)
    if cfg.channels_last:
        net = net.to(memory_format=torch.channels_last)
    if distributed:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    ema_net = copy.deepcopy(unwrap_model(net))
    ema_net.requires_grad_(False)
    if is_main:
        print(f"  params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")

    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_lr(s, cfg.warmup))
    # bf16 needs no loss scaling; only fp16 does.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and cfg.amp == "fp16"))
    FM = build_fm(cfg.model, cfg.sigma)

    start_step = 0
    if cfg.resume:
        if not os.path.isfile(cfg.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {cfg.resume}")
        ck = torch.load(cfg.resume, map_location=run_device, weights_only=False)
        unwrap_model(net).load_state_dict(ck["net_model"])
        ema_net.load_state_dict(ck["ema_model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_step = int(ck["step"]) + 1
        if is_main:
            print(f"  resumed from {cfg.resume} @ step {ck['step']} -> continuing "
                  f"at {start_step}")
            old_bs = ck.get("cfg", {}).get("batch_size")
            if old_bs is not None and old_bs != cfg.batch_size:
                print(f"  note: checkpoint used batch_size={old_bs}, this run uses "
                      f"{cfg.batch_size}; Adam carries the old batch statistics, so "
                      "expect some loss jitter for the first steps")
    if distributed:
        dist.barrier()

    # A fixed batch of conditions for the periodic monitoring figure. Every rank
    # consumes the same iteration: calling iter(loader) again would reset the
    # training iterator under persistent_workers, and would desynchronise ranks.
    vis_x1 = vis_cond = vis_y = None
    vis_batch = next(data_iter)
    if is_main:
        vis_x1, vis_cond, vis_y = vis_batch
        vis_cond = vis_cond[:cfg.sample_n].to(run_device)
        vis_y = vis_y[:cfg.sample_n].to(run_device)
        vis_x1 = vis_x1[:cfg.sample_n]
    del vis_batch

    log_path = os.path.join(savedir, "loss_log.csv")
    base, ex1sq = zero_pred_baseline(data_iter, run_device)
    if is_main:
        print(f"  E[x1^2]={ex1sq:.4f} -> L_FM of an all-zero prediction = {base:.4f}")
        print(f"  A loss stuck near {base:.3f} means the network outputs zero; a loss "
              "approaching 0 means a condition is leaking the answer.")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"# zero_pred_baseline={base:.6f} E[x1^2]={ex1sq:.6f} "
                        f"lambda_fm={cfg.lambda_fm} lambda_well={cfg.lambda_well}\n")
                f.write("step,loss,loss_ema,lr,ema_decay,l_well,l_total\n")
    loss_ema = None

    with trange(start_step, cfg.total_steps, dynamic_ncols=True,
                disable=not is_main) as pbar:
        for step in pbar:
            optim.zero_grad(set_to_none=True)
            x1, cond, y = next(data_iter)
            x1 = x1.to(run_device, non_blocking=True)
            cond = cond.to(run_device, non_blocking=True)
            y = y.to(run_device, non_blocking=True)
            x0 = torch.randn_like(x1)

            # The OT coupling is computed in FP32; low precision would perturb
            # the transport plan itself. The conditions come back reordered to
            # match xt/ut, which is all the constraint below needs.
            t, xt, ut, cond_aligned, y_aligned, _ = sample_flow_batch(
                FM, cfg.model, x0, x1, cond, y)
            net_in = torch.cat([xt, cond_aligned], dim=1)
            if cfg.channels_last:
                net_in = net_in.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                vt = net(t, net_in, y=y_aligned)
            # Accumulate the loss in FP32; bf16 has too few mantissa bits here.
            vt = vt.float()
            loss_fm = torch.mean((vt - ut) ** 2)

            parts = {}
            use_phys = cfg.lambda_well > 0
            logging_now = is_main and cfg.log_every > 0 and step % cfg.log_every == 0
            if use_phys or logging_now:
                # x1_hat = x_t + (1-t) * v_theta, from the forward pass just done.
                tb = t.view(-1, *([1] * (x1.dim() - 1)))
                x1_hat = xt + (1.0 - tb) * vt
            if use_phys:
                parts = physics_loss(x1_hat, cond_aligned, cfg)
            loss = combine_losses(loss_fm, parts, cfg)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)                  # must unscale before clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                optim.step()
            sched.step()
            d = ema_decay_at(step, cfg.ema_decay, cfg.ema_warmup)
            ema_update(unwrap_model(net), ema_net, d)

            # This .item() comes after optim.step(), a natural synchronisation
            # point, so it costs the least.
            lv = loss_fm.item()
            loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
            if logging_now:
                # Log the well consistency even when lambda_well is zero, so
                # that constrained and control runs stay directly comparable.
                l_well = (float(parts["well"]) if "well" in parts
                          else physics_metrics(x1_hat, cond_aligned)["well"])
                l_tot = float(loss)
                pbar.set_description(
                    f"L_FM {lv:.4f}(ema {loss_ema:.4f}) well {l_well:.4f}" if parts
                    else f"L_FM {lv:.4f}(ema {loss_ema:.4f})")
                with open(log_path, "a") as f:
                    f.write(f"{step},{lv:.6f},{loss_ema:.6f},"
                            f"{sched.get_last_lr()[0]:.3e},{d:.6f},"
                            f"{l_well:.6f},{l_tot:.6f}\n")
            elif is_main:
                pbar.set_description(f"L_FM {lv:.4f} (ema {loss_ema:.4f})")

            if (is_main and cfg.save_step > 0
                    and (step % cfg.save_step == 0 or step + 1 == cfg.total_steps)):
                _snapshot(ema_net, vis_cond, vis_y, vis_x1, cfg, savedir, step)
                torch.save({
                    "net_model": unwrap_model(net).state_dict(),
                    "ema_model": ema_net.state_dict(),
                    "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "step": step, "cfg": cfg.to_dict(),
                }, os.path.join(savedir, f"ckpt_step_{step}.pt"))
                print(f"  saved checkpoint @ step {step}")
                # Write the new checkpoint before deleting old ones, so a crash
                # mid-rotation still leaves one complete file behind.
                rotate_ckpt(savedir, cfg.keep_last_n, protect=cfg.resume)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


@torch.no_grad()
def _snapshot(ema_net, cond, y, x1_true, cfg, savedir, step):
    gen = euler_sample(ema_net, cond, y, cfg.ode_steps, cfg.image_size).cpu()
    grid = torch.cat([x1_true, gen], dim=0)          # top row truth, bottom row generated
    save_image((grid / 2 + 0.5).clamp(0, 1),
               os.path.join(savedir, f"sample_step_{step}.png"), nrow=cfg.sample_n)
    ema_net.train()


if __name__ == "__main__":
    main()
