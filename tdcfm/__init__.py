"""Time-domain conditional flow matching for seismic initial velocity models.

The package is organised as a five-stage pipeline; each stage writes a
self-describing directory (`manifest.json` plus HDF5/npz shards) that the next
stage consumes:

    tdcfm.synth        stage 1  synthetic depth-domain velocity models
    tdcfm.timeconv     stage 2  vertical depth-to-time conversion
    tdcfm.conditions   stage 3  forward modelling of the network conditions
    tdcfm.models       stage 4  conditional UNet velocity field + Euler sampler
    tdcfm.training     stage 4  dataloader, OT-CFM objective, well consistency loss
    tdcfm.inference    stage 4  conditional sampling from a checkpoint
    tdcfm.evaluation   stage 5  metrics, trivial baselines, figures
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
