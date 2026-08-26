"""Baseline D (OPTIONAL): RDAH-Net adapter.

RDAH-Net (Remote Sensing 2026, 18(7):1024) is the published reference that this
project's hypothesis is modelled on: frozen Depth Anything V2 as a relative-depth
prior + a small (~5.4 M param) MobileNetV2/MobileViT + CBAM + bidirectional
cross-attention head predicting nDSM. Code (MIT) and checkpoints+data (Figshare)
are public:
    code      : https://github.com/Elenairene/RDAH-Net
    data/ckpt : https://doi.org/10.6084/m9.figshare.31986864   (Figshare)
    paper     : https://www.mdpi.com/2072-4292/18/7/1024

BOUNDED-EFFORT POLICY (per the experiment spec): spend <= ~30 min trying to run
the authors' code/checkpoint. If it does not run quickly (dependency drift,
undocumented single-image inference path, checkpoint format), DO NOT let it block
Phase 1 -- document the attempt and skip. Our Baseline C already tests the SAME
principle (learned depth->height fusion) with our own small head; RDAH-Net is a
"can we reproduce the published reference?" bonus, not a dependency.

IMPORTANT: We must NOT claim RDAH-Net's published accuracy as ours. This adapter,
if wired, reports numbers WE measured on OUR split -- nothing inherited.

To enable: git clone the repo next to this project, download a checkpoint from
Figshare, then set models.rdah_net.repo_dir and checkpoint_path in the config and
implement the two TODOs below against the authors' actual module names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .base import HeightEstimator, Sample


class RDAHNetAdapter(HeightEstimator):
    name = "D_rdah_net"
    metric = True

    def __init__(self, repo_dir: str | None = None, checkpoint_path: str | None = None,
                 device: str | None = None):
        self.repo_dir = repo_dir
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._net = None

    def available(self) -> bool:
        return bool(
            self.repo_dir and self.checkpoint_path
            and Path(self.repo_dir).exists() and Path(self.checkpoint_path).exists()
        )

    def fit(self, train_samples: Iterable[Sample]) -> "RDAHNetAdapter":
        # Reference is pretrained; we run it as-is (inference-only) on OUR test set.
        return self

    def _load(self):
        if self._net is not None:
            return
        if not self.available():
            raise RuntimeError(
                "RDAH-Net not available. Clone https://github.com/Elenairene/RDAH-Net, "
                "download a checkpoint from Figshare (10.6084/m9.figshare.31986864), and "
                "set repo_dir + checkpoint_path in the config. See TODOs in rdah_net.py."
            )
        import sys, torch
        sys.path.insert(0, self.repo_dir)
        # TODO(1): import the authors' model class (inspect their train.py/test.py),
        #   e.g. `from model import RDAHNet` -- exact name per their repo.
        # TODO(2): build it, load state_dict from self.checkpoint_path, .eval().
        raise NotImplementedError(
            "Wire RDAH-Net model construction here (TODO 1/2). Left unimplemented so "
            "Phase 1 is never blocked by an external repo; Baseline C tests the same idea."
        )

    def predict(self, sample: Sample) -> np.ndarray:  # pragma: no cover
        self._load()
        raise NotImplementedError
