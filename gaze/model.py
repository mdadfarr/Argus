"""Inference-only port of the gaze model from https://github.com/pperle/gaze-tracking (MIT).

Three deliberate deviations from upstream, all inference-only concerns:

  * `nn.Module`, not `LightningModule`. Upstream subclasses pytorch_lightning
    purely for its training loop. Nothing in the forward path needs it, and it
    is a large dependency to ship in an app bundle for code we never call.
    Checkpoints still load: Lightning writes plain `state_dict` tensors under a
    "state_dict" key, which `load_checkpoint` below unwraps.

  * `models.vgg16(weights=None)`. Upstream passes `pretrained=True`, which was
    removed in torchvision 0.15. It would also pull ~500 MB of ImageNet weights
    at import time -- pointless here, because the checkpoint overwrites every
    one of those tensors before the first forward pass.

  * No CUDA assumption. `pick_device()` prefers Apple's MPS backend, which is
    what this actually runs on.

Verified against torch 2.13: 5.0M parameters, ~29 ms/frame on plain CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)

# Fixed by the architecture -- these are what mpiifacegaze normalization emits.
FACE_INPUT_HW = (96, 96)
EYE_INPUT_HW = (64, 96)

# Upstream trains on 15 MPIIFaceGaze participants, each also mirrored, and adds
# a learned per-participant (pitch, yaw) offset to the subject-independent
# prediction. At inference against a stock checkpoint you are borrowing some
# stranger's offset; see `calibrate_subject_bias` for why that matters.
N_SUBJECT_BIASES = 15 * 2


class SELayer(nn.Module):
    """Squeeze-and-Excitation block: learned per-channel reweighting."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GazeModel(nn.Module):
    """Predicts (pitch, yaw) in radians in the *normalized* camera space.

    Layer shapes must match upstream exactly or the checkpoint will not load,
    so this is a transcription, not a redesign. The dilation schedules that
    look like typos (`(4, 5)` and `(5, 11)` in the eye path against `(5, 5)`
    and `(11, 11)` in the face path) are upstream's -- the eye input is 64x96,
    not square, so the asymmetry is deliberate.
    """

    def __init__(self) -> None:
        super().__init__()

        self.subject_biases = nn.Parameter(torch.zeros(N_SUBJECT_BIASES, 2))

        self.cnn_face = nn.Sequential(
            models.vgg16(weights=None).features[:9],
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid", dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid", dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="valid", dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="valid", dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            models.vgg16(weights=None).features[:9],
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid", dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid", dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="valid", dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="valid", dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            SELayer(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        person_idx: torch.Tensor,
        full_face: torch.Tensor,
        right_eye: torch.Tensor,
        left_eye: torch.Tensor,
    ) -> torch.Tensor:
        out_fc_face = self.fc_face(self.cnn_face(full_face))

        # Both eyes go through the *same* weights. Upstream's stated trick is
        # that the eye crops are mirrored so both noses point the same way, so
        # the shared extractor never has to spend capacity on left-vs-right.
        out_cnn_eye = torch.cat((self.cnn_eye(right_eye), self.cnn_eye(left_eye)), dim=1)
        out_fc_eye = self.fc_eye(self.cnn_eye2fc(out_cnn_eye))

        t_hat = self.fc_eyes_face(torch.cat((out_fc_face, out_fc_eye), dim=1))
        return t_hat + self.subject_biases[person_idx].squeeze(1)


def pick_device(prefer: str | None = None) -> torch.device:
    """MPS on Apple Silicon, else CPU. CUDA is included only for completeness.

    Note MPS is not automatically the win it looks like: this model is 5M
    parameters on 96x96 inputs, small enough that kernel-launch overhead can
    eat the gain. Benchmark before assuming -- `tools/gaze_spike.py --benchmark`
    reports both.
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> GazeModel:
    """Load a pperle `.ckpt` (e.g. `p00.ckpt`) into the ported module.

    Raises on unexpected/missing keys rather than warning. A silently
    half-loaded model still runs and still emits confident-looking angles --
    it is the exact failure that would waste an evening of A5 accuracy work
    chasing a calibration problem that was never there.
    """
    device = device or pick_device()
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob

    model = GazeModel()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {Path(path).name} does not match the ported architecture.\n"
            f"  missing:    {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected: {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )

    model.eval().to(device)
    # eval() matters more than usual here: the net is dense with BatchNorm and
    # carries two Dropout(0.5) layers. Left in train mode it would both leak
    # batch statistics and randomly zero half the fused features every frame.
    log.info("loaded gaze checkpoint %s onto %s", Path(path).name, device)
    return model
