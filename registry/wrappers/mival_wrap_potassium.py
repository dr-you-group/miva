"""Kardio-Net 12-lead potassium (Cedars-Sinai) and von Bachmann potassium (Uppsala).

Kardio-Net: the authors remove baseline wander (two median filters), denoise
each lead with a discrete wavelet transform, then z-score every lead with the
mean and SD of the *target dataset* (L2-2). The statistics are cohort
constants here, fitted once on a MIMIC sample by ``fit_kardionet_stats`` and
stored as buffers, as the ledger's L2-2 interim measure describes.

von Bachmann: 8 leads (I, II, V1-V6) at 400 Hz, an elliptic 0.8 Hz high-pass
applied to the 10 s record before it is centred in 4096 samples. The contract
pads first, so the filter runs on the unpadded centre here.
"""

import os
import sys

import numpy as np
import torch
from torch import nn

# Authors' repositories: /opt on the reference server, ${MIVAL_HOME}/code in a
# site bundle.
CODE_ROOT = os.environ["MIVAL_HOME"] + "/code" if "MIVAL_HOME" in os.environ else "/opt"
KARDIONET_CODE = CODE_ROOT + "/hyperkalemia"
ELECTROLYTE_CODE = CODE_ROOT + "/ecg-electrolyte-regression"


def _load_file(name, path):
    """Import one author file under a unique module name (several repos call their package ``src``)."""
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _path(code):
    if code not in sys.path:
        sys.path.insert(0, code)


def _clean_one(record, fs=500):
    _path(KARDIONET_CODE)
    from utils.ecg_utils import remove_baseline_wander, wavelet_denoise_signal

    signal = remove_baseline_wander(np.ascontiguousarray(record.T), sampling_frequency=fs)
    for lead in range(signal.shape[1]):
        signal[:, lead] = wavelet_denoise_signal(signal[:, lead])[: signal.shape[0]]
    return signal.T.astype(np.float32)


_POOL = None
#: The authors' per-record cleaning is ~0.18 s on one core; a process pool
#: keeps a 120k-record cohort to minutes. Spawned workers never touch CUDA.
POOL_WORKERS = 12


def _pool():
    global _POOL
    if _POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(POOL_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    return _POOL


def kardionet_clean(batch, fs=500):
    """(B, 12, T) mV -> baseline-removed, wavelet-denoised, as in preprocessing.py."""
    batch = np.asarray(batch, dtype=np.float32)
    if len(batch) < 4:
        return np.stack([_clean_one(record, fs) for record in batch]) if len(batch) else batch
    return np.stack(list(_pool().map(_clean_one, batch, [fs] * len(batch))))


class KardioNet12Lead(nn.Module):
    def __init__(self):
        super().__init__()
        _path(KARDIONET_CODE)
        from utils.models import EffNet

        self.m = EffNet(input_channels=12, output_neurons=1)
        self.register_buffer("lead_mean", torch.zeros(1, 12, 1))
        self.register_buffer("lead_std", torch.ones(1, 12, 1))

    def forward(self, x):
        clean = torch.from_numpy(kardionet_clean(x.detach().cpu().numpy())).to(x.device)
        return self.m((clean - self.lead_mean) / self.lead_std)


def fit_kardionet_stats(weights, tensors, target):
    """Write weights + per-lead mean/SD of cleaned ``tensors`` (N, 12, T) as one state_dict."""
    state = torch.load(weights, map_location="cpu", weights_only=False)
    clean = kardionet_clean(np.asarray(tensors, dtype=np.float32))
    state["lead_mean"] = torch.tensor(clean.mean(axis=(0, 2)), dtype=torch.float32).reshape(1, 12, 1)
    state["lead_std"] = torch.tensor(clean.std(axis=(0, 2)), dtype=torch.float32).reshape(1, 12, 1)
    torch.save(state, target)
    return state["lead_mean"].flatten().tolist(), state["lead_std"].flatten().tolist()


def _elliptic_baseline_sos(fs=400):
    from scipy import signal as sgn

    fc, fst, rp, rs = 0.8, 0.2, 0.5, 40
    wn, wst = fc / (fs / 2), fst / (fs / 2)
    order, _ = sgn.ellipord(wn, wst, rp, rs)
    return sgn.iirfilter(order, wn, rp, rs, btype="high", ftype="ellip", output="sos")


class VonBachmannPotassium(nn.Module):
    def __init__(self, signal_samples=4000, total_samples=4096):
        super().__init__()
        # resnet.py imports its own package as ``src.models``; give it that
        # package for the import, then drop it so no other repo's ``src`` is shadowed.
        saved = {key: sys.modules.pop(key) for key in list(sys.modules) if key == "src" or key.startswith("src.")}
        sys.path.insert(0, ELECTROLYTE_CODE)
        try:
            ResNet1d = _load_file("mival_ext_electrolyte_resnet", ELECTROLYTE_CODE + "/src/models/resnet.py").ResNet1d
        finally:
            sys.path.remove(ELECTROLYTE_CODE)
            for key in [key for key in sys.modules if key == "src" or key.startswith("src.")]:
                del sys.modules[key]
            sys.modules.update(saved)

        self.net = ResNet1d(input_dim=(8, total_samples),
                            blocks_dim=list(zip([64, 128, 196, 256, 320], [4096, 1024, 256, 64, 16])),
                            n_classes=1, kernel_size=17, dropout_rate=0.8)
        self.start = (total_samples - signal_samples) // 2
        self.stop = self.start + signal_samples
        self.sos = _elliptic_baseline_sos()

    def forward(self, x):
        from scipy import signal as sgn

        raw = x.detach().cpu().numpy()
        centre = sgn.sosfiltfilt(self.sos, raw[..., self.start:self.stop], padtype="constant", axis=-1)
        filtered = np.zeros_like(raw)
        filtered[..., self.start:self.stop] = centre
        return self.net(torch.from_numpy(filtered.astype(np.float32)).to(x.device))
