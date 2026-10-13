"""cavalab ECG survival benchmark, Ribeiro ResNet + fusion head (DeepSurv, no demographics).

The published pipeline (opt/ecg-survival-benchmark) z-scores each lead with
statistics stored in the checkpoint (NM, NS; norm type nchW), then zero-pads
the 7 s, 400 Hz record by 648 samples on each side to 4096. Both happen
after the contract's crop, so they live here. The checkpoint is converted
once by ``convert`` into a plain state_dict holding the module weights and
the two statistic vectors as buffers.
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

# Authors' repositories: /opt on the reference server, ${MIVAL_HOME}/code in a
# site bundle.
CODE_ROOT = os.environ["MIVAL_HOME"] + "/code" if "MIVAL_HOME" in os.environ else "/opt"
SURVIVAL_CODE = CODE_ROOT + "/ecg-survival-benchmark/Modeling"
FILTERS = [64, 128, 196, 256, 320]
SEQ_LENGTHS = [4096, 1024, 256, 64, 16]
PAD = 648


def _resnet():
    if SURVIVAL_CODE not in sys.path:
        sys.path.insert(0, SURVIVAL_CODE)
    from MODELS.Ribeiro_Support import ResNet1d

    return ResNet1d(input_dim=(12, 4096), blocks_dim=list(zip(FILTERS, SEQ_LENGTHS)),
                    n_classes=0, kernel_size=17, dropout_rate=0.8)


class DeepSurvCode15(nn.Module):
    """FusionModel of the benchmark: ECG features [+ covariate MLP] -> fusion MLP -> outputs.

    DeepSurv (no demographics): 1 output, no covariates. MTLR with
    demographics: 100 outputs, covariates (Age in years, Is_Male 0/1, raw)
    through ``cov_layers`` Linear+ReLU of width ``cov_dim``.
    """

    def __init__(self, fusion_dim=128, fusion_layers=3, n_outputs=1, n_covariates=0, cov_layers=0, cov_dim=32):
        super().__init__()
        self.ECG_Model = _resnet()
        covariate = []
        width_z = n_covariates
        for _ in range(cov_layers):
            covariate += [nn.Linear(width_z, cov_dim), nn.ReLU()]
            width_z = cov_dim
        self.covariate_module_list = nn.ModuleList(covariate)
        layers = []
        width = FILTERS[-1] * SEQ_LENGTHS[-1] + (width_z if cov_layers else 0)
        for _ in range(fusion_layers):
            layers += [nn.Linear(width, fusion_dim), nn.ReLU()]
            width = fusion_dim
        layers.append(nn.Linear(width, n_outputs))
        self.fusion_module_list = nn.ModuleList(layers)
        self.register_buffer("lead_mean", torch.zeros(12, 1))
        self.register_buffer("lead_std", torch.ones(12, 1))

    def forward(self, x, z=None):
        x = (x - self.lead_mean) / self.lead_std
        x = F.pad(x, (PAD, PAD), "constant", 0.0)
        out = self.ECG_Model(x)
        if len(self.covariate_module_list):
            b = z
            for layer in self.covariate_module_list:
                b = layer(b)
            out = torch.cat((out, b), dim=1)
        for layer in self.fusion_module_list:
            out = layer(out)
        return out


def convert(source, target):
    """Published checkpoint -> state_dict for DeepSurvCode15 (run once, on the server)."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = {key: value for key, value in checkpoint["model_state_dict"].items()
             if not (key.startswith("covariate_module_list") and type(value).__name__ == "UninitializedParameter")}
    assert checkpoint["NT"] == "nchW", checkpoint["NT"]
    state["lead_mean"] = torch.tensor([float(v) for v in checkpoint["NM"]]).reshape(12, 1)
    state["lead_std"] = torch.tensor([float(v) for v in checkpoint["NS"]]).reshape(12, 1)
    torch.save(state, target)
