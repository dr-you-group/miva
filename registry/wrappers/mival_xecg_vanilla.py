"""xECG checkpoint (sLSTM ``cuda`` layout) -> sLSTM ``vanilla`` layout.

The xECG weights were saved from the ``cuda`` sLSTM backend. The ``vanilla``
backend (plain torch, what the card's ``runtime.windows`` selects) keeps the
same parameters in a different internal layout (xlstm
``blocks/slstm/cell.py``: ``sLSTMCell_cuda`` / ``sLSTMCell_vanilla``
``_recurrent_kernel_ext2int`` and ``_bias_ext2int``). With ``NH`` heads,
``G`` gates and head size ``D``, both start from the external layout
``R[NH, D_in, G, D_out]`` and ``b[NH, G, D]``:

- cuda    ``_recurrent_kernel_`` = R.reshape(NH, D, G*D)                  -> [NH, D, G*D]
  vanilla ``_recurrent_kernel_`` = R.permute(0, 2, 3, 1).reshape(NH, G*D, D) -> [NH, G*D, D]
  so vanilla = cuda.permute(0, 2, 1) (the author's commented-out conversion).
- cuda    ``_bias_`` = b flattened in (NH, G, D) order
  vanilla ``_bias_`` = b.permute(1, 0, 2) flattened, i.e. (G, NH, D) order
  so vanilla = cuda.reshape(NH, G, D).permute(1, 0, 2).reshape(-1).

Converting the kernel without the bias reorder loads strictly but gives
other outputs (adaptations L2-4, cosine 0.32). The card names this function
as ``runtime.windows.state_dict_transform``; the torch adapter calls it on the
loaded state dict before the strict load. Canonical copy: mi-va
``registry/site/wrappers/`` (scripts/site/sync_to_shared.sh copies it into the
shared copy's ``registry/wrappers/``, which the bundle carries at
``${MIVAL_HOME}/miva/registry/wrappers/``); the research card reads the same
file from ``/data/mi-val/models/_wrappers`` (copy of ``registry/wrappers/``).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

KERNEL_SUFFIX = "slstm_cell._recurrent_kernel_"
BIAS_SUFFIX = "slstm_cell._bias_"


def kernel_to_vanilla(kernel: Any) -> Any:
    """cuda ``[NH, D, G*D]`` -> vanilla ``[NH, G*D, D]``."""
    if kernel.ndim != 3 or kernel.shape[2] % kernel.shape[1] or kernel.shape[2] == kernel.shape[1]:
        raise ValueError(f"sLSTM recurrent kernel of shape {tuple(kernel.shape)} is not the cuda layout "
                         "[heads, head_dim, gates*head_dim] with more than one gate")
    return kernel.permute(0, 2, 1).contiguous()


def bias_to_vanilla(bias: Any, num_heads: int, num_gates: int) -> Any:
    """cuda ``(NH, G, D)``-ordered flat bias -> vanilla ``(G, NH, D)``-ordered flat bias."""
    if bias.ndim != 1 or bias.shape[0] % (num_heads * num_gates):
        raise ValueError(f"sLSTM bias of shape {tuple(bias.shape)} does not split into "
                         f"{num_heads} heads x {num_gates} gates")
    head_dim = bias.shape[0] // (num_heads * num_gates)
    return bias.reshape(num_heads, num_gates, head_dim).permute(1, 0, 2).reshape(-1).contiguous()


def to_vanilla(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The state dict with every sLSTM cell's kernel and bias in the vanilla layout."""
    out = dict(state)
    kernels = [key for key in out if key.endswith(KERNEL_SUFFIX)]
    if not kernels:
        raise ValueError(f"no {KERNEL_SUFFIX} in the state dict; is this an xLSTM checkpoint?")
    for key in kernels:
        kernel = out[key]
        num_heads, head_dim = int(kernel.shape[0]), int(kernel.shape[1])
        num_gates = int(kernel.shape[2]) // head_dim if head_dim else 0
        bias_key = key[: -len(KERNEL_SUFFIX)] + BIAS_SUFFIX
        if bias_key not in out:
            raise ValueError(f"{key} has no matching {bias_key}")
        out[key] = kernel_to_vanilla(kernel)
        if int(out[bias_key].shape[0]) != num_heads * num_gates * head_dim:
            raise ValueError(f"{bias_key} has {int(out[bias_key].shape[0])} values; {key} implies "
                             f"{num_heads}x{num_gates}x{head_dim}")
        out[bias_key] = bias_to_vanilla(out[bias_key], num_heads, num_gates)
    return out
