"""Singstad ECG-age InceptionTime (PhysioNet 2021 data), 12 leads, 100 Hz, 10 s.

The repository ships weights only; ``build`` recreates the architecture with
the authors' ``build_model((1000, 12), 1)``. Training inputs were raw digital
values (uV) put through ``pad_sequences``, which casts to int32, so the same
truncation is applied in front of the network.
"""

import os
import sys

# Authors' repositories: /opt on the reference server, ${MIVAL_HOME}/code in a
# site bundle.
CODE_ROOT = os.environ["MIVAL_HOME"] + "/code" if "MIVAL_HOME" in os.environ else "/opt"
SINGSTAD_CODE = CODE_ROOT + "/ECG-age"


def build(samples=1000, leads=12):
    import tensorflow as tf

    if SINGSTAD_CODE not in sys.path:
        sys.path.insert(0, SINGSTAD_CODE)
    from src.models.models import build_model

    core = build_model((samples, leads), 1)
    inputs = tf.keras.layers.Input((samples, leads))
    truncated = tf.keras.layers.Lambda(lambda x: tf.cast(tf.cast(x, tf.int32), tf.float32))(inputs)
    model = tf.keras.models.Model(inputs, core(truncated))
    # load_weights on the outer model would look for the inner layers by the
    # outer topology; the weights belong to ``core``.
    model.load_weights = core.load_weights
    return model
