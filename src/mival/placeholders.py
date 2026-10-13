"""``${NAME}`` placeholders filled from the environment.

Locations that differ per site (weights, code, envs, the DB connection file)
are written relative to ``${MIVAL_HOME}`` in model cards and in the site
config, and the site sets that one variable. An unset name is an error rather
than an empty string, so a missing export cannot turn ``${MIVAL_HOME}/models``
into ``/models``.
"""

from __future__ import annotations

import os
import re
from typing import Any

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(value: Any, where: Any) -> Any:
    """``value`` with every ``${NAME}`` in its strings replaced; ``where`` names the source in errors."""
    if isinstance(value, dict):
        return {key: expand(item, where) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, where) for item in value]
    if isinstance(value, str):

        def substitute(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"{where}: ${{{name}}} is used but {name} is not set")
            return os.environ[name]

        return _PLACEHOLDER.sub(substitute, value)
    return value
