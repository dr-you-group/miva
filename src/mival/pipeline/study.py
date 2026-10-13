"""Loading the study spec layer (spec §3.2).

    studies/<study_id>/study.yaml   # references the specs below + seed + site
      cohort/…  preprocess/…  eval/…

``study.yaml`` carries the study-wide settings and, for each stage, either an
inline spec or a path to one. Only the stage's own spec goes into its
``config_hash``: if the whole file did, editing the evaluation grid would
invalidate every preprocessing run that never read it.

Settings that belong to the site rather than the study (site name, DBMS,
schema, connection file, DICOM root, cohort name) can live in one file that
every study points at with ``site_config``. Its ``site`` and per-stage keys
are merged into the study before anything reads it, so each stage's spec and
config_hash are exactly what they would be with the values written inline. A
key set in both places is an error: the study would otherwise silently run
with whichever value won.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from mival.placeholders import expand

_REQUIRED = ("study_id", "site")


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"reading {path} requires PyYAML. Install it, or write the spec as JSON."
            ) from exc
        body = yaml.safe_load(text)
    else:
        body = json.loads(text)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(body).__name__}")
    return body


@dataclass(frozen=True)
class Study:
    """A loaded study.yaml with its stage specs resolved."""

    study_id: str
    site: str
    root: Path
    seed: Any
    stages: Mapping[str, Any]

    def stage_spec(self, stage: str) -> Dict[str, Any]:
        """The spec for one stage, always as a mapping.

        A stage with no entry gets ``{}`` rather than an error: a stage whose
        behaviour is fully determined by its inputs has nothing to configure,
        and it still gets a stable config_hash from those inputs.
        """
        spec = self.stages.get(stage, {})
        if isinstance(spec, str):
            spec = _load_mapping(self.root / spec)
        if not isinstance(spec, dict):
            raise ValueError(
                f"spec for stage {stage!r} must be a mapping or a path to one, got "
                f"{type(spec).__name__}"
            )
        return spec


def _merge_site_config(body: Dict[str, Any], target: Path) -> Dict[str, Any]:
    """``body`` with the file named by its ``site_config`` merged in (see the module docstring)."""
    reference = body.pop("site_config", None)
    if reference is None:
        return body
    path = Path(expand(str(reference), target))
    if not path.is_absolute():
        path = target.parent / path
    if not path.is_file():
        raise ValueError(f"{target}: site_config {path} does not exist")
    site = expand(_load_mapping(path), path)
    if "site" in site:
        if "site" in body:
            raise ValueError(f"{target}: site is set both here and in {path}")
        body["site"] = site.pop("site")
    stages = body.setdefault("stages", {})
    for stage, settings in site.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: {stage!r} must be a mapping of that stage's settings")
        spec = stages.setdefault(stage, {})
        if not isinstance(spec, dict):
            raise ValueError(f"{target}: stage {stage!r} must be inline to take settings from {path}")
        clash = sorted(set(spec) & set(settings))
        if clash:
            raise ValueError(f"{target}: {stage}.{', '.join(clash)} set both here and in {path}")
        spec.update(settings)
    return body


def load_study(path: Union[str, Path]) -> Study:
    target = Path(path)
    if target.is_dir():
        target = target / "study.yaml"
    body = _merge_site_config(_load_mapping(target), target)
    for key in _REQUIRED:
        if key not in body:
            raise ValueError(f"{target}: {key} is required")
    stages = body.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError(f"{target}: 'stages' must be a mapping of stage name to spec")
    return Study(
        study_id=str(body["study_id"]),
        site=str(body["site"]),
        root=target.parent,
        seed=body.get("seed"),
        stages=stages,
    )
