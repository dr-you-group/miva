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

A few site keys say how to run rather than what to compute (``gpus``: which
GPUs the models stage may spread its arms over; ``workers``: how many CPU
processes the evaluation stage spreads its slices over). They are kept out of
every stage spec and land in ``Study.execution``, so a site with eight GPUs
and one with none get the same config_hash for the same study.

A removed stage key (``retrieve.local_path_root``) with a value is refused
when the study loads, in the study or in its site config. Written as null,
the value the site.yaml template once carried, it is dropped, so the stage's
config_hash is what it is without the line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from mival.placeholders import expand

_REQUIRED = ("study_id", "site")

#: Site-config keys that set how the study runs, never what it computes.
EXECUTION_KEYS = ("gpus", "workers")
#: (stage, key) -> why the key is no longer read.
REMOVED_KEYS = {
    ("retrieve", "local_path_root"): (
        "this setting was removed. The CDM image_occurrence.local_path is read as it is "
        "stored; delete the line and make that path readable on this server (README 4.2)"),
}


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
    execution: Mapping[str, Any] = field(default_factory=dict)

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


def _drop_removed_keys(stages: Mapping[str, Any], source: Path) -> None:
    """Refuse a removed key that has a value; drop it when it is null (see the module docstring)."""
    for (stage, key), why in REMOVED_KEYS.items():
        spec = stages.get(stage)
        if not isinstance(spec, dict) or key not in spec:
            continue
        if spec[key] is not None:
            raise ValueError(f"{source}: {stage}.{key}: {why}")
        del spec[key]


def _merge_site_config(body: Dict[str, Any], target: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``body`` with the file named by its ``site_config`` merged in, and the execution keys.

    See the module docstring for both.
    """
    reference = body.pop("site_config", None)
    if reference is None:
        return body, {}
    path = Path(expand(str(reference), target))
    if not path.is_absolute():
        path = target.parent / path
    if not path.is_file():
        raise ValueError(f"{target}: site_config {path} does not exist")
    site = expand(_load_mapping(path), path)
    _drop_removed_keys(site, path)
    if "site" in site:
        if "site" in body:
            raise ValueError(f"{target}: site is set both here and in {path}")
        body["site"] = site.pop("site")
    execution = {key: site.pop(key) for key in EXECUTION_KEYS if key in site}
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
    return body, execution


def load_study(path: Union[str, Path]) -> Study:
    target = Path(path)
    if target.is_dir():
        target = target / "study.yaml"
    body = _load_mapping(target)
    if isinstance(body.get("stages"), dict):
        _drop_removed_keys(body["stages"], target)
    body, execution = _merge_site_config(body, target)
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
        execution=execution,
    )
