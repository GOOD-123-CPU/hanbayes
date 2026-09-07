"""Frozen configuration loading."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

CONFIG_RESOURCE = "configs/frozen.json"


def default_config_path() -> Path:
    """Path of the bundled frozen configuration file.

    In a source checkout this is ``<repo>/configs/frozen.json`` (two levels
    above the package); inside an installed wheel it is ``hanbayes/frozen.json``
    next to the package code.
    """

    package_dir = Path(__file__).resolve().parent
    source_layout = package_dir.parent / CONFIG_RESOURCE  # src layout
    if source_layout.exists():
        return source_layout
    return package_dir / "frozen.json"  # wheel layout


def load_frozen_config(config_path=None) -> dict:
    """Load the frozen configuration JSON.

    Defaults to the configuration bundled with the package; a custom
    ``config_path`` may be supplied to override it.
    """

    path = Path(config_path) if config_path else default_config_path()

    if not path.exists():
        resource = resources.files("hanbayes").joinpath("frozen.json")
        return json.loads(resource.read_text(encoding="utf-8"))

    with open(path, encoding="utf-8") as file:
        return json.load(file)
