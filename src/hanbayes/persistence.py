"""Model persistence helpers.

Models are plain-Python/numpy objects; we use pickle for simplicity, with a
``metadata`` envelope storing the config and vocabulary needed to reload.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersistedAnalyzer:
    """Envelope saved alongside the analyzer payload."""

    version: str
    config: dict
    metadata: dict


def save_analyzer(analyzer, path, config: dict | None = None) -> Path:
    """Persist a fitted :class:`ChineseSentimentAnalyzer`."""

    from . import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = PersistedAnalyzer(
        version=__version__,
        config=config or analyzer.config,
        metadata=analyzer.metadata_,
    )
    with open(path, "wb") as file:
        pickle.dump({"envelope": envelope, "analyzer": analyzer}, file)
    return path


def load_analyzer(path):
    """Reload an analyzer saved by :func:`save_analyzer`."""

    path = Path(path)
    with open(path, "rb") as file:
        payload = pickle.load(file)
    return payload["analyzer"]
