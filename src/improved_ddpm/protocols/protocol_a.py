"""Small immutable specification used by controlled cross-dataset runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


DATASET_NAMES = {"M": "MSU-MFSD", "S": "SiW-Mv2"}


@dataclass(frozen=True)
class ProtocolASpec:
    name: str
    sources: Tuple[str, ...]
    target: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source_names"] = [DATASET_NAMES[key] for key in self.sources]
        payload["target_name"] = DATASET_NAMES[self.target]
        return payload

