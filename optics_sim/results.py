from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from .core import Plane


@dataclass
class Result:
    module: str
    parameters: dict
    metrics: dict
    planes: dict[str, Plane] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = dict(self.arrays)
        for name, plane in self.planes.items():
            arrays[name+"_field"] = plane.field
            arrays[name+"_x_m"] = plane.x
        np.savez_compressed(directory/"data.npz", **arrays)
        metadata = {"module": self.module, "parameters": self.parameters,
                    "metrics": self.metrics, "notes": self.notes,
                    "data_keys": sorted(arrays)}
        (directory/"metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return directory
