from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import MODELS_DIR


def write_model_metadata(model_name: str, metrics: dict[str, Any], artifact_path: Path, extra: dict[str, Any] | None = None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    output_path = MODELS_DIR / f"{model_name}_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


def read_model_metadata(model_name: str) -> dict[str, Any]:
    path = MODELS_DIR / f"{model_name}_metadata.json"
    if not path.exists():
        return {"model_name": model_name, "status": "metadata_not_found"}
    return json.loads(path.read_text(encoding="utf-8"))
