from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from jsonschema import Draft202012Validator


class SchemaValidationError(ValueError):
    pass


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_request(request_path: Union[str, Path], schema_path: Union[str, Path]) -> Dict[str, Any]:
    req = _load_json(request_path)
    schema = _load_json(schema_path)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(req), key=lambda e: list(e.path))

    if errors:
        lines = []
        for e in errors[:50]:
            loc = "$" + "".join([f".{p}" if isinstance(p, str) else f"[{p}]" for p in e.path])
            lines.append(f"{loc}: {e.message}")
        raise SchemaValidationError("Invalid DMP Chef request JSON:\n" + "\n".join(lines))

    return req
