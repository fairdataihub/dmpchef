# dmpchef/api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from src.core_pipeline import DMPPipeline
from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template


def _resolve_path(p: str | Path, base: Path) -> Path:
    """
    Resolve a path robustly:
    - expands ~
    - if relative, resolves it relative to `base`
    - returns an absolute Path
    """
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = (base / p)
    return p.resolve()


def generate(
    input_json: str | Path,
    *,
    config_path: str | Path = "config/config.yaml",
    nih_template_path: str | Path = "data/inputs/nih-dms-plan-template.docx",
    out_root: str | Path = "data/outputs",
    export_pdf: bool = False,
    use_rag: Optional[bool] = None,
    funding_agency: Optional[str] = None,
) -> Dict[str, str]:
    # ✅ Repo root: dmpchef/api.py -> parent of dmpchef/ folder
    REPO_ROOT = Path(__file__).resolve().parents[1]

    input_json = _resolve_path(input_json, REPO_ROOT)
    req = json.loads(input_json.read_text(encoding="utf-8"))

    title = (req.get("title") or "").strip()
    if not title:
        raise ValueError("input.json must include a non-empty 'title'.")

    inputs: Dict[str, Any] = req.get("inputs") or {}

    if funding_agency is None:
        funding_agency = (req.get("funding_agency") or "NIH").strip().upper()

    if use_rag is None and "use_rag" in req:
        use_rag = bool(req["use_rag"])

    # ✅ Resolve these relative to repo root too (fixes notebook cwd issues)
    config_path = _resolve_path(config_path, REPO_ROOT)
    nih_template_path = _resolve_path(nih_template_path, REPO_ROOT)
    out_root = _resolve_path(out_root, REPO_ROOT)

    (out_root / "markdown").mkdir(parents=True, exist_ok=True)
    (out_root / "docx").mkdir(parents=True, exist_ok=True)
    (out_root / "json").mkdir(parents=True, exist_ok=True)
    (out_root / "pdf").mkdir(parents=True, exist_ok=True)

    pipeline = DMPPipeline(config_path=str(config_path), force_rebuild_index=False)
    md = pipeline.generate_dmp(
        title,
        inputs,
        use_rag=use_rag,
        funding_agency=funding_agency,
    )
    run_stem = pipeline.last_run_stem or title.replace("/", "_")

    md_path = out_root / "markdown" / f"{run_stem}.md"
    md_path.write_text(md, encoding="utf-8")

    dmptool_payload = build_dmptool_json(
        template_title="NIH Data Management and Sharing Plan",
        project_title=title,
        form_inputs=inputs,
        generated_markdown=md,
        provenance="dmpchef",
    )
    json_path = out_root / "json" / f"{run_stem}.dmptool.json"
    json_path.write_text(json.dumps(dmptool_payload, indent=2), encoding="utf-8")

    docx_path = out_root / "docx" / f"{run_stem}.docx"
    build_nih_docx_from_template(
        template_docx_path=str(nih_template_path),
        output_docx_path=str(docx_path),
        project_title=title,
        generated_markdown=md,
    )

    pdf_path = out_root / "pdf" / f"{run_stem}.pdf"
    if export_pdf:
        try:
            from docx2pdf import convert
            convert(str(docx_path), str(pdf_path))
        except Exception:
            pdf_path = Path("")

    return {
        "markdown": str(md_path),
        "docx": str(docx_path),
        "dmptool_json": str(json_path),
        "pdf": str(pdf_path) if str(pdf_path) else "",
        "funding_agency": funding_agency,
        "use_rag": str(use_rag),
        "run_stem": run_stem,
    }


draft = generate
