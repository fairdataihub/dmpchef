# dmpchef/api.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from docx2pdf import convert as docx2pdf_convert

from config.schema_validate import validate_request
from src.core_pipeline import DMPPipeline
from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template


# -------------------------------------------------------------------
# Utilities (kept aligned with main.py behavior)
# -------------------------------------------------------------------
def safe_filename(title: str) -> str:
    """Convert a string into a filesystem-safe filename stem."""
    s = re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "request"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cleanup_title_json(out_json_dir: Path, file_stem: str) -> None:
    """
    Keep only: <stem>.dmptool.json
    Remove older variants: <stem>*.json except that one.
    """
    keep_name = f"{file_stem}.dmptool.json"
    for p in out_json_dir.glob(f"{file_stem}*.json"):
        if p.name != keep_name:
            try:
                p.unlink()
            except Exception:
                pass


def _to_bool(v: Any, default: Optional[bool] = None) -> Optional[bool]:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _get_nested(d: Dict[str, Any], path: str, default=None):
    """Safe nested getter: path like 'config.pipeline.rag'."""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_request_json(in_path: Path) -> Dict[str, Any]:
    req = json.loads(in_path.read_text(encoding="utf-8")) or {}
    if not isinstance(req, dict):
        raise ValueError("Input JSON must be an object at the top level.")
    return req


def _extract_inputs(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both:
      - {"inputs": {...}}
      - {"project": {...}}
    """
    inputs = req.get("inputs")
    if isinstance(inputs, dict):
        return inputs
    project = req.get("project")
    if isinstance(project, dict):
        return project
    return {}


def _extract_funding_agency(req: Dict[str, Any]) -> str:
    """
    Supports:
      - old: req["funding_agency"]
      - new: req["config"]["funding"]["agency"]
    """
    v = req.get("funding_agency")
    if not v:
        v = _get_nested(req, "config.funding.agency")
    return (v or "NIH").strip().upper()


def _extract_funding_subagency(req: Dict[str, Any]) -> str:
    """
    Supports:
      - old: req["funding_subagency"]
      - new: req["config"]["funding"]["subagency"]
    """
    v = req.get("funding_subagency")
    if not v:
        v = _get_nested(req, "config.funding.subagency")
    return (v or "").strip().upper()


def _extract_use_rag(req: Dict[str, Any], api_override: Optional[bool]) -> Optional[bool]:
    """
    Priority (aligned with main.py):
      API arg override > JSON(old: use_rag) > JSON(new: config.pipeline.rag) > None (YAML default)
    """
    if api_override is not None:
        return api_override

    if "use_rag" in req:
        return _to_bool(req.get("use_rag"), default=None)

    nested = _get_nested(req, "config.pipeline.rag", default=None)
    if nested is not None:
        return _to_bool(nested, default=None)

    return None


def _extract_llm_model_name(req: Dict[str, Any]) -> str:
    """
    Read model override from request:
      config.pipeline.llm.model_name
    """
    v = _get_nested(req, "config.pipeline.llm.model_name", default=None)
    return (v or "").strip()


def _extract_title(req: Dict[str, Any]) -> str:
    """
    IMPORTANT FIX:
    Only accept explicit title fields; never use run_id/request_id as title.
    """
    title = (req.get("title") or req.get("project_title") or "").strip()
    if re.fullmatch(r"req_[a-z0-9]{6,}", title.lower()):
        return ""
    return title


def _prefix_run_stem_with_funding(run_stem: str, agency: str, subagency: str) -> str:
    """
    Prefix <AGENCY>_<SUBAGENCY>__ before the *base* stem, preserving the pipeline suffix.
    """
    agency_safe = safe_filename(agency)
    subagency_safe = safe_filename(subagency) if subagency else ""

    suffix = ""
    base = run_stem

    if "__rag__" in run_stem:
        base, rest = run_stem.split("__rag__", 1)
        suffix = "__rag__" + rest
    elif "__norag__" in run_stem:
        base, rest = run_stem.split("__norag__", 1)
        suffix = "__norag__" + rest

    prefix_parts = [p for p in [agency_safe, subagency_safe] if p]
    prefix = "_".join(prefix_parts).strip("_")

    if prefix:
        base = f"{prefix}__{base}" if base else prefix

    return f"{base}{suffix}"


def _make_pdf_from_docx(docx_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    docx2pdf_convert(str(docx_path), str(pdf_path))


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def generate(
    input_json: str | Path,
    *,
    out_root: str | Path = "data/outputs",
    config_path: str | Path = "config/config.yaml",
    nih_template_path: str | Path = "data/inputs/nih-dms-plan-template.docx",
    use_rag: Optional[bool] = None,  # API override (like CLI flag)
    write_files: bool = True,
    export_docx: bool = True,
    export_dmptool_json: bool = True,
    export_pdf: bool = True,
    show_json: bool = False,
    return_json: bool = True,
) -> Dict[str, Any]:
    """
    Generate a DMP using the core pipeline and (optionally) write outputs.

    Design notes:
    - We initialize DMPPipeline first so we can resolve all paths consistently
      via pipeline.config.resolve_path(), which respects config.root_dir.
    - Title comes ONLY from req["title"] / req["project_title"] (never run_id).
    """

    # Initialize pipeline first (root_dir-aware)
    pipeline = DMPPipeline(config_path=str(config_path), force_rebuild_index=False)

    # Resolve paths using the SAME root_dir logic as the pipeline
    in_path = pipeline.config.resolve_path(input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    # Schema path stays next to repo (relative to repo containing main.py/config/)
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "config" / "dmpchef_request.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    validate_request(in_path, schema_path)
    print(f"Schema validation: PASS ({in_path.name})")

    req = _read_request_json(in_path)

    inputs: Dict[str, Any] = _extract_inputs(req)
    funding_agency = _extract_funding_agency(req)
    funding_subagency = _extract_funding_subagency(req)
    use_rag_final = _extract_use_rag(req, api_override=use_rag)
    llm_model_name = _extract_llm_model_name(req)

    # Title (explicit only)
    title = _extract_title(req)

    # Inject agency/subagency for prompt visibility (same as main.py)
    if funding_agency:
        inputs.setdefault("funding_agency", funding_agency)
    if funding_subagency:
        inputs.setdefault("funding_subagency", funding_subagency)

    # Output roots resolved consistently
    out_root_path = pipeline.config.resolve_path(out_root)
    out_json = out_root_path / "json"
    out_md = out_root_path / "markdown"
    out_docx = out_root_path / "docx"
    out_pdf = out_root_path / "pdf"

    for p in [out_json, out_md, out_docx, out_pdf]:
        ensure_dir(p)

    # Core pipeline (Markdown only)
    md_text = pipeline.generate_dmp(
        title=title,
        form_inputs=inputs,
        use_rag=use_rag_final,
        funding_agency=funding_agency,
        llm_model_name=(llm_model_name or None),
    )

    core_run_stem = pipeline.last_run_stem or safe_filename(title)
    actual_mode = "rag" if "__rag__" in core_run_stem else "norag"

    # Prefix filename with agency/subagency
    run_stem = _prefix_run_stem_with_funding(core_run_stem, funding_agency, funding_subagency)
    pipeline.last_run_stem = run_stem

    md_path = out_md / f"{run_stem}.md"
    dmptool_json_path = out_json / f"{run_stem}.dmptool.json"
    docx_path = out_docx / f"{run_stem}.docx"
    pdf_path = out_pdf / f"{run_stem}.pdf"

    written: Dict[str, Any] = {"markdown": False, "dmptool_json": False, "docx": False, "pdf": False, "errors": {}}
    dmptool_payload: Optional[Dict[str, Any]] = None

    if write_files:
        # Markdown
        try:
            md_path.write_text(md_text, encoding="utf-8")
            written["markdown"] = md_path.exists()
        except Exception as e:
            written["errors"]["markdown"] = str(e)

        # DMPTool JSON
        if export_dmptool_json:
            try:
                dmptool_payload = build_dmptool_json(
                    template_title="NIH Data Management and Sharing Plan",
                    project_title=title,
                    form_inputs=inputs,
                    generated_markdown=md_text,
                    provenance="dmpchef",
                )

                cleanup_title_json(out_json, run_stem)
                dmptool_json_path.write_text(
                    json.dumps(dmptool_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                written["dmptool_json"] = dmptool_json_path.exists()

                if show_json:
                    print("DMPTOOL JSON (PREVIEW)")
                    print(json.dumps(dmptool_payload, indent=2, ensure_ascii=False))

            except Exception as e:
                written["errors"]["dmptool_json"] = str(e)

        # DOCX
        if export_docx:
            try:
                template_path = pipeline.config.resolve_path(nih_template_path)
                if not template_path.exists():
                    raise FileNotFoundError(f"NIH template DOCX not found: {template_path}")

                build_nih_docx_from_template(
                    template_docx_path=str(template_path),
                    output_docx_path=str(docx_path),
                    project_title=title,  # kept for backward compat; not inserted
                    plan_json=dmptool_payload,
                )
                written["docx"] = docx_path.exists()
            except Exception as e:
                written["errors"]["docx"] = str(e)

        # PDF
        if export_pdf:
            try:
                if not docx_path.exists():
                    raise FileNotFoundError(f"DOCX not found for PDF conversion: {docx_path}")
                _make_pdf_from_docx(docx_path, pdf_path)
                written["pdf"] = pdf_path.exists()
            except Exception as e:
                written["errors"]["pdf"] = str(e)

    return {
        "run_stem": run_stem,
        "input_json": str(in_path),
        "funding": {"agency": funding_agency, "subagency": funding_subagency},
        "pipeline": {
            "use_rag_requested": use_rag_final,
            "use_rag_actual": actual_mode,
            "llm_model_name_requested": (llm_model_name or None),
            "llm_model_name_actual": getattr(pipeline, "llm_name", None),
            "title_used": (title or None),
        },
        "outputs": {
            "markdown": str(md_path),
            "dmptool_json": str(dmptool_json_path),
            "docx": str(docx_path),
            "pdf": str(pdf_path),
        },
        "written": written,
        "markdown_text": md_text,
        "dmptool_payload": (dmptool_payload if return_json else None),
    }



# -------------------------------------------------------------------
def draft(input_json: str | Path, **kwargs) -> Dict[str, Any]:
    """Alias for generate() kept for older notebooks/imports."""
    return generate(input_json, **kwargs)


