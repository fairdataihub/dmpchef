# dmpchef/api.py
from __future__ import annotations

import inspect
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.core_pipeline import DMPPipeline
from utils.schema_validate import validate_request


def _resolve_path(p: str | Path, base: Path) -> Path:
    """
    Resolve a path robustly:
    - expands ~
    - if relative, resolves it relative to `base`
    - returns an absolute Path
    """
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _safe_filename(text: str) -> str:
    """Make a filesystem-safe filename stem."""
    s = re.sub(r'[\\/*?:"<>|]', "_", (text or "").strip())
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "run"


def _get_nested(d: Dict[str, Any], path: str, default=None):
    """
    Safe nested getter: path like 'config.pipeline.rag'
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_bool(v, default: Optional[bool] = None) -> Optional[bool]:
    """Convert common truthy/falsy inputs to bool; otherwise return default."""
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


def _pdf_corpus_ready(repo_root: Path, pdf_folder: str) -> bool:
    """
    Minimal readiness check: do we have at least one PDF in repo_root/data/<pdf_folder> ?
    """
    pdf_dir = repo_root / "data" / pdf_folder
    return pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))


def _canonicalize(obj: Any) -> Any:
    """Make dict/list stable for hashing (sort keys recursively)."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    return obj


def _make_run_id(
    *,
    inputs: Dict[str, Any],
    funding_agency: str,
    funding_subagency: str,
    # IMPORTANT: do NOT include rag in identity so rag/norag share same base
    prefix: str = "req",
    n: int = 10,
) -> str:
    """
    Deterministic ID derived from inputs + funding. Same request -> same ID.
    RAG/NORAG will still differ because core_pipeline appends __rag__/__norag__ suffix.
    """
    payload = {
        "funding_agency": funding_agency,
        "funding_subagency": funding_subagency,
        "inputs": inputs,
    }
    canonical = _canonicalize(payload)
    s = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"


#  time-only stamp for filenames (HHMMSS)
def _time_stamp() -> str:
    return datetime.now().strftime("%H%M%S")


def _stamp_stem_with_time(run_stem: str) -> str:
    """
    Append time before __rag__/__norag__ suffix (if present) so you keep mode info.
    """
    t = _time_stamp()

    if "__rag__" in run_stem:
        base, rest = run_stem.split("__rag__", 1)
        base = base.strip("_") or "request"
        return f"{base}__{t}__rag__{rest}"

    if "__norag__" in run_stem:
        base, rest = run_stem.split("__norag__", 1)
        base = base.strip("_") or "request"
        return f"{base}__{t}__norag__{rest}"

    base = run_stem.strip("_") or "request"
    return f"{base}__{t}"


def prepare_nih_corpus(
    *,
    data_root: str | Path | None = None,
    json_links: str | Path = r"data\web_links.json",
    export_pdf_folder: str = "database",
    export_mode: str = "move",
    max_depth: int = 5,
    crawl_delay: float = 1.2,
    max_pages: int = 18000,
    dmptool_safety_max_pages: int = 500,
    keep_last_n_sessions: int = 2,
    copy_forward_previous: bool = True,
) -> Dict[str, str]:
    """
    Run the NIH web ingestion to populate data/<export_pdf_folder> with PDFs for RAG.
    Typically run once, then reuse.
    """
    repo_root = Path(__file__).resolve().parents[1]

    json_links = _resolve_path(json_links, repo_root)

    if data_root is None:
        data_root = repo_root / "data"
    data_root = _resolve_path(data_root, repo_root)

    from src.NIH_data_ingestion import UnifiedWebIngestion  # update if needed

    crawler = UnifiedWebIngestion(
        data_root=data_root,
        json_links=str(json_links),
        max_depth=max_depth,
        crawl_delay=crawl_delay,
        max_pages=max_pages,
        export_pdf_folder=export_pdf_folder,
        export_mode=export_mode,
        dmptool_safety_max_pages=dmptool_safety_max_pages,
        keep_last_n_sessions=keep_last_n_sessions,
        copy_forward_previous=copy_forward_previous,
    )
    crawler.run_all()

    return {
        "data_root": str(data_root),
        "export_pdf_folder": str((Path(data_root) / export_pdf_folder).resolve()),
        "json_links": str(json_links),
    }


def _extract_inputs(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both:
      - {"inputs": {...}}
      - {"project": {...}}  (some variants)
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
      - new: req["config"]["funding"]["subagency"]
      - optional old: req["funding_subagency"]
    """
    v = req.get("funding_subagency")
    if not v:
        v = _get_nested(req, "config.funding.subagency")
    return (v or "").strip().upper()


def _extract_use_rag(req: Dict[str, Any], cli_use_rag: Optional[bool]) -> Optional[bool]:
    """
    Priority:
      CLI override > JSON(old: use_rag) > JSON(new: config.pipeline.rag) > None (pipeline YAML default)
    """
    if cli_use_rag is not None:
        return cli_use_rag

    if "use_rag" in req:
        return _to_bool(req.get("use_rag"), default=None)

    nested = _get_nested(req, "config.pipeline.rag", default=None)
    if nested is not None:
        return _to_bool(nested, default=None)

    return None


def _call_with_supported_kwargs(func, **kwargs):
    """
    Call func(**kwargs) but only pass keyword args that exist in func's signature.
    """
    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**filtered)


def _build_dmptool_payload_like_main(
    *,
    inputs: Dict[str, Any],
    markdown_text: str,
    project_title: str,
    req: Dict[str, Any],
    funding_agency: str,
    funding_subagency: str,
) -> Dict[str, Any]:
    """
    Build DMPTool JSON payload using the same call pattern as main.py, when possible.
    Falls back to alternate signatures if your utils function differs.
    """
    from utils.dmptool_json import build_dmptool_json

    sig = inspect.signature(build_dmptool_json)
    params = set(sig.parameters.keys())

    # main.py signature
    if {"template_title", "form_inputs", "generated_markdown"}.issubset(params):
        payload = build_dmptool_json(
            template_title="NIH Data Management and Sharing Plan",
            project_title=project_title,
            form_inputs=inputs,
            generated_markdown=markdown_text,
            provenance="dmpchef",
        )
        if not isinstance(payload, dict):
            raise ValueError("build_dmptool_json did not return a dict payload.")
        return payload

    # fallback signature variants
    payload = _call_with_supported_kwargs(
        build_dmptool_json,
        request=req,
        inputs=inputs,
        form_inputs=inputs,
        markdown_text=markdown_text,
        generated_markdown=markdown_text,
        funding_agency=funding_agency,
        funding_subagency=funding_subagency,
        provenance="dmpchef",
        template_title="NIH Data Management and Sharing Plan",
        project_title=project_title,
    )
    if not isinstance(payload, dict):
        raise ValueError("build_dmptool_json did not return a dict payload.")
    return payload


def generate(
    input_json: str | Path,
    *,
    config_path: str | Path = "config/config.yaml",
    out_root: str | Path = "data/outputs",
    nih_template_path: str | Path = "data/inputs/nih-dms-plan-template.docx",
    use_rag: Optional[bool] = None,
    auto_prepare_rag: bool = False,
    rag_pdf_folder: str = "database",
    rag_json_links: str | Path = r"data\web_links.json",
    write_files: bool = True,
    export_docx: bool = True,
    export_dmptool_json: bool = True,
    export_pdf: bool = True,
    show_json: bool = False,
    return_json: bool = True,
) -> Dict[str, Any]:
    """
    Importable API for DMP Chef.

    Fixes:
    - No title in JSON: creates deterministic run_id from inputs
    - Prevents rag/norag collisions: run_id is stable + core adds __rag__/__norag__ suffix
    - Avoid overwriting outputs: appends TIME-only stamp (HHMMSS) to run_stem for each call
    - Validates corpus folder used by core pipeline (paths.data_pdfs)
    - Validates input JSON against config/dmpchef_request.schema.json
    """
    repo_root = Path(__file__).resolve().parents[1]

    input_json_path = _resolve_path(input_json, repo_root)
    config_path = _resolve_path(config_path, repo_root)
    out_root = _resolve_path(out_root, repo_root)
    template_path = _resolve_path(nih_template_path, repo_root)

    # -------------------------
    # JSON Schema validation (required, like main.py)
    # -------------------------
    schema_path = repo_root / "config" / "dmpchef_request.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    try:
        validate_request(input_json_path, schema_path)
        print(f"Schema validation: PASS ({input_json_path.name})")
    except Exception:
        print(f"Schema validation: FAIL ({input_json_path.name})")
        raise

    req = json.loads(input_json_path.read_text(encoding="utf-8")) or {}
    if not isinstance(req, dict):
        raise ValueError("Input JSON must be an object at the top level.")

    # No title in JSON. Build deterministic run_id from inputs.
    original_inputs: Dict[str, Any] = _extract_inputs(req)
    inputs: Dict[str, Any] = dict(original_inputs)

    funding_agency = _extract_funding_agency(req)
    funding_subagency = _extract_funding_subagency(req)

    if funding_agency:
        inputs.setdefault("funding_agency", funding_agency)
    if funding_subagency:
        inputs.setdefault("funding_subagency", funding_subagency)

    use_rag_final = _extract_use_rag(req, cli_use_rag=use_rag)

    # Deterministic request ID (used as the "title" input to core pipeline for stable base stem)
    run_id = _make_run_id(
        inputs=inputs,
        funding_agency=funding_agency,
        funding_subagency=funding_subagency,
    )

    # You said you want title empty inside dmptool_json
    project_title = ""

    # Output directories
    out_md = out_root / "markdown"
    out_docx = out_root / "docx"
    out_json_dir = out_root / "json"
    out_pdf = out_root / "pdf"
    out_debug = out_root / "debug"

    out_md.mkdir(parents=True, exist_ok=True)
    out_docx.mkdir(parents=True, exist_ok=True)
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_pdf.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)

    # Effective request debug file (time-stamped; no overwrite)
    effective_request_path = out_debug / f"effective_request__{_time_stamp()}.json"
    effective_request_path.write_text(
        json.dumps(
            {
                "config": req.get("config", {}),
                "funding_agency": funding_agency,
                "funding_subagency": funding_subagency,
                "use_rag": use_rag_final,
                "run_id": run_id,
                "original_inputs": original_inputs,
                "inputs": inputs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Run pipeline to get markdown text
    pipeline = DMPPipeline(config_path=str(config_path), force_rebuild_index=False)

    # IMPORTANT: core pipeline uses config.yaml paths.data_pdfs
    core_pdf_dir = getattr(pipeline, "data_pdfs", None)

    # If caller asks RAG=True, validate corpus where core will read PDFs
    if use_rag_final is True:
        if core_pdf_dir is not None:
            core_pdf_dir = Path(core_pdf_dir)
            if not (core_pdf_dir.exists() and any(core_pdf_dir.glob("*.pdf"))):
                if auto_prepare_rag:
                    # Prepares data/<rag_pdf_folder>, but your YAML might point elsewhere.
                    prepare_nih_corpus(
                        data_root=repo_root / "data",
                        json_links=rag_json_links,
                        export_pdf_folder=rag_pdf_folder,
                    )
                    if not (core_pdf_dir.exists() and any(core_pdf_dir.glob("*.pdf"))):
                        raise RuntimeError(
                            "RAG enabled, but core pipeline corpus folder is still empty after auto_prepare_rag.\n"
                            f"core pipeline expects PDFs under YAML paths.data_pdfs: {core_pdf_dir}\n"
                            f"auto_prepare_rag populated: {repo_root / 'data' / rag_pdf_folder}\n"
                            "Fix: set config.yaml paths.data_pdfs to point to your prepared folder."
                        )
                else:
                    raise RuntimeError(
                        "RAG is enabled but core pipeline PDF corpus folder is empty.\n"
                        f"core pipeline expects PDFs under YAML paths.data_pdfs: {core_pdf_dir}\n"
                        "Fix: put PDFs there OR update config.yaml paths.data_pdfs to your corpus folder."
                    )
        else:
            # fallback check if pipeline doesn't expose data_pdfs
            if not _pdf_corpus_ready(repo_root, pdf_folder=rag_pdf_folder):
                if auto_prepare_rag:
                    prepare_nih_corpus(
                        data_root=repo_root / "data",
                        json_links=rag_json_links,
                        export_pdf_folder=rag_pdf_folder,
                    )
                else:
                    raise RuntimeError(
                        "RAG is enabled but the PDF corpus is not prepared.\n"
                        f"Expected PDFs under: {repo_root / 'data' / rag_pdf_folder}\n"
                        "Fix: prepare corpus or point config.yaml paths.data_pdfs correctly."
                    )

    markdown_text = pipeline.generate_dmp(
        title=run_id,                 # stable base stem without needing a real title
        form_inputs=inputs,
        use_rag=use_rag_final,         #force True/False when provided
        funding_agency=funding_agency,
    )

    # Core sets run_stem like: <safe(title)>__rag__/__norag__...
    base_stem = pipeline.last_run_stem or _safe_filename(run_id)

    # Make it unique per call (time-only, no overwrite)
    run_stem = _stamp_stem_with_time(base_stem)
    pipeline.last_run_stem = run_stem

    md_path = out_md / f"{run_stem}.md"
    docx_path = out_docx / f"{run_stem}.docx"
    dmptool_json_path = out_json_dir / f"{run_stem}.dmptool.json"
    pdf_path = out_pdf / f"{run_stem}.pdf"

    written: Dict[str, Any] = {
        "markdown": False,
        "dmptool_json": False,
        "docx": False,
        "pdf": False,
        "errors": {},
    }

    dmptool_payload: Optional[Dict[str, Any]] = None

    if write_files:
        # 1) Markdown
        try:
            md_path.write_text(markdown_text, encoding="utf-8")
            written["markdown"] = md_path.exists()
        except Exception as e:
            written["errors"]["markdown"] = str(e)

        # 2) DMPTool JSON (build like main.py)
        if export_dmptool_json:
            try:
                dmptool_payload = _build_dmptool_payload_like_main(
                    inputs=inputs,
                    markdown_text=markdown_text,
                    project_title=project_title,
                    req=req,
                    funding_agency=funding_agency,
                    funding_subagency=funding_subagency,
                )

                dmptool_json_path.write_text(
                    json.dumps(dmptool_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                written["dmptool_json"] = dmptool_json_path.exists()

                if show_json and dmptool_payload is not None:
                    print("\n========== DMPTOOL JSON (PREVIEW) ==========\n")
                    print(json.dumps(dmptool_payload, indent=2, ensure_ascii=False))
                    print("\n===========================================\n")

            except Exception as e:
                written["errors"]["dmptool_json"] = str(e)

        # 3) DOCX (prefer using plan_json if available)
        if export_docx:
            try:
                from utils.nih_docx_writer import build_nih_docx_from_template

                if not template_path.exists():
                    raise FileNotFoundError(f"NIH template DOCX not found: {template_path}")

                _call_with_supported_kwargs(
                    build_nih_docx_from_template,
                    template_docx_path=str(template_path),
                    output_docx_path=str(docx_path),
                    project_title=project_title,
                    plan_json=dmptool_payload,
                    markdown_text=markdown_text,
                    output_path=str(docx_path),
                    out_path=str(docx_path),
                    output_docx=str(docx_path),
                )

                written["docx"] = docx_path.exists()
            except Exception as e:
                written["errors"]["docx"] = str(e)

        # 4) PDF (docx2pdf)
        if export_pdf:
            try:
                if not docx_path.exists():
                    raise FileNotFoundError(f"DOCX not found for PDF conversion: {docx_path}")

                from docx2pdf import convert as docx2pdf_convert

                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                docx2pdf_convert(str(docx_path), str(pdf_path))
                written["pdf"] = pdf_path.exists()
            except Exception as e:
                written["errors"]["pdf"] = str(e)

    return {
        "run_stem": run_stem,
        "base_stem": base_stem,
        "run_id": run_id,
        "repo_root": str(repo_root),
        "input_json": str(input_json_path),
        "effective_request_json": str(effective_request_path),
        "config_path": str(config_path),
        "funding": {"agency": funding_agency, "subagency": funding_subagency},
        "pipeline": {
            "use_rag_requested": use_rag_final,
            "use_rag_actual": ("rag" if "__rag__" in run_stem else "norag"),
            "core_pdf_dir": (str(core_pdf_dir) if core_pdf_dir is not None else None),
        },
        "outputs": {
            "markdown": str(md_path),
            "dmptool_json": str(dmptool_json_path),
            "docx": str(docx_path),
            "pdf": str(pdf_path),
        },
        "written": written,
        "markdown_text": markdown_text,
        "dmptool_payload": (dmptool_payload if return_json else None),
        "note": "API writes md/json/docx/pdf when write_files=True. run_id is deterministic; run_stem is time-stamped (HHMMSS) to avoid overwrites.",
    }


draft = generate