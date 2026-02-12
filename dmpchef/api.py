# dmpchef/api.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any


from src.core_pipeline import DMPPipeline


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
    return re.sub(r'[\\/*?:"<>|]', "_", (text or "").strip()).strip() or "run"


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
      - {"project": {...}}   (some variants)
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


def generate(
    input_json: str | Path,
    *,
    config_path: str | Path = "config/config.yaml",
    out_root: str | Path = "data/outputs",
    use_rag: Optional[bool] = None,
    auto_prepare_rag: bool = False,
    rag_pdf_folder: str = "database",
    rag_json_links: str | Path = r"data\web_links.json",
) -> Dict[str, Any]:
    """
    Importable API for DMP Chef (supports old + new request JSON shapes).

    - Reads request JSON (title optional)
    - Funding:
        - agency: req.funding_agency OR req.config.funding.agency (default NIH)
        - subagency: req.config.funding.subagency (optional)
      -> Injects BOTH into form_inputs so core includes them in the prompt.
    - RAG:
        - use_rag arg overrides JSON
        - JSON supports req.use_rag OR req.config.pipeline.rag
        - If RAG requested and PDFs missing, either auto-prep or raise
    - Runs core pipeline: pipeline.generate_dmp(...)
    - Returns run metadata + output paths
    - NOTE: No file writing here (same as before): returns paths + markdown_text
      (main.py is still responsible for docx/pdf writing)
    """
    repo_root = Path(__file__).resolve().parents[1]

    input_json_path = _resolve_path(input_json, repo_root)
    config_path = _resolve_path(config_path, repo_root)
    out_root = _resolve_path(out_root, repo_root)

    req = json.loads(input_json_path.read_text(encoding="utf-8")) or {}
    if not isinstance(req, dict):
        raise ValueError("Input JSON must be an object at the top level.")

    # Title optional
    title = (req.get("title") or req.get("project_title") or "").strip()

    # Inputs may live under "inputs" or "project"
    inputs: Dict[str, Any] = _extract_inputs(req)

    # Funding (supports new schema)
    funding_agency = _extract_funding_agency(req)
    funding_subagency = _extract_funding_subagency(req)

    # ✅ Inject into inputs so CORE sees it (important requirement)
    if funding_agency:
        inputs.setdefault("funding_agency", funding_agency)
    if funding_subagency:
        inputs.setdefault("funding_subagency", funding_subagency)

    # Decide RAG usage (arg > JSON > pipeline YAML default)
    use_rag_final = _extract_use_rag(req, cli_use_rag=use_rag)

    # RAG readiness check (only when explicitly requested true)
    if use_rag_final is True:
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
                    f"Expected PDFs under: {repo_root / 'data' / rag_pdf_folder}\n\n"
                    "Run one-time prep:\n"
                    "  from dmpchef.api import prepare_nih_corpus\n"
                    "  prepare_nih_corpus(export_pdf_folder='database')\n\n"
                    "Or call generate(..., auto_prepare_rag=True)."
                )

    # Ensure output folders exist (main.py uses these too)
    (out_root / "markdown").mkdir(parents=True, exist_ok=True)
    (out_root / "docx").mkdir(parents=True, exist_ok=True)
    (out_root / "json").mkdir(parents=True, exist_ok=True)
    (out_root / "pdf").mkdir(parents=True, exist_ok=True)

    # Write an "effective request" next to outputs (so you can reproduce runs)
    effective_request_path = out_root / "debug" / f"{_safe_filename('effective_request')}.json"
    effective_request_path.parent.mkdir(parents=True, exist_ok=True)
    effective_request_path.write_text(
        json.dumps(
            {
                "title": title,
                "config": req.get("config", {}),
                "funding_agency": funding_agency,
                "funding_subagency": funding_subagency,
                "use_rag": use_rag_final,
                "inputs": inputs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Run pipeline (core returns markdown only)
    pipeline = DMPPipeline(config_path=str(config_path), force_rebuild_index=False)

    markdown_text = pipeline.generate_dmp(
        title=title,
        form_inputs=inputs,
        use_rag=use_rag_final,  # Optional[bool]; None -> pipeline YAML default
        funding_agency=funding_agency,
    )

    # Use the pipeline-generated stem (may already include agency/subagency if you updated main.py similarly)
    run_stem = pipeline.last_run_stem or _safe_filename("run")

    md_path = out_root / "markdown" / f"{run_stem}.md"
    docx_path = out_root / "docx" / f"{run_stem}.docx"
    dmptool_json_path = out_root / "json" / f"{run_stem}.dmptool.json"
    pdf_path = out_root / "pdf" / f"{run_stem}.pdf"

    return {
        "run_stem": run_stem,
        "repo_root": str(repo_root),
        "input_json": str(input_json_path),
        "effective_request_json": str(effective_request_path),
        "config_path": str(config_path),
        "funding": {
            "agency": funding_agency,
            "subagency": funding_subagency,
        },
        "pipeline": {
            "use_rag": use_rag_final,
        },
        "outputs": {
            "markdown": str(md_path),
            "docx": str(docx_path),
            "dmptool_json": str(dmptool_json_path),
            "pdf": str(pdf_path),
        },
        "markdown_text": markdown_text,
        "note": "This API returns paths + markdown only. File writing (DOCX/PDF/JSON) is handled by main.py.",
    }


draft = generate
