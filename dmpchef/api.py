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
        p = base / p
    return p.resolve()


def _nih_pdf_corpus_ready(repo_root: Path, export_pdf_folder: str = "NIH_95") -> bool:
    """
    Minimal readiness check for NIH RAG corpus: do we have at least one PDF in data/NIH_95?
    (If your pipeline needs a vector index, you can extend this check to look for index files.)
    """
    pdf_dir = repo_root / "data" / export_pdf_folder
    return pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))


def prepare_nih_corpus(
    *,
    data_root: str | Path | None = None,
    json_links: str | Path = r"data\web_links.json",
    export_pdf_folder: str = "NIH_95",
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
    This is a heavy step (selenium/web crawl). Typically run once, then reuse.

    Returns basic info about where files were written.
    """
    repo_root = Path(__file__).resolve().parents[1]

    # Resolve default paths relative to repo root
    json_links = _resolve_path(json_links, repo_root)

    # If caller doesn't provide data_root, default to <repo_root>/data
    if data_root is None:
        data_root = repo_root / "data"
    data_root = _resolve_path(data_root, repo_root)

    # ------------------------------------------------------------------
    # IMPORTANT: Update this import to match your actual file name in src/
    # Example: if the file is src/NIH_data_ingestion.py then:
    #   from src.NIH_data_ingestion import UnifiedWebIngestion
    # ------------------------------------------------------------------
    from src.NIH_data_ingestion import UnifiedWebIngestion  # <-- CHANGE IF NEEDED

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


def generate(
    input_json: str | Path,
    *,
    config_path: str | Path = "config/config.yaml",
    nih_template_path: str | Path = "data/inputs/nih-dms-plan-template.docx",
    out_root: str | Path = "data/outputs",
    export_pdf: bool = False,
    use_rag: Optional[bool] = None,
    funding_agency: Optional[str] = None,
    # NEW:
    auto_prepare_rag: bool = False,
    rag_pdf_folder: str = "NIH_95",
    rag_json_links: str | Path = r"data\web_links.json",
) -> Dict[str, str]:
    """
    Importable API for DMP Chef.

    - Reads input.json
    - Runs pipeline (RAG/No-RAG + funder)
    - Writes Markdown, DOCX (template-preserving), DMPTool JSON
    - Optionally attempts PDF conversion
    - If RAG enabled, checks corpus readiness (and can optionally prepare it)

    Returns output paths + run metadata.
    """
    repo_root = Path(__file__).resolve().parents[1]

    input_json = _resolve_path(input_json, repo_root)
    req = json.loads(input_json.read_text(encoding="utf-8"))

    title = (req.get("title") or "").strip()
    if not title:
        raise ValueError("input.json must include a non-empty 'title'.")

    inputs: Dict[str, Any] = req.get("inputs") or {}

    if funding_agency is None:
        funding_agency = (req.get("funding_agency") or "NIH").strip().upper()

    if use_rag is None and "use_rag" in req:
        use_rag = bool(req["use_rag"])

    # Resolve paths relative to repo root
    config_path = _resolve_path(config_path, repo_root)
    nih_template_path = _resolve_path(nih_template_path, repo_root)
    out_root = _resolve_path(out_root, repo_root)

    # If RAG requested, ensure NIH corpus exists (or prepare it)
    if use_rag:
        if not _nih_pdf_corpus_ready(repo_root, export_pdf_folder=rag_pdf_folder):
            if auto_prepare_rag:
                prepare_nih_corpus(
                    data_root=repo_root / "data",
                    json_links=rag_json_links,
                    export_pdf_folder=rag_pdf_folder,
                )
            else:
                raise RuntimeError(
                    "RAG is enabled but the NIH corpus is not prepared.\n"
                    f"Expected PDFs under: {repo_root / 'data' / rag_pdf_folder}\n\n"
                    "Run one-time prep:\n"
                    "  from dmpchef import prepare_nih_corpus\n"
                    "  prepare_nih_corpus()\n\n"
                    "Or call generate(..., auto_prepare_rag=True)."
                )

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
        "repo_root": str(repo_root),
    }


# Alias if you prefer the name draft()
draft = generate
