# main.py  (JSON in -> Markdown + NIH-template DOCX + PDF + ONLY DMPTool JSON)
# ✅ supports RAG toggle via input JSON:  "use_rag": true/false  (or CLI --use_rag)
# ✅ funding_agency is read from TOP-level JSON (outside inputs)
#
# Responsibility of this script:
#   1) Read a single request JSON (title, funding_agency, inputs, use_rag)
#   2) Run the DMPPipeline to generate markdown text
#   3) Save outputs: markdown + dmptool JSON + NIH DOCX (template-based) + PDF
#   4) Keep output JSON folder clean by removing older JSON variants

# -------------------------------
# Standard library imports
# -------------------------------
import json                         # Read/write JSON request + JSON output
import re                           # Sanitize filenames (replace illegal characters)
from pathlib import Path            # Safe cross-platform filesystem paths
from typing import Dict, Any, Optional  # Type hints for readability and editor support

# -------------------------------
# Local project imports
# -------------------------------
from src.core_pipeline import DMPPipeline                  # Core RAG/No-RAG DMP generation pipeline
from utils.dmptool_json import build_dmptool_json          # Create DMPTool-compatible JSON payload
from utils.nih_docx_writer import build_nih_docx_from_template  # Generate NIH DOCX using official template formatting

# -------------------------------
# External dependency for DOCX -> PDF
# -------------------------------
# docx2pdf:
#   - On Windows, typically uses Microsoft Word for conversion
#   - On macOS, uses JXA / Word automation
#   - On Linux, support is limited (often not available / not reliable)
from docx2pdf import convert as docx2pdf_convert


def safe_filename(title: str) -> str:
    """
    Convert a user-provided title into a filesystem-safe filename stem.

    Why:
      Windows forbids characters like: \ / * ? : " < > |
      This keeps output paths portable across OSes.

    Args:
      title: raw project title

    Returns:
      Sanitized filename string (no forbidden characters).
    """
    return re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()


def ensure_dir(p: Path) -> None:
    """
    Ensure a directory exists (create parents if needed).

    Args:
      p: Path to a directory
    """
    p.mkdir(parents=True, exist_ok=True)


def cleanup_title_json(out_json_dir: Path, safe_title: str) -> None:
    """
    Delete any .json files for this title EXCEPT the dmptool one.

    Why:
      Older pipeline versions may have created multiple JSON outputs per run.
      This prevents duplicates and keeps only the intended DMPTool JSON artifact.

    Args:
      out_json_dir: directory where JSON outputs are stored
      safe_title: sanitized project title used as output file stem
    """
    keep_name = f"{safe_title}.dmptool.json"
    for p in out_json_dir.glob(f"{safe_title}*.json"):
        if p.name != keep_name:
            try:
                p.unlink()
            except Exception:
                # Ignore deletion failures (non-critical cleanup)
                pass


def _to_bool(v, default: Optional[bool] = None) -> Optional[bool]:
    """
    Robust bool parser for JSON/CLI/YAML values.

    Why:
      bool("false") is True in Python because it's a non-empty string.
      This helper prevents accidental truthiness bugs.

    Args:
      v: value to parse (bool/int/float/str/None)
      default: returned when v is None or cannot be parsed confidently

    Returns:
      True/False if parsed; otherwise `default`.
    """
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


def make_pdf_from_docx(docx_path: Path, pdf_path: Path) -> None:
    """
    Convert a DOCX to PDF using docx2pdf.

    Notes:
      - On Windows/macOS: usually works if MS Word is installed/available.
      - On Linux: often not supported; conversion may fail.

    Args:
      docx_path: input DOCX file path
      pdf_path: output PDF file path
    """
    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # docx2pdf expects string paths
    docx2pdf_convert(str(docx_path), str(pdf_path))


def main(
    input_json_path: str = "data/inputs/input.json",
    out_root: str = "data/outputs",
    config_path: str = "config/config.yaml",
    nih_template_path: str = "data/inputs/nih-dms-plan-template.docx",
    use_rag: Optional[bool] = None,  # None => use YAML default (rag.enabled)
):
    """
    Main entrypoint for running one DMP generation job.

    Args:
      input_json_path: path to the request JSON file
      out_root: root outputs folder (subfolders: json/ markdown/ docx/ pdf/)
      config_path: YAML config path (paths/models/rag settings)
      nih_template_path: NIH DOCX template path
      use_rag: Optional override; if None we fall back to JSON then YAML default
    """
    # ---- Resolve + validate input path ----
    in_path = Path(input_json_path).expanduser().resolve()  # normalize and resolve input path
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    # ---- Load request JSON ----
    # Expected structure:
    #   { "title": "...", "funding_agency": "NIH", "use_rag": true/false, "inputs": {...} }
    req = json.loads(in_path.read_text(encoding="utf-8"))
    title = (req.get("title") or "").strip()
    inputs: Dict[str, Any] = req.get("inputs") or {}

    # ✅ funding_agency is top-level and normalized to uppercase
    funding_agency = (req.get("funding_agency") or "NIH").strip().upper()

    # Validate title
    if not title:
        raise ValueError("Input JSON must include a non-empty 'title'.")

    # Build a safe filename stem from title
    st = safe_filename(title)
    if not st:
        raise ValueError("Safe filename is empty. Please provide a valid title.")

    # ---- Decide RAG usage (priority order) ----
    # 1) CLI arg (passed into main via use_rag parameter)
    # 2) JSON field "use_rag"
    # 3) YAML default rag.enabled inside pipeline
    if use_rag is None and "use_rag" in req:
        use_rag = _to_bool(req.get("use_rag"), default=None)

    # ---- Build output folder paths ----
    out_root = Path(out_root).expanduser().resolve()
    out_json = out_root / "json"
    out_md = out_root / "markdown"
    out_docx = out_root / "docx"
    out_pdf = out_root / "pdf"  # ✅ NEW: PDF output folder

    # Ensure all output folders exist
    ensure_dir(out_json)
    ensure_dir(out_md)
    ensure_dir(out_docx)
    ensure_dir(out_pdf)

    # ---- Run pipeline (returns markdown) ----
    # force_rebuild_index=False: reuse existing FAISS index unless pipeline decides to rebuild
    pipeline = DMPPipeline(config_path=config_path, force_rebuild_index=False)

    # ✅ Pass funding_agency into generate_dmp (currently used for logging/future routing)
    md_text = pipeline.generate_dmp(
        title,
        inputs,
        use_rag=use_rag,
        funding_agency=funding_agency
    )

    # ---- Save Markdown output ----
    md_path = out_md / f"{st}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # ---- Save ONLY DMPTool JSON output ----
    # Even though pipeline also saves JSON in its own output dir, this script enforces the "ONLY dmptool" policy here too.
    dmptool_payload = build_dmptool_json(
        template_title="NIH Data Management and Sharing Plan",
        project_title=title,
        form_inputs=inputs,
        generated_markdown=md_text,
        provenance="dmpchef",
    )
    dmptool_json_path = out_json / f"{st}.dmptool.json"

    # Remove older JSON variants for this title before writing the new dmptool JSON
    cleanup_title_json(out_json, st)

    # Write JSON with pretty indentation for readability
    dmptool_json_path.write_text(
        json.dumps(dmptool_payload, indent=2),
        encoding="utf-8"
    )

    # ---- Generate NIH template DOCX ----
    template_path = Path(nih_template_path).expanduser().resolve()
    if not template_path.exists():
        # Helpful message that matches your CLI flag name
        raise FileNotFoundError(
            f"NIH template DOCX not found: {template_path}\n"
            f'Fix: python main.py --nih_template "PATH\\TO\\nih-dms-plan-template.docx"'
        )

    docx_path = out_docx / f"{st}.docx"
    build_nih_docx_from_template(
        template_docx_path=str(template_path),
        output_docx_path=str(docx_path),
        project_title=title,
        generated_markdown=md_text,
    )

    # ---- Convert DOCX to PDF ----
    pdf_path = out_pdf / f"{st}.pdf"
    make_pdf_from_docx(docx_path, pdf_path)

    # Determine what to print for "use_rag" (explicit override vs YAML default)
    final_mode = pipeline.use_rag_default if use_rag is None else use_rag

    # ---- Console summary ----
    print("✅ Done")
    print(f"- Input: {in_path}")
    print(f"- funding_agency: {funding_agency}")
    print(f"- use_rag: {final_mode}")
    print(f"- Markdown: {md_path}")
    print(f"- DMPTool JSON: {dmptool_json_path}")
    print(f"- NIH DOCX: {docx_path}")
    print(f"- PDF: {pdf_path}")
    print(f"- Template used: {template_path}")


if __name__ == "__main__":
    # CLI wrapper so you can run:
    #   python main.py --input ... --out_root ... --config ... --nih_template ... --use_rag true/false
    import argparse

    parser = argparse.ArgumentParser()

    # Input/output/config arguments
    parser.add_argument("--input", default="data/inputs/input.json")
    parser.add_argument("--out_root", default="data/outputs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--nih_template", default="data/inputs/nih-dms-plan-template.docx")

    # Optional RAG toggle: parse "true"/"false" into a real bool
    parser.add_argument("--use_rag", choices=["true", "false"], default=None)

    args = parser.parse_args()

    # Convert CLI string to bool (or None if not provided)
    use_rag_val: Optional[bool] = None
    if args.use_rag == "true":
        use_rag_val = True
    elif args.use_rag == "false":
        use_rag_val = False

    # Run main with CLI-provided paths and optional RAG override
    main(
        input_json_path=args.input,
        out_root=args.out_root,
        config_path=args.config,
        nih_template_path=args.nih_template,
        use_rag=use_rag_val,
    )
