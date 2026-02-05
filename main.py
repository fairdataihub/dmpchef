# main.py  (JSON in -> Markdown + NIH-template DOCX + PDF + ONLY DMPTool JSON)
# ✅ supports RAG toggle via input JSON:  "use_rag": true/false  (or CLI --use_rag)
# ✅ funding_agency is read from TOP-level JSON (outside inputs)
#
# Responsibility of this script:
#   1) Read a single request JSON (title, funding_agency, inputs, use_rag)
#   2) Run the DMPPipeline to generate markdown text
#   3) Save outputs: markdown + dmptool JSON + NIH DOCX (template-based) + PDF
#   4) Keep output JSON folder clean by removing older JSON variants

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from src.core_pipeline import DMPPipeline  # ✅ make sure this matches your file name
from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template

from docx2pdf import convert as docx2pdf_convert


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cleanup_title_json(out_json_dir: Path, file_stem: str) -> None:
    keep_name = f"{file_stem}.dmptool.json"
    for p in out_json_dir.glob(f"{file_stem}*.json"):
        if p.name != keep_name:
            try:
                p.unlink()
            except Exception:
                pass


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


def make_pdf_from_docx(docx_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    docx2pdf_convert(str(docx_path), str(pdf_path))


def main(
    input_json_path: str = "data/inputs/input.json",
    out_root: str = "data/outputs",
    config_path: str = "config/config.yaml",
    nih_template_path: str = "data/inputs/nih-dms-plan-template.docx",
    use_rag: Optional[bool] = None,
):
    in_path = Path(input_json_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    req = json.loads(in_path.read_text(encoding="utf-8"))
    title = (req.get("title") or "").strip()
    inputs: Dict[str, Any] = req.get("inputs") or {}
    funding_agency = (req.get("funding_agency") or "NIH").strip().upper()

    if not title:
        raise ValueError("Input JSON must include a non-empty 'title'.")

    # Decide RAG usage (CLI > JSON > YAML default)
    if use_rag is None and "use_rag" in req:
        use_rag = _to_bool(req.get("use_rag"), default=None)

    # Output dirs
    out_root = Path(out_root).expanduser().resolve()
    out_json = out_root / "json"
    out_md = out_root / "markdown"
    out_docx = out_root / "docx"
    out_pdf = out_root / "pdf"

    ensure_dir(out_json)
    ensure_dir(out_md)
    ensure_dir(out_docx)
    ensure_dir(out_pdf)

    # Run pipeline
    pipeline = DMPPipeline(config_path=config_path, force_rebuild_index=False)

    md_text = pipeline.generate_dmp(
        title,
        inputs,
        use_rag=use_rag,
        funding_agency=funding_agency,
    )

    # ✅ IMPORTANT: use the pipeline-generated stem (includes __rag__/__norag__)
    run_stem = pipeline.last_run_stem or safe_filename(title)

    # Save Markdown
    md_path = out_md / f"{run_stem}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # Save ONLY dmptool JSON
    dmptool_payload = build_dmptool_json(
        template_title="NIH Data Management and Sharing Plan",
        project_title=title,
        form_inputs=inputs,
        generated_markdown=md_text,
        provenance="dmpchef",
    )
    dmptool_json_path = out_json / f"{run_stem}.dmptool.json"
    cleanup_title_json(out_json, run_stem)
    dmptool_json_path.write_text(json.dumps(dmptool_payload, indent=2), encoding="utf-8")

    # Generate DOCX (main.py still uses NIH template path; pipeline already also writes docx in its folders)
    template_path = Path(nih_template_path).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"NIH template DOCX not found: {template_path}\n"
            f'Fix: python main.py --nih_template "PATH\\TO\\nih-dms-plan-template.docx"'
        )

    docx_path = out_docx / f"{run_stem}.docx"
    build_nih_docx_from_template(
        template_docx_path=str(template_path),
        output_docx_path=str(docx_path),
        project_title=title,
        generated_markdown=md_text,
    )

    # Convert to PDF
    pdf_path = out_pdf / f"{run_stem}.pdf"
    make_pdf_from_docx(docx_path, pdf_path)

    final_mode = pipeline.use_rag_default if use_rag is None else use_rag

    print("✅ Done")
    print(f"- Input: {in_path}")
    print(f"- funding_agency: {funding_agency}")
    print(f"- use_rag: {final_mode}")
    print(f"- Output stem: {run_stem}")
    print(f"- Markdown: {md_path}")
    print(f"- DMPTool JSON: {dmptool_json_path}")
    print(f"- NIH DOCX: {docx_path}")
    print(f"- PDF: {pdf_path}")
    print(f"- Template used: {template_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/inputs/input.json")
    parser.add_argument("--out_root", default="data/outputs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--nih_template", default="data/inputs/nih-dms-plan-template.docx")
    parser.add_argument("--use_rag", choices=["true", "false"], default=None)

    args = parser.parse_args()

    use_rag_val: Optional[bool] = None
    if args.use_rag == "true":
        use_rag_val = True
    elif args.use_rag == "false":
        use_rag_val = False

    main(
        input_json_path=args.input,
        out_root=args.out_root,
        config_path=args.config,
        nih_template_path=args.nih_template,
        use_rag=use_rag_val,
    )
