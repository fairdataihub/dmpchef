# main.py

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from src.core_pipeline import DMPPipeline
from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template

from docx2pdf import convert as docx2pdf_convert


# -------------------------
# Small utilities
# -------------------------
def safe_filename(title: str) -> str:
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


def _read_request_json(in_path: Path) -> Dict[str, Any]:
    req = json.loads(in_path.read_text(encoding="utf-8")) or {}
    if not isinstance(req, dict):
        raise ValueError("Input JSON must be an object at the top level.")
    return req


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
      CLI --use_rag > JSON(old: use_rag) > JSON(new: config.pipeline.rag) > None (pipeline YAML default)
    """
    if cli_use_rag is not None:
        return cli_use_rag

    if "use_rag" in req:  # old key
        return _to_bool(req.get("use_rag"), default=None)

    nested = _get_nested(req, "config.pipeline.rag", default=None)  # new key
    if nested is not None:
        return _to_bool(nested, default=None)

    return None


def _prefix_run_stem_with_funding(run_stem: str, agency: str, subagency: str) -> str:
    """
    Prefix <AGENCY>_<SUBAGENCY>__ before the *base* stem, preserving the pipeline suffix (__rag__/__norag__...).
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


# -------------------------
# Main entry
# -------------------------
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

    req = _read_request_json(in_path)

    title = (req.get("title") or req.get("project_title") or "").strip()
    inputs: Dict[str, Any] = _extract_inputs(req)

    funding_agency = _extract_funding_agency(req)
    funding_subagency = _extract_funding_subagency(req)

    # Inject agency/subagency into inputs so core pipeline includes them in the prompt
    if funding_agency:
        inputs.setdefault("funding_agency", funding_agency)
    if funding_subagency:
        inputs.setdefault("funding_subagency", funding_subagency)

    use_rag = _extract_use_rag(req, cli_use_rag=use_rag)

    out_root_path = Path(out_root).expanduser().resolve()
    out_json = out_root_path / "json"
    out_md = out_root_path / "markdown"
    out_docx = out_root_path / "docx"
    out_pdf = out_root_path / "pdf"

    ensure_dir(out_json)
    ensure_dir(out_md)
    ensure_dir(out_docx)
    ensure_dir(out_pdf)

    pipeline = DMPPipeline(config_path=config_path, force_rebuild_index=False)

    md_text = pipeline.generate_dmp(
        title=title,
        form_inputs=inputs,
        use_rag=use_rag,
        funding_agency=funding_agency,
    )

    run_stem = pipeline.last_run_stem or safe_filename(title)
    actual_mode = "rag" if "__rag__" in run_stem else "norag"

    # Prefix filename with agency/subagency
    run_stem = _prefix_run_stem_with_funding(run_stem, funding_agency, funding_subagency)
    pipeline.last_run_stem = run_stem

    # Save Markdown
    md_path = out_md / f"{run_stem}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # Build DMPTool JSON (this is the structured answers we will use for DOCX)
    dmptool_payload = build_dmptool_json(
        template_title="NIH Data Management and Sharing Plan",
        project_title=title,  # harmless even if empty; not emitted anymore
        form_inputs=inputs,
        generated_markdown=md_text,
        provenance="dmpchef",
    )

    # Save JSON
    dmptool_json_path = out_json / f"{run_stem}.dmptool.json"
    cleanup_title_json(out_json, run_stem)
    dmptool_json_path.write_text(
        json.dumps(dmptool_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    template_path = Path(nih_template_path).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"NIH template DOCX not found: {template_path}\n"
            f'Fix: python main.py --nih_template "PATH\\TO\\nih-dms-plan-template.docx"'
        )

    #  DOCX: use dmptool_json (plan_json) instead of markdown parsing
    docx_path = out_docx / f"{run_stem}.docx"
    build_nih_docx_from_template(
        template_docx_path=str(template_path),
        output_docx_path=str(docx_path),
        project_title=title,          # kept for backwards compat; not inserted
        
        plan_json=dmptool_payload,    
    )

    # PDF
    pdf_path = out_pdf / f"{run_stem}.pdf"
    make_pdf_from_docx(docx_path, pdf_path)

    print("Done")
    print(f"- Input: {in_path}")
    print(f"- funding_agency: {funding_agency}")
    print(f"- funding_subagency: {funding_subagency if funding_subagency else '(none)'}")
    print(f"- use_rag (requested): {use_rag if use_rag is not None else 'YAML default'}")
    print(f"- use_rag (actual): {actual_mode}")
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
