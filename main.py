# main.py  (JSON in -> Markdown + NIH-template DOCX + ONLY DMPTool JSON)
# ✅ supports RAG toggle via input JSON:  "use_rag": true/false  (or CLI --use_rag)
# ✅ keeps ONLY <safe_title>.dmptool.json (deletes other title*.json)

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from src.core_pipeline import DMPPipeline
from utils.dmptool_json import build_dmptool_json
from utils.nih_docx_writer import build_nih_docx_from_template


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", (title or "").strip()).strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cleanup_title_json(out_json_dir: Path, safe_title: str) -> None:
    """
    Delete any .json files for this title EXCEPT the dmptool one.
    This prevents duplicates from old runs or pipeline-side JSON outputs.
    """
    keep_name = f"{safe_title}.dmptool.json"
    for p in out_json_dir.glob(f"{safe_title}*.json"):
        if p.name != keep_name:
            try:
                p.unlink()
            except Exception:
                pass


def main(
    input_json_path: str = "data/inputs/input.json",
    out_root: str = "data/outputs",
    config_path: str = "config/config.yaml",
    nih_template_path: str = "data/inputs/nih-dms-plan-template.docx",
    use_rag: Optional[bool] = None,  # None => use YAML default (rag.enabled)
):
    # ---- Resolve + validate input path ----
    in_path = Path(input_json_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    # Load request
    req = json.loads(in_path.read_text(encoding="utf-8"))
    title = (req.get("title") or "").strip()
    inputs: Dict[str, Any] = req.get("inputs") or {}

    if not title:
        raise ValueError("Input JSON must include a non-empty 'title'.")

    st = safe_filename(title)
    if not st:
        raise ValueError("Safe filename is empty. Please provide a valid title.")

    # RAG toggle priority:
    # 1) CLI arg (if provided)
    # 2) JSON field "use_rag"
    # 3) YAML default rag.enabled inside pipeline
    if use_rag is None and "use_rag" in req:
        use_rag = bool(req.get("use_rag"))

    # Output folders
    out_root = Path(out_root).expanduser().resolve()
    out_json = out_root / "json"
    out_md = out_root / "markdown"
    out_docx = out_root / "docx"
    ensure_dir(out_json)
    ensure_dir(out_md)
    ensure_dir(out_docx)

    # Run pipeline (returns markdown)
    pipeline = DMPPipeline(config_path=config_path, force_rebuild_index=False)
    md_text = pipeline.generate_dmp(title, inputs, use_rag=use_rag)

    # ✅ Save Markdown output
    md_path = out_md / f"{st}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # ✅ Save ONLY DMPTool JSON (and delete other title JSONs)
    dmptool_payload = build_dmptool_json(
        template_title="NIH Data Management and Sharing Plan",
        project_title=title,
        form_inputs=inputs,
        generated_markdown=md_text,
        provenance="dmpchef",
    )
    dmptool_json_path = out_json / f"{st}.dmptool.json"

    cleanup_title_json(out_json, st)
    dmptool_json_path.write_text(json.dumps(dmptool_payload, indent=2), encoding="utf-8")

    # ✅ NIH template DOCX
    template_path = Path(nih_template_path).expanduser().resolve()
    if not template_path.exists():
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

    # Print final mode
    final_mode = pipeline.use_rag_default if use_rag is None else use_rag

    print("✅ Done")
    print(f"- Input: {in_path}")
    print(f"- use_rag: {final_mode}")
    print(f"- Markdown: {md_path}")
    print(f"- DMPTool JSON: {dmptool_json_path}")
    print(f"- NIH DOCX: {docx_path}")
    print(f"- Template used: {template_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/inputs/input.json")
    parser.add_argument("--out_root", default="data/outputs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--nih_template", default="data/inputs/nih-dms-plan-template.docx")

    # Optional toggle:
    #   python main.py --use_rag true
    #   python main.py --use_rag false
    # If omitted => uses JSON "use_rag" if present; otherwise YAML rag.enabled
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
