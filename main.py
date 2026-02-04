# app.py  (JSON in -> Markdown + NIH-template DOCX + ONLY DMPTool JSON)
# ✅ also removes any other JSON files for this title (keeps ONLY <safe_title>.dmptool.json)

import json
import re
from pathlib import Path
from typing import Dict, Any

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

    # Output folders
    out_root = Path(out_root).expanduser().resolve()
    out_json = out_root / "json"
    out_md = out_root / "markdown"
    out_docx = out_root / "docx"
    ensure_dir(out_json)
    ensure_dir(out_md)
    ensure_dir(out_docx)

    # Run pipeline
    pipeline = DMPPipeline(config_path=config_path, force_rebuild_index=False)
    md_text = pipeline.generate_dmp(title, inputs)

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

    # remove any duplicates for this title (old files, pipeline outputs, etc.)
    cleanup_title_json(out_json, st)

    dmptool_json_path.write_text(json.dumps(dmptool_payload, indent=2), encoding="utf-8")

    # ✅ NIH template DOCX
    template_path = Path(nih_template_path).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"NIH template DOCX not found: {template_path}\n"
            f'Fix: python app.py --nih_template "PATH\\TO\\nih-dms-plan-template.docx"'
        )

    docx_path = out_docx / f"{st}.docx"
    build_nih_docx_from_template(
        template_docx_path=str(template_path),
        output_docx_path=str(docx_path),
        project_title=title,
        generated_markdown=md_text,
    )

    print("✅ Done")
    print(f"- Input: {in_path}")
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
    args = parser.parse_args()

    main(
        input_json_path=args.input,
        out_root=args.out_root,
        config_path=args.config,
        nih_template_path=args.nih_template,
    )
