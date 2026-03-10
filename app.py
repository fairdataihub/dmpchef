CONFIG_PATH = "config/config.yaml"
BASE_INPUT_PATH = "data_prd/inputs/input.json"

import re
import uuid
import time
import queue
import threading
import logging

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path

from src.core_pipeline_prd import DMPPipeline
from utils.dmptool_json import build_dmptool_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
CORS(app)

from main import (
    _read_request_json,
    _extract_inputs,
    _extract_funding_agency,
    _extract_funding_subagency,
    _extract_use_rag,
)
BASE_REQ = _read_request_json(Path(BASE_INPUT_PATH))
pipeline = DMPPipeline(config_path=CONFIG_PATH, force_rebuild_index=False)

job_queue = queue.Queue()
jobs = {}


ELEMENTS_WITH_SUBSECTIONS = {"Element 1", "Element 4", "Element 5"}

def parse_subsections(text: str):
    """
    Extract subsections from a text block.
    Handles ### headings or *bolded subsection titles.
    """
    # Match ### headings
    pattern_h3 = r"###\s+(.*?)\n(.*?)(?=\n###|\Z)"
    matches_h3 = re.findall(pattern_h3, text, re.S)
    if matches_h3:
        return [{"title": t.strip(), "description": b.strip()} for t, b in matches_h3]

    # Fallback: match *Title:* pattern
    pattern_star = r"\*\s*(.*?)\s*:\*\s*(.*?)(?=\n\*|\Z)"
    matches_star = re.findall(pattern_star, text, re.S)
    if matches_star:
        return [{"title": t.strip(), "description": b.strip()} for t, b in matches_star]

    # If no subsections, return entire text as a single section
    return [{"title": "Section", "description": text.strip()}]


def markdown_to_json(md_text: str) -> dict:
    if not md_text:
        return {}

    result = {}

    # Split by Element headings (bold or standard)
    elements = re.split(r"\*\*\s*(Element\s+\d+:[^*]+?)\s*\*\*", md_text)
    for i in range(1, len(elements), 2):
        element_title = elements[i].strip()
        element_body = elements[i + 1] if i + 1 < len(elements) else ""
        element_key = element_title.split(":")[0]

        element_body = element_body.strip()
        # Handle elements with subsections
        if element_key in ELEMENTS_WITH_SUBSECTIONS:
            sections = parse_subsections(element_body)
            structured = {}
            for idx, sec in enumerate(sections, start=1):
                structured[str(idx)] = {
                    "title": sec["title"],
                    "description": sec["description"]
                }
            result[element_title] = structured
        else:
            # Single description elements
            # Split body into paragraphs
            paragraphs = [p.strip() for p in element_body.split("\n\n") if p.strip()]

            # Always remove the first paragraph
            paragraphs = paragraphs[1:] if len(paragraphs) > 1 else []

            # Rejoin the remaining paragraphs
            clean_text = "\n\n".join(paragraphs).strip()
            result[element_title] = {"description": clean_text}

    return result


def worker():
    while True:
        task = job_queue.get()
        job_id = task["job_id"]

        try:
            logger.info(f"🚀 Processing job {job_id}")

            base_inputs = _extract_inputs(BASE_REQ)
            funding_agency = _extract_funding_agency(BASE_REQ)
            funding_subagency = _extract_funding_subagency(BASE_REQ)
            
            use_rag = _extract_use_rag(BASE_REQ, cli_use_rag=None)

            override_subagency = task.get("agency")

            if override_subagency:
                funding_subagency = override_subagency.strip().upper()

            override_inputs = {
                "research_area": task.get("projectSummary"),
                "research_context": task.get("projectSummary"),
                "data_types": task.get("dataType"),
                "data_source": task.get("dataSource"),
                "human_subjects": task.get("humanSubjects"),
                "consent_status": task.get("dataSharing"),
                "data_volume": task.get("dataVolume"),
            }

            override_inputs = {k: v for k, v in override_inputs.items() if v is not None}

            final_inputs = {**base_inputs, **override_inputs}
            if funding_agency:
                final_inputs.setdefault("funding_agency", funding_agency)

            if funding_subagency:
                final_inputs["funding_subagency"] = funding_subagency

            title = (task.get("title") or "").strip()

            markdown = pipeline.generate_dmp(
                title=title,
                form_inputs=final_inputs,
                use_rag=use_rag,
                funding_agency=funding_agency,
            )

            job_output_root = Path("data_prd/outputs/jobs") / job_id
            md_dir = job_output_root / "markdown"
            docx_dir = job_output_root / "docx"
            json_dir = job_output_root / "json"

            for d in [md_dir, docx_dir, json_dir]:
                d.mkdir(parents=True, exist_ok=True)

            jobs[job_id]["output_path"] = str(job_output_root)

            # Convert markdown to structured json
            structured_json = markdown_to_json(markdown)


            md_path = md_dir / "dmp.md"
            md_path.write_text(markdown, encoding="utf-8")

            import json as _json
            dmptool_payload = build_dmptool_json(
                template_title="NIH Data Management and Sharing Plan",
                project_title=title,
                form_inputs=final_inputs,
                generated_markdown=markdown,
                provenance="dmpchef",
            )

            json_path = json_dir / "dmp.json"
            json_path.write_text(
                _json.dumps(dmptool_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            docx_path = docx_dir / "dmp.docx"


            if docx_path.exists():
                logger.info("DOCX already exists, skipping regeneration")
            else:
                try:
                    from utils.nih_docx_writer import build_nih_docx_from_template

                    template_path = pipeline.config.resolve_path(
                        "data_prd/inputs/nih-dms-plan-template.docx"
                    )

                    build_nih_docx_from_template(
                        template_docx_path=str(template_path),
                        output_docx_path=str(docx_path),
                        project_title=title,
                        plan_json=dmptool_payload,
                    )
                except Exception as e:
                    logger.warning(f"DOCX generation failed: {e}")

            jobs[job_id].update({
                "status": "completed",
                "result": structured_json,
                "finished_at": time.time()
            })

            logger.info(f"Job {job_id} completed {final_inputs}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "finished_at": time.time()
            })

        finally:
            job_queue.task_done()


# Start worker thread
threading.Thread(target=worker, daemon=True).start()


@app.route("/query", methods=["POST"])
def generate():
    """
    Client calls this.
    Returns job_id immediately.
    """

    data = request.get_json()

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing",
        "created_at": time.time()
    }

    job_queue.put({
        "job_id": job_id,
        "title": data.get("title"),
        "agency": data.get("agency"),
        "projectSummary": data.get("projectSummary"),
        "dataType": data.get("dataType"),
        "dataSource": data.get("dataSource"),
        "humanSubjects": data.get("humanSubjects"),
        "dataSharing": data.get("dataSharing"),
        "dataVolume": data.get("dataVolume"),
    })

    logger.info(f"Job {job_id} queued")

    return jsonify({
        "job_id": job_id
    }), 202


@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """
    Poll this endpoint to check job progress.
    """

    job = jobs.get(job_id)

    if not job:
        return jsonify({
            "status": "not_found"
        }), 404

    return jsonify(job)


@app.route("/download/<job_id>/<file_type>", methods=["GET"])
def download_artifact(job_id, file_type):

    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        return jsonify({"error": "invalid_job"}), 400

    if "output_path" not in job:
        return jsonify({"error": "output_path_missing"}), 400

    output_root = Path(job["output_path"])

    file_map = {
        "md": output_root / "markdown" / "dmp.md",
        "docx": output_root / "docx" / "dmp.docx",
        "json": output_root / "json" / "dmp.json",
    }

    if file_type not in file_map:
        return jsonify({"error": "invalid_type"}), 400

    file_path = file_map[file_type]

    if not file_path.exists():
        return jsonify({"error": "file_missing"}), 404

    return send_file(file_path, as_attachment=True)


@app.route("/up", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "queue_size": job_queue.qsize()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)