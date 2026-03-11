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


def clean_markdown(text: str) -> str:
    """
    Remove markdown formatting symbols.
    """
    # remove markdown headers and stars
    text = re.sub(r"[#*_`]", "", text)

    # normalize spacing
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean extracted section text.
    """
    # remove leading colon
    text = re.sub(r"^:\s*", "", text)

    # remove trailing subsection numbers like "2." or "3."
    text = re.sub(r"\n\s*\d+\.\s*$", "", text)

    # normalize whitespace
    text = re.sub(r"\n\s+", "\n", text)

    return text.strip()


ELEMENT_STRUCTURE = {
    "Element 1: Data Type": [
        "Types and amount of scientific data expected to be generated in the project",
        "Scientific data that will be preserved and shared, and the rationale for doing so",
        "Metadata, other relevant data, and associated documentation",
    ],
    "Element 2: Related Tools, Software and/or Code": [],
    "Element 3: Standards": [],
    "Element 4: Data Preservation, Access, and Associated Timelines": [
        "Repository where scientific data and metadata will be archived",
        "How scientific data will be findable and identifiable",
        "When and how long the scientific data will be made available",
    ],
    "Element 5: Access, Distribution, or Reuse Considerations": [
        "Factors affecting subsequent access, distribution, or reuse of scientific data",
        "Whether access to scientific data will be controlled",
        "Protections for privacy, rights, and confidentiality of human research participants",
    ],
    "Element 6: Oversight of Data Management and Sharing": [],
}


def markdown_to_json(md_text: str):

    md_text = clean_markdown(md_text)

    result = {}

    for element, subsections in ELEMENT_STRUCTURE.items():

        if element not in md_text:
            continue

        start = md_text.index(element)

        next_positions = [
            md_text.index(e)
            for e in ELEMENT_STRUCTURE.keys()
            if e in md_text and md_text.index(e) > start
        ]

        end = min(next_positions) if next_positions else len(md_text)

        body = md_text[start + len(element):end].strip()

        if subsections:

            structured = {}
            current_text = body

            for i, title in enumerate(subsections):

                if title not in current_text:
                    continue

                s = current_text.index(title)

                next_titles = [
                    current_text.index(t)
                    for t in subsections
                    if t in current_text and current_text.index(t) > s
                ]

                e = min(next_titles) if next_titles else len(current_text)

                description = current_text[s + len(title):e]
                description = clean_text(description)

                structured[str(i + 1)] = {
                    "title": title.strip().rstrip(":"),
                    "description": description
                }

            result[element] = structured

        else:

            result[element] = {
                "description": clean_text(body)
            }

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