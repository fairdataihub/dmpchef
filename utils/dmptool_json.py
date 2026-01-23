# utils/dmptool_json.py
from typing import Any, Dict, List, Optional


def _pretty_question_text(key: str) -> str:
    """
    Convert snake_case keys to readable question labels.
    Example: 'data_sharing_consent_status' -> 'Data sharing consent status'
    """
    return key.replace("_", " ").strip().capitalize()


def _build_questions(form_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert flat form_inputs into dmptool narrative questions.
    Keeps only non-empty string values.
    """
    questions: List[Dict[str, Any]] = []
    q_order = 1

    for k, v in (form_inputs or {}).items():
        if isinstance(v, str) and v.strip():
            questions.append(
                {
                    "question_order": q_order,
                    "question_text": _pretty_question_text(k),
                    "answer_json": {
                        "answer": v.strip(),
                        "type": "textArea",
                    },
                }
            )
            q_order += 1

    return questions


def build_dmptool_json(
    title: str,
    form_inputs: Dict[str, Any],
    generated_markdown: str,
    template_title: str = "NIH DMS Plan Template",
    provenance: str = "dmp_chef",
    outputs: Optional[Dict[str, Any]] = None,
    template_used: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build dmptool-style JSON output.

    Output shape:
    {
      "dmptool": {
        "provenance": "dmp_chef",
        "narrative": {
          "template_title": "NIH DMS Plan Template",
          "project_title": "...",
          "sections": [...]
        },
        "source": {...},
        "outputs": {...}
      }
    }
    """
    title = (title or "").strip()

    sections = [
        {
            "section_order": 1,
            "section_title": "Project Details",
            "section_description": "Inputs provided through the web form.",
            "questions": _build_questions(form_inputs),
        },
        {
            "section_order": 2,
            "section_title": "Generated NIH DMS Plan (Markdown)",
            "section_description": "LLM-generated plan using retrieved NIH context and the Markdown template.",
            "questions": [
                {
                    "question_order": 1,
                    "question_text": "Generated DMP",
                    "answer_json": {
                        "answer": (generated_markdown or "").strip(),
                        "type": "markdown",
                    },
                }
            ],
        },
    ]

    dmptool_obj: Dict[str, Any] = {
        "dmptool": {
            "provenance": provenance,
            "narrative": {
                "template_title": template_title,
                "project_title": title,
                "sections": sections,
            },
            # Traceability to the template used (optional but helpful)
            "source": {
                "template_used": template_used,
            },
            # Paths to generated artifacts (optional but helpful)
            "outputs": outputs or {},
        }
    }

    return dmptool_obj
