# utils/dmptool_json.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _clean(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def _first_nonempty(form_inputs: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = form_inputs.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _question(order: int, text: str, answer: str, qtype: str = "textArea") -> Dict[str, Any]:
    return {
        "order": order,
        "text": text,
        "answer": {
            "json": {
                "answer": _clean(answer),
                "type": qtype,
            }
        },
    }


# ============================================================
# Template structure (ends at Element 6)
# ============================================================
NIH_SECTIONS: List[Dict[str, Any]] = [
    {
        "order": 1,
        "title": "Element 1: Data Type",
        "description": "",
        "questions": [
            (1, "Types and amount of scientific data expected to be generated in the project"),
            (2, "Scientific data that will be preserved and shared, and the rationale for doing so"),
            (3, "Metadata, other relevant data, and associated documentation"),
        ],
    },
    {
        "order": 2,
        "title": "Element 2: Related Tools, Software and/or Code",
        "description": "",
        "questions": [(1, "Element 2: Related Tools, Software and/or Code")],
    },
    {
        "order": 3,
        "title": "Element 3: Standards",
        "description": "",
        "questions": [(1, "Element 3: Standards")],
    },
    {
        "order": 4,
        "title": "Element 4: Data Preservation, Access, and Associated Timelines",
        "description": "",
        "questions": [
            (1, "Repository where scientific data and metadata will be archived"),
            (2, "How scientific data will be findable and identifiable"),
            (3, "When and how long the scientific data will be made available"),
        ],
    },
    {
        "order": 5,
        "title": "Element 5: Access, Distribution, or Reuse Considerations",
        "description": "",
        "questions": [
            (1, "Factors affecting subsequent access, distribution, or reuse of scientific data"),
            (2, "Whether access to scientific data will be controlled"),
            (3, "Protections for privacy, rights, and confidentiality of human research participants"),
        ],
    },
    {
        "order": 6,
        "title": "Element 6: Oversight of Data Management and Sharing",
        "description": "",
        "questions": [(1, "Element 6: Oversight of Data Management and Sharing")],
    },
]


# ============================================================
# Markdown parsing helpers
# Goal: map generated markdown -> answers for (sec.order, q.order)
# Keys returned: "1.1", "1.2", ..., "6.1"
# ============================================================

_ELEMENT_HEADING_RE = re.compile(
    r"(?m)^\*\*Element\s*(?P<num>[1-6])\s*:\s*(?P<title>.+?)\*\*\s*$"
)

_SUBHEADING_RE = re.compile(r"(?m)^###\s*(?P<h>.+?)\s*$")

# Markdown "setext" underline lines: ----- or =====
_SETEXT_UNDERLINE_RE = re.compile(r"(?m)^\s*[-=]{3,}\s*$")


def _slice_between(text: str, start_idx: int, end_idx: Optional[int]) -> str:
    if end_idx is None:
        return text[start_idx:]
    return text[start_idx:end_idx]


def _strip_setext_underlines(text: str) -> str:
    """
    Remove markdown underline lines made of only '-' or '='.
    Helps clean answers when model outputs:
        Heading
        --------
    """
    text = _clean(text)
    if not text:
        return ""
    # remove these lines anywhere (safe)
    text = _SETEXT_UNDERLINE_RE.sub("", text)
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _parse_elements(md: str) -> Dict[int, str]:
    """
    Returns {1: "<element1 body>", 2: "<element2 body>", ...}
    Body excludes the **Element X: ...** line.
    """
    md = _clean(md)
    matches = list(_ELEMENT_HEADING_RE.finditer(md))
    out: Dict[int, str] = {}

    for i, m in enumerate(matches):
        num = int(m.group("num"))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else None
        body = _slice_between(md, body_start, body_end).strip()
        body = _strip_setext_underlines(body)
        out[num] = body

    return out


def _extract_under_subheading(block: str, subheading_text: str) -> str:
    """
    In a block (one element body), find '### {subheading_text}' and capture text until next ### or end.
    Returns "" if not found.
    """
    block = _strip_setext_underlines(block)
    # exact-ish match but forgiving whitespace/case
    pat = re.compile(
        rf"(?mi)^###\s*{re.escape(subheading_text)}\s*:\s*$|(?mi)^###\s*{re.escape(subheading_text)}\s*$"
    )
    m = pat.search(block)
    if not m:
        return ""

    start = m.end()
    next_m = _SUBHEADING_RE.search(block, pos=start)
    end = next_m.start() if next_m else None
    return _strip_setext_underlines(_slice_between(block, start, end).strip())


def _best_effort_element_body(block: str) -> str:
    """
    For elements without ### subheadings (2,3,6), just return the body,
    but clean out underline artifacts.
    """
    return _strip_setext_underlines(block)


def _parse_generated_markdown_to_answers(generated_markdown: str) -> Dict[str, str]:
    """
    Map generated markdown into answers for keys:
      1.1, 1.2, 1.3, 2.1, 3.1, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1
    """
    md = _clean(generated_markdown)
    elements = _parse_elements(md)

    answers: Dict[str, str] = {}

    # Element 1: three ### subsections
    e1 = elements.get(1, "")
    answers["1.1"] = _extract_under_subheading(
        e1, "Types and amount of scientific data expected to be generated in the project"
    )
    answers["1.2"] = _extract_under_subheading(
        e1, "Scientific data that will be preserved and shared, and the rationale for doing so"
    )
    answers["1.3"] = _extract_under_subheading(e1, "Metadata, other relevant data, and associated documentation")

    # Element 2: whole body
    answers["2.1"] = _best_effort_element_body(elements.get(2, ""))

    # Element 3: whole body
    answers["3.1"] = _best_effort_element_body(elements.get(3, ""))

    # Element 4: three ### subsections
    e4 = elements.get(4, "")
    answers["4.1"] = _extract_under_subheading(e4, "Repository where scientific data and metadata will be archived")
    answers["4.2"] = _extract_under_subheading(e4, "How scientific data will be findable and identifiable")
    answers["4.3"] = _extract_under_subheading(e4, "When and how long the scientific data will be made available")

    # Element 5: three ### subsections
    e5 = elements.get(5, "")
    answers["5.1"] = _extract_under_subheading(
        e5, "Factors affecting subsequent access, distribution, or reuse of scientific data"
    )
    answers["5.2"] = _extract_under_subheading(e5, "Whether access to scientific data will be controlled")
    answers["5.3"] = _extract_under_subheading(
        e5, "Protections for privacy, rights, and confidentiality of human research participants"
    )

    # Element 6: whole body
    answers["6.1"] = _best_effort_element_body(elements.get(6, ""))

    # Final clean
    for k, v in list(answers.items()):
        answers[k] = _clean(v)

    return answers


def build_dmptool_json(
    template_title: str,
    project_title: str,  # kept for backward compatibility; not emitted anymore
    form_inputs: Dict[str, Any],
    generated_markdown: str = "",
    provenance: str = "dmpchef",
) -> Dict[str, Any]:
    """
    Ends at Section 6. No Section 7.

    Strategy:
    1) Parse generated_markdown into per-question answers (preferred).
    2) If parsing yields empty for a question, fallback to form_inputs mapping.

    Change:
    - Remove "project_title" from the output JSON.
    - Clean out markdown underline artifacts like "-----" in answers.
    """
    form_inputs = form_inputs or {}

    # fallback mapping (only used if parsed markdown doesn't contain that answer)
    FIELD_MAP: Dict[str, List[str]] = {
        # Element 1
        "1.1": ["types_and_amount", "data_types_and_amount", "data_types", "data_volume"],
        "1.2": ["preserved_shared_rationale", "data_preserved_shared", "share_rationale"],
        "1.3": ["metadata_docs", "metadata_and_docs", "documentation"],

        # Element 2
        "2.1": ["tools_software_code", "tools", "software_code"],

        # Element 3
        "3.1": ["standards", "data_standards"],

        # Element 4
        "4.1": ["repository", "repositories", "data_repository"],
        "4.2": ["findable_identifiable", "identifiers", "pids_indexing"],
        "4.3": ["availability_timeline", "timeline", "retention_period"],

        # Element 5
        "5.1": ["reuse_factors", "limitations", "access_distribution_reuse_factors"],
        "5.2": ["controlled_access", "access_controlled"],
        "5.3": ["privacy_protections", "human_subjects_protections", "confidentiality"],

        # Element 6
        "6.1": ["oversight", "compliance_oversight", "roles_responsibilities"],
    }

    parsed = _parse_generated_markdown_to_answers(generated_markdown) if generated_markdown else {}

    sections_out: List[Dict[str, Any]] = []

    for sec in NIH_SECTIONS:
        sec_order = sec["order"]

        questions_out: List[Dict[str, Any]] = []
        for q_order, q_text in sec["questions"]:
            map_key = f"{sec_order}.{q_order}"

            # 1) prefer parsed markdown (so JSON matches the generated plan)
            answer = parsed.get(map_key, "")

            # 2) fallback to form input
            if not answer:
                answer = _first_nonempty(form_inputs, FIELD_MAP.get(map_key, []))

            questions_out.append(_question(q_order, q_text, answer, "textArea"))

        sections_out.append(
            {
                "order": sec_order,
                "title": sec["title"],
                "description": sec.get("description", ""),
                "question": questions_out,
            }
        )

    return {
        "dmptool": {
            "provenance": provenance,
            "narrative": {
                "title": _clean(template_title),
                "section": sections_out,
            },
        }
    }
