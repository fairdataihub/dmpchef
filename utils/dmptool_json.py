# utils/dmptool_json.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ----------------------------
# Basic helpers
# ----------------------------
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
# NIH template structure (ends at Element 6)
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
# Markdown parsing regexes
# ============================================================

# Element heading: forgiving
# Matches:
#   **Element 1: Title**
#   **Element 1 - Title**
#   ## Element 1: Title
#   Element 1: Title
_ELEMENT_HEADING_RE = re.compile(
    r"(?mi)^(?:\*\*|##\s*|#\s*)?\s*Element\s*(?P<num>[1-6])\s*[:\-]\s*(?P<title>.+?)\s*(?:\*\*)?\s*$"
)

# Subheading boundary (any heading level 2-4)
_SUBHEADING_RE = re.compile(r"(?m)^#{2,4}\s*(?P<h>.+?)\s*$")

# Markdown underline lines: ----- or =====
_SETEXT_UNDERLINE_RE = re.compile(r"(?m)^\s*[-=]{3,}\s*$")


# ============================================================
# Cleanup layers (make JSON answers cleaner)
# ============================================================
def _strip_setext_underlines(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    text = _SETEXT_UNDERLINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _strip_leading_instruction(text: str) -> str:
    """
    Remove common instruction blocks LLMs include, e.g.
    *State whether ...*
    State what common data standards ...
    Describe how compliance ...
    """
    text = _clean(text)
    if not text:
        return ""

    # Remove first italic paragraph if it looks like an instruction
    # (kept broad but bounded so we don't delete real content)
    text = re.sub(
        r"(?s)^\s*\*(?:state|describe|indicate|provide|explain)\b[^*]{10,600}\*\s*\n+",
        "",
        text,
        count=1,
    )

    # Remove first plain-text instruction line (non-italic)
    text = re.sub(
        r"(?mi)^\s*(state|describe|indicate|provide|explain)\b[^\n]{10,600}\n+",
        "",
        text,
        count=1,
    )

    return text.strip()


def _strip_footer_notes(text: str) -> str:
    """
    Cut off common footers:
      --- separator
      "Please note ..." disclaimers
      "I hope this ..." closings
    """
    text = _clean(text)
    if not text:
        return ""

    # Stop at horizontal rule
    parts = re.split(r"(?mi)^\s*---\s*$", text, maxsplit=1)
    text = parts[0]

    # Remove trailing "Please note ..." block
    text = re.split(r"(?mi)^\s*please note\b.*$", text, maxsplit=1)[0]

    # Remove trailing "I hope ..." closing
    text = re.split(r"(?mi)^\s*i hope\b.*$", text, maxsplit=1)[0]

    return text.strip()


# ✅ NEW: strip markdown artifacts that leak into JSON answers
_BULLET_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-+*•]\s+)")
_STANDALONE_STAR_RE = re.compile(r"(?m)^\s*\*\s*$")


def _strip_wrapping_italics(text: str) -> str:
    """
    If the whole answer is wrapped like *...* or **...**, remove only the wrapper.
    Fixes cases like: "*The proposed research aims ... files.*"
    """
    text = _clean(text)
    if not text:
        return ""

    for _ in range(3):
        t = text.strip()

        # unwrap **...**
        if len(t) >= 4 and t.startswith("**") and t.endswith("**"):
            inner = t[2:-2].strip()
            if inner:
                text = inner
                continue

        # unwrap *...*
        if len(t) >= 2 and t.startswith("*") and t.endswith("*"):
            inner = t[1:-1].strip()
            if inner:
                text = inner
                continue

        break

    return text


def _strip_inline_md_emphasis(text: str) -> str:
    """
    Remove remaining **bold** and *italic* markers (best-effort).
    """
    text = _clean(text)
    if not text:
        return ""

    # **bold** -> bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # *italic* -> italic (best-effort; does not try to be a full markdown parser)
    text = re.sub(r"(?s)\*(?!\s)([^*\n]+?)(?<!\s)\*", r"\1", text)

    return text


def _strip_bullet_prefixes(text: str) -> str:
    """
    Remove list bullet prefixes at start of lines.
    """
    text = _clean(text)
    if not text:
        return ""
    text = _BULLET_PREFIX_RE.sub("", text)
    text = _STANDALONE_STAR_RE.sub("", text)
    return text


def _collapse_whitespace(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _final_answer_clean(text: str) -> str:
    """
    Final standardized cleanup for any extracted answer.
    """
    text = _strip_setext_underlines(text)
    text = _strip_leading_instruction(text)
    text = _strip_footer_notes(text)

    # ✅ remove markdown artifacts
    text = _strip_wrapping_italics(text)
    text = _strip_inline_md_emphasis(text)
    text = _strip_bullet_prefixes(text)
    text = _collapse_whitespace(text)

    return _clean(text)


# ============================================================
# Parsing helpers
# ============================================================
def _slice_between(text: str, start_idx: int, end_idx: Optional[int]) -> str:
    if end_idx is None:
        return text[start_idx:]
    return text[start_idx:end_idx]


def _parse_elements(md: str) -> Dict[int, str]:
    """
    Returns {1: "<element1 body>", 2: "<element2 body>", ...}
    Body excludes the element heading line.
    """
    md = _clean(md)
    matches = list(_ELEMENT_HEADING_RE.finditer(md))
    out: Dict[int, str] = {}

    for i, m in enumerate(matches):
        num = int(m.group("num"))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else None
        body = _slice_between(md, body_start, body_end).strip()
        out[num] = _strip_setext_underlines(body)

    return out


def _extract_under_prompt(block: str, prompt_text: str) -> str:
    """
    Extract answer text under either:
      1) ##/###/#### prompt_text
      2) 1. **prompt_text:**   (colon inside bold)
      3) 1. **prompt_text**:   (colon after bold)

    Captures until the next heading or next bold-question line.
    """
    block = _strip_setext_underlines(block)
    if not block:
        return ""

    # 1) Heading form
    h_pat = re.compile(rf"(?mi)^#{{2,4}}\s*{re.escape(prompt_text)}\s*:?\s*$")

    # 2/3) numbered/bulleted bold form (colon inside OR after bold)
    b_pat = re.compile(
        rf"(?mi)^\s*(?:\d+\.\s+|[-*]\s+)\*\*{re.escape(prompt_text)}(?:\s*:)?\*\*\s*:?\s*$"
    )

    m = h_pat.search(block) or b_pat.search(block)
    if not m:
        return ""

    start = m.end()
    rest = block[start:]

    # Next boundary: heading or next bold-question line
    next_heading = re.search(r"(?mi)^#{2,4}\s+.+$", rest)
    next_boldq = re.search(
        r"(?mi)^\s*(?:\d+\.\s+|[-*]\s+)\*\*.+?(?:\s*:)?\*\*\s*:?\s*$",
        rest,
    )

    candidates: List[int] = []
    if next_heading:
        candidates.append(start + next_heading.start())
    if next_boldq:
        candidates.append(start + next_boldq.start())

    end = min(candidates) if candidates else None
    return _final_answer_clean(_slice_between(block, start, end).strip())


def _best_effort_element_body(block: str) -> str:
    """
    For elements without explicit sub-questions (2,3,6), return the whole body.
    """
    return _final_answer_clean(block)


def _parse_generated_markdown_to_answers(generated_markdown: str) -> Dict[str, str]:
    """
    Map generated markdown into answers for keys:
      1.1, 1.2, 1.3, 2.1, 3.1, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1
    """
    md = _clean(generated_markdown)
    elements = _parse_elements(md)

    answers: Dict[str, str] = {}

    # Element 1
    e1 = elements.get(1, "")
    answers["1.1"] = _extract_under_prompt(
        e1, "Types and amount of scientific data expected to be generated in the project"
    )
    answers["1.2"] = _extract_under_prompt(
        e1, "Scientific data that will be preserved and shared, and the rationale for doing so"
    )
    answers["1.3"] = _extract_under_prompt(e1, "Metadata, other relevant data, and associated documentation")

    # Element 2
    answers["2.1"] = _best_effort_element_body(elements.get(2, ""))

    # Element 3
    answers["3.1"] = _best_effort_element_body(elements.get(3, ""))

    # Element 4
    e4 = elements.get(4, "")
    answers["4.1"] = _extract_under_prompt(e4, "Repository where scientific data and metadata will be archived")
    answers["4.2"] = _extract_under_prompt(e4, "How scientific data will be findable and identifiable")
    answers["4.3"] = _extract_under_prompt(e4, "When and how long the scientific data will be made available")

    # Element 5
    e5 = elements.get(5, "")
    answers["5.1"] = _extract_under_prompt(
        e5, "Factors affecting subsequent access, distribution, or reuse of scientific data"
    )
    answers["5.2"] = _extract_under_prompt(e5, "Whether access to scientific data will be controlled")
    answers["5.3"] = _extract_under_prompt(
        e5, "Protections for privacy, rights, and confidentiality of human research participants"
    )

    # Element 6
    answers["6.1"] = _best_effort_element_body(elements.get(6, ""))

    # Final clean (important if anything slipped through)
    for k, v in list(answers.items()):
        answers[k] = _final_answer_clean(v)

    return answers


# ============================================================
# Public builder
# ============================================================
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
    """
    form_inputs = form_inputs or {}

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

            answer = parsed.get(map_key, "")
            if not answer:
                answer = _first_nonempty(form_inputs, FIELD_MAP.get(map_key, []))

            # ✅ ensure final clean even for form_inputs fallback
            answer = _final_answer_clean(answer)

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
